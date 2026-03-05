import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, timezone, timedelta
import yaml
import yfinance as yf
import logging
import json
import io
import base64
import re
from typing import Dict, Any, List, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s", 
    datefmt="%H:%M:%S"
)
log = logging.getLogger("DepotBot")

# ─── 1. Datenbeschaffung (Robust) ──────────────────────────────────────────────

class DataFetcher:
    """Holt Marktdaten via yfinance und behandelt Fehler robust."""
    
    def __init__(self):
        self.exchange_rate = 1.0
        self._fetch_exchange_rate()
        
    def _fetch_exchange_rate(self):
        """Holt den aktuellen EUR/USD Wechselkurs."""
        try:
            eur_usd = yf.Ticker("EURUSD=X")
            hist = eur_usd.history(period="1d")
            if not hist.empty:
                self.exchange_rate = hist['Close'].iloc[-1]
                log.info(f"💶 Aktueller EUR/USD Kurs: {self.exchange_rate:.4f}")
        except Exception as e:
            log.warning(f"Konnte EUR/USD Wechselkurs nicht abrufen, nutze 1.0: {e}")

    def get_market_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Holt historische Daten, Basisinfos, News und Kalender für alle Symbole."""
        log.info(f"📈 Hole Marktdaten für {len(symbols)} Symbole...")
        data = {}
        
        # 7 Tage zurück für die 1-Wochen-Performance
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

                if hist.empty:
                    log.warning(f"⚠️ Keine Historie für {symbol} gefunden.")
                    data[symbol] = {"error": "Keine Historie"}
                    continue
                
                # Preise
                start_price = hist['Close'].iloc[0]
                end_price = hist['Close'].iloc[-1]
                perf_pct = ((end_price - start_price) / start_price) * 100
                history_prices = hist['Close'].tolist()

                # Info Dictionary sicher parsen
                info = {}
                try:
                    info = ticker.info
                except Exception as e:
                    log.warning(f"⚠️ Konnte Info für {symbol} nicht abrufen: {e}")
                
                trailing_pe = info.get('trailingPE', 'N/A')
                target_price = info.get('targetMeanPrice', 'N/A')
                recommendation = info.get('recommendationKey', 'N/A')
                currency = info.get('currency', 'USD')
                is_usd = currency == 'USD'

                # Währung in Euro umrechnen (nur wenn in USD)
                sp_eur = start_price / self.exchange_rate if is_usd else start_price
                ep_eur = end_price / self.exchange_rate if is_usd else end_price

                # News sicher parsen (oft verschachtelt geändert bei Yahoo)
                news_headlines = []
                try:
                    raw_news = ticker.news
                    if raw_news:
                        for n in raw_news[:3]:
                            # Struktur ist manchmal n['title'], manchmal n['content']['title']
                            title = n.get('title')
                            if not title and 'content' in n:
                                title = n['content'].get('title')
                            
                            if title:
                                news_headlines.append(title)
                except Exception as e:
                    log.debug(f"News Parsing Fehler für {symbol} (übersprungen): {e}")

                # Dividenden-Ex-Date
                ex_div_date = None
                try:
                    raw_ex = info.get('exDividendDate')
                    if raw_ex:
                        ex_div_date = datetime.fromtimestamp(raw_ex).strftime('%d.%m.%Y')
                except Exception:
                    pass

                # Earnings
                earnings_date = None
                try:
                    cal = ticker.calendar
                    if isinstance(cal, dict) and 'Earnings Date' in cal:
                        ed_list = cal['Earnings Date']
                        if ed_list:
                            earnings_date = ed_list[0].strftime('%d.%m.%Y')
                except Exception:
                    pass

                data[symbol] = {
                    "start_price_eur": round(sp_eur, 2),
                    "current_price_eur": round(ep_eur, 2),
                    "performance_1w_pct": round(perf_pct, 2),
                    "history_prices": history_prices,
                    "trailing_pe": trailing_pe,
                    "target_price": target_price,
                    "analyst_rating": recommendation,
                    "news_headlines": news_headlines,
                    "ex_dividend_date": ex_div_date,
                    "earnings_date": earnings_date,
                    "error": None
                }
                
                sign = "+" if perf_pct > 0 else ""
                log.info(f"  ✅ {symbol}: {round(ep_eur, 2)} EUR ({sign}{round(perf_pct, 2)}%)")

            except Exception as e:
                log.error(f"❌ Schwerer Fehler bei {symbol}: {e}")
                data[symbol] = {"error": str(e)}

        return data

# ─── 2. Analyse Engine ─────────────────────────────────────────────────────────

class AnalyticsEngine:
    """Berechnet Depot-Gewichtung, Top/Flops und generiert Sparklines."""
    
    @staticmethod
    def generate_sparkline(prices: List[float], color: str = "#10b981") -> str:
        """Generiert eine kleine Verlaufsgrafik als Base64-String."""
        if not prices or len(prices) < 2:
            return ""
        try:
            fig, ax = plt.subplots(figsize=(2.2, 0.6))
            ax.plot(prices, color=color, linewidth=2.5)
            ax.fill_between(range(len(prices)), prices, alpha=0.15, color=color)
            ax.axis('off')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, transparent=True, bbox_inches='tight', pad_inches=0)
            buf.seek(0)
            plt.close(fig)

            encoded = base64.b64encode(buf.read()).decode('utf-8')
            return f"data:image/png;base64,{encoded}"
        except Exception as e:
            log.warning(f"Sparkline Exception: {e}")
            return ""

    @staticmethod
    def analyze(portfolio_config: List[Dict], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Bündelt alle Metriken für das Html-Template und die KI."""
        total_start_val = 0.0
        total_current_val = 0.0
        
        best = {"name": "", "symbol": "", "perf": -9999.0, "prices": [], "news": []}
        worst = {"name": "", "symbol": "", "perf": 9999.0, "prices": [], "news": []}
        
        calendar_events = []
        portfolio_details = []

        # Basis-Berechnungen (Top/Flop und Gesamtwert)
        for item in portfolio_config:
            sym = item["symbol"]
            d = market_data.get(sym)
            if not d or d.get("error"):
                continue

            shares = item.get("shares", 0)
            perf = d["performance_1w_pct"]
            
            # Hebel-Zertifikate manuell justieren falls OUST in config ist
            if sym == "OUST":
                perf = perf * 2
                
            if shares > 0:
                pos_start_val = d["start_price_eur"] * shares
                pos_current_val = d["current_price_eur"] * shares
                total_start_val += pos_start_val
                total_current_val += pos_current_val
                
            # Top/Flop check
            if perf > best["perf"]:
                best = {"name": item["name"], "symbol": sym, "perf": perf, "prices": d["history_prices"], "news": d["news_headlines"]}
            if perf < worst["perf"]:
                worst = {"name": item["name"], "symbol": sym, "perf": perf, "prices": d["history_prices"], "news": d["news_headlines"]}
                
            # Kalender - nur Events innerhalb der nächsten 7 Tage anzeigen
            now = datetime.now()
            in_7_days = now + timedelta(days=7)
            
            if d.get("ex_dividend_date"):
                try:
                    event_date = datetime.strptime(d["ex_dividend_date"], "%d.%m.%Y")
                    # Wenn das Event heute oder in den nächsten 7 Tagen liegt
                    if now.date() <= event_date.date() <= in_7_days.date():
                        calendar_events.append({"type": "💰 Ex-Div", "name": item["name"], "date": d["ex_dividend_date"], "date_obj": event_date})
                except ValueError:
                    pass
                    
            if d.get("earnings_date"):
                try:
                    event_date = datetime.strptime(d["earnings_date"], "%d.%m.%Y")
                    if now.date() <= event_date.date() <= in_7_days.date():
                        calendar_events.append({"type": "📊 Earnings", "name": item["name"], "date": d["earnings_date"], "date_obj": event_date})
                except ValueError:
                    pass
                
            portfolio_details.append({
                "item": item,
                "data": d,
                "adjusted_perf": perf  # Inklusive Hebel-Anpassung
            })

        # Portfolio-Rendite berechnen
        total_perf_pct = 0.0
        total_profit_eur = 0.0
        if total_start_val > 0:
            total_perf_pct = ((total_current_val - total_start_val) / total_start_val) * 100
            total_profit_eur = total_current_val - total_start_val

        # Gewichtung berechnen
        weights = []
        for p in portfolio_details:
            shares = p["item"].get("shares", 0)
            if shares > 0 and total_current_val > 0:
                pos_val = p["data"]["current_price_eur"] * shares
                w = (pos_val / total_current_val) * 100
                weights.append({"name": p["item"]["name"], "symbol": p["item"]["symbol"], "weight_pct": w})

        # Termine chronologisch sortieren
        calendar_events.sort(key=lambda x: x["date_obj"])
        # date_obj entfernen, da wir es im HTML nicht mehr brauchen
        for e in calendar_events:
            del e["date_obj"]

        return {
            "total_value": total_current_val,
            "total_perf_pct": total_perf_pct,
            "total_profit_eur": total_profit_eur,
            "best": best,
            "worst": worst,
            "weights": sorted(weights, key=lambda x: x["weight_pct"], reverse=True),
            "calendar_events": calendar_events,
            "portfolio_details": portfolio_details
        }

# ─── 3. Gemini KI Assistent ────────────────────────────────────────────────────

class AIAssistant:
    """Kommuniziert mit google.generativeai und sanitisiert Output explizit."""
    
    def __init__(self, api_key: str):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
    def _sanitize_html(self, raw_text: str) -> str:
        """Entfernt Markdown Codeblocks und gefährliche Tags."""
        # Entferne ```html ... ``` Markdown-Wrappers
        text = re.sub(r'```(?:html)?', '', raw_text).strip()
        # Entferne <html>, <body>, <head>
        text = re.sub(r'</?(html|body|head|DOCTYPE)[^>]*>', '', text, flags=re.IGNORECASE)
        # Ersetze potentiell gefährliche Script Tags
        text = re.sub(r'<script.*?>.*?</script>', '', text, flags=re.IGNORECASE|re.DOTALL)
        return text.strip()

    def analyze_portfolio(self, analysis_data: Dict) -> str:
        log.info("🧠 Gemini analysiert Depot-Performance...")
        
        # Portfolio String für Prompt bauen
        p_text = "MEIN DEPOT (1W Performance):\n"
        for p in analysis_data["portfolio_details"]:
            d = p["data"]
            i = p["item"]
            p_text += f"- {i['name']} ({i['symbol']}): {d['current_price_eur']} EUR | 1W: {p['adjusted_perf']:.2f}% | P/E: {d['trailing_pe']} | Anl.: {d['analyst_rating']}\n"
            
        w_text = "GEWICHTUNG:\n"
        for w in analysis_data["weights"][:5]: # Top 5
            w_text += f"- {w['name']}: {w['weight_pct']:.1f}%\n"

        news_text = "AKTUELLE NACHRICHTEN ZU GEWINNER/VERLIERER (Nutze das als Kontextverknüpfung!):\n"
        best = analysis_data["best"]
        worst = analysis_data["worst"]
        if best["news"]:
            news_text += f"- {best['name']} (Gewinner): {'; '.join(best['news'])}\n"
        if worst["news"]:
            news_text += f"- {worst['name']} (Verlierer): {'; '.join(worst['news'])}\n"

        prompt = f"""
Du bist ein knallharter, datengetriebener Portfolio-Analyst.
Analysiere die wöchentliche Performance meines Portfolios (auf Deutsch).

{p_text}

{w_text}

{news_text}

AUFGABE (3 kurze Absätze):
1. **Trend & Tops/Flops**: Erkläre (unter Einbezug der News) WARUM meine stärkste Position gestiegen und meine schwächste gefallen ist.
2. **Klumpenrisiko-Check**: Ist eine Position >25% gewichtet? Falls ja, sprich eine analytische Warnung zur Diversifikation aus. Falls nein, lobe die solide Verteilung.
3. **Action Items (Nur wenn nötig)**: Welche 1-2 Aktien sind derzeit fundamental gefährdet oder massiv überbewertet (Verkauf/Gewinnmitnahme) bzw. wo lohnt sich ein Rebuy? Überspringe halte-Kandidaten komplett. Falls keine Action nötig ist, schreibe: "Es gibt akut keinen fundamentalen Handlungsbedarf im Depot."

Regeln für die Ausgabe:
- Formatiere als reines HTML (nur <p>, <h3>, <ul>, <li>, <strong>).
- KEINE Markdown-Codeblöcke wie ```html drum herum.
- Sachlich, objektiv, keine emotionale 'Anlageberatung'.
"""
        try:
            response = self.model.generate_content(prompt)
            return self._sanitize_html(response.text)
        except Exception as e:
            log.error(f"❌ Gemini Portfolio Analyse Fehler: {e}")
            return "<p>KI-Analyse derzeit nicht verfügbar.</p>"

    def scout_opportunities(self, watchlist_config: List[Dict], watchlist_data: Dict) -> str:
        log.info("🎯 Gemini scoutet Watchlist und Sektoren...")
        
        # Umgehung von "Anlageberatung" Content-Filtern: wir fragen explizit nach Unternehmensprofilen!
        w_text = "BEOBACHTUNGSLISTE:\n"
        for item in watchlist_config:
            sym = item["symbol"]
            d = watchlist_data.get(sym)
            if d and not d.get("error"):
                w_text += f"- {item['name']}: Preis {d['current_price_eur']}€ | P/E: {d['trailing_pe']} | Anl.: {d['analyst_rating']}\n"

        prompt = f"""
Du bist ein quantitativer Analyst für fundamental starke Unternehmen. Das ist keine Anlageberatung, sondern eine objektive Fundamentaldatenanalyse.

{w_text}

AUFGABE:
Stelle mir 2-3 Aktien vor, die aktuell aufgrund ihrer Fundamentaldaten (Günstige Bewertung, Wachstumskatalysator, Margen) extrem interessant sind. 
Du kannst Aktien aus meiner Beobachtungsliste nutzen ODER eigene fundamental starke Mid-Cap/Nischen-Hits bringen (z.B. Cybersecurity, Medizintechnik, Industrials).

Für jedes Unternehmen (HTML formatiert):
- <h3>Name (Symbol)</h3>
- <p><strong>Der fundamentale Katalysator:</strong> Warum ist das Unternehmen in der jetzigen Marktlage spannend? (Umsatzwachstum, Marktposition, P/E-Ratio).</p>

Regeln für die Ausgabe:
- Formatiere als reines HTML (nur <p>, <h3>, <ul>, <li>, <strong>).
- KEINE Markdown-Codeblöcke wie ```html drum herum.
"""
        try:
            response = self.model.generate_content(prompt)
            return self._sanitize_html(response.text)
        except Exception as e:
            log.error(f"❌ Gemini Scouting Fehler: {e}")
            return "<p>Scouting derzeit nicht verfügbar.</p>"


# ─── 4. Email Builder ──────────────────────────────────────────────────────────

class EmailBuilder:
    """Erstellt das HTML für die Mail (Design ähnlich NewsBot)."""

    @staticmethod
    def build_html(analysis_data: Dict, ai_portfolio_html: str, ai_opportunities_html: str) -> str:
        now = datetime.now(timezone(timedelta(hours=1)))
        
        # 1. Sparklines abrufen
        best = analysis_data["best"]
        worst = analysis_data["worst"]
        best_img = AnalyticsEngine.generate_sparkline(best["prices"], "#10b981")
        worst_img = AnalyticsEngine.generate_sparkline(worst["prices"], "#ef4444")
        
        best_img_html = f'<br/><img src="{best_img}" width="120" style="margin-top:4px;" />' if best_img else ''
        worst_img_html = f'<br/><img src="{worst_img}" width="120" style="margin-top:4px;" />' if worst_img else ''

        # 2. Kennzahlen
        perf_pct = analysis_data["total_perf_pct"]
        profit_eur = analysis_data["total_profit_eur"]
        sign = "+" if perf_pct > 0 else ""
        color = "#10b981" if perf_pct > 0 else "#ef4444"

        # 3. Kalender HTML
        cal_html = ""
        if analysis_data["calendar_events"]:
            cal_items = [f"<li><strong>{e['type']}</strong>: {e['name']} ({e['date']})</li>" for e in analysis_data["calendar_events"]]
            cal_html = f"""
            <div style="margin-top:20px;padding-top:15px;border-top:1px dashed #cbd5e1;">
                <h4 style="margin:0 0 10px 0;font-size:14px;color:#475569;">📅 Anstehende Termine:</h4>
                <ul style="margin:0;padding-left:20px;font-size:13px;color:#475569;">
                    {''.join(cal_items)}
                </ul>
            </div>
            """

        # 4. Das Haupt-Template (Angelehnt an den NewsBot Newsletter)
        html = f"""<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" lang="de">
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Dein Weekly Depot Report</title>
</head>
<body style="margin:0;padding:0;background-color:#f8fafc;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;">
<table width="100%" border="0" cellspacing="0" cellpadding="0" bgcolor="#f8fafc">
    <tr>
        <td align="center" style="padding:40px 15px;">
            <table width="100%" max-width="650" border="0" cellspacing="0" cellpadding="0" bgcolor="#ffffff" style="max-width:650px;border-radius:12px;overflow:hidden;box-shadow:0 4px 15px rgba(0,0,0,0.05);border:1px solid #e2e8f0;">
                
                <!-- HEADER -->
                <tr>
                    <td bgcolor="#1e293b" style="padding:40px 30px;background-color:#1e293b;text-align:center;">
                        <h1 style="margin:0 0 5px 0;color:#ffffff;font-size:28px;letter-spacing:-0.5px;">📈 Depot Weekly</h1>
                        <p style="margin:0;color:#94a3b8;font-size:16px;">Wochenanalyse vom {now.strftime('%d.%m.%Y')}</p>
                    </td>
                </tr>

                <!-- PERFORMANCE SUMMARY BOX -->
                <tr>
                    <td style="padding:35px 30px 15px 30px;">
                        <div style="background-color:#f1f5f9;border:1px solid #e2e8f0;border-radius:8px;padding:25px;">
                            <h3 style="margin:0 0 15px 0;font-size:14px;color:#64748b;text-transform:uppercase;letter-spacing:1px;">1-Wochen-Entwicklung</h3>
                            <p style="margin:0 0 20px 0;font-size:24px;color:{color};font-weight:bold;">{sign}{perf_pct:.2f}% <span style="font-size:16px;color:#64748b;font-weight:normal;">({sign}{profit_eur:.2f} €)</span></p>

                            <table width="100%" border="0" cellspacing="0" cellpadding="0">
                                <tr>
                                    <td width="50%" valign="top">
                                        <p style="margin:0;font-size:14px;color:#475569;"><strong>🚀 Top der Woche:</strong><br/>{best['name']} <span style="color:#10b981;">(+{best['perf']:.2f}%)</span>{best_img_html}</p>
                                    </td>
                                    <td width="50%" valign="top">
                                        <p style="margin:0;font-size:14px;color:#475569;"><strong>🔻 Flop der Woche:</strong><br/>{worst['name']} <span style="color:#ef4444;">({worst['perf']:.2f}%)</span>{worst_img_html}</p>
                                    </td>
                                </tr>
                            </table>
                            {cal_html}
                        </div>
                    </td>
                </tr>

                <!-- KI PORTFOLIO ANALYSE -->
                <tr>
                    <td style="padding:15px 30px;">
                        <h2 style="margin:0;color:#4f46e5;font-size:13px;text-transform:uppercase;letter-spacing:1px;border-bottom:2px solid #e0e7ff;padding-bottom:10px;">🧠 KI Depot-Analyse</h2>
                        <div style="padding-top:15px;font-size:15px;color:#334155;line-height:1.6;">
                            {ai_portfolio_html}
                        </div>
                    </td>
                </tr>

                <!-- KI OPPORTUNITIES -->
                <tr>
                    <td style="padding:20px 30px 40px 30px;">
                        <h2 style="margin:0;color:#10b981;font-size:13px;text-transform:uppercase;letter-spacing:1px;border-bottom:2px solid #d1fae5;padding-bottom:10px;">🎯 Fundamentale Kaufchancen</h2>
                        <div style="padding-top:15px;font-size:15px;color:#334155;line-height:1.6;">
                            {ai_opportunities_html}
                        </div>
                    </td>
                </tr>

                <!-- FOOTER -->
                <tr>
                    <td bgcolor="#f8fafc" style="padding:25px 30px;border-top:1px solid #e2e8f0;text-align:center;">
                        <p style="margin:0 0 5px 0;font-size:12px;color:#94a3b8;">Generiert von <strong>DepotBot Analyst</strong> (yfinance + Gemini Flash 2.5)</p>
                        <p style="margin:0;font-size:11px;color:#cbd5e1;">Dies ist keine professionelle Anlageberatung.</p>
                    </td>
                </tr>
            </table>
        </td>
    </tr>
</table>
</body>
</html>"""
        return html


def send_email(html_content: str, email_address: str, email_password: str, email_recipient: str, smtp_server="smtp.gmail.com", smtp_port=587):
    """Versendet die HTML-E-Mail über SMTP."""
    now = datetime.now(timezone(timedelta(hours=1)))
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"📈 Depot Weekly - {now.strftime('%d.%m.%Y')}"
    msg["From"] = f"DepotBot Analyst <{email_address}>"
    msg["To"] = email_recipient
    
    html_part = MIMEText(html_content, "html", "utf-8")
    msg.attach(html_part)

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(email_address, email_password)
            server.send_message(msg)
        log.info(f"✅ Depot Review erfolgreich an {email_recipient} gesendet.")
        return True
    except Exception as e:
        log.error(f"❌ Fehler beim E-Mail-Versand: {e}")
        return False


# ─── 5. Hauptprogramm ─────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Depot Analyzer")
    parser.add_argument("--dry-run", action="store_true", help="Nur lokale Speicherung als HTML, kein E-Mail Versand")
    args = parser.parse_args()

    log.info("🚀 Depot Analyzer (V3) startet...")

    # Config parsen
    try:
        with open("portfolio.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            portfolio_items = config.get("portfolio", [])
            watchlist_items = config.get("watchlist", [])
    except Exception as e:
        log.error(f"❌ Konnte portfolio.yaml nicht lesen: {e}")
        return

    # API Keys holen
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if not gemini_key:
        log.warning("⚠️ GEMINI_API_KEY nicht gesetzt. KI-Generierung kann fehlschlagen.")

    # 1. Daten holen
    fetcher = DataFetcher()
    portfolio_symbols = [p["symbol"] for p in portfolio_items]
    watchlist_symbols = [w["symbol"] for w in watchlist_items]
    
    portfolio_data = fetcher.get_market_data(portfolio_symbols)
    watchlist_data = fetcher.get_market_data(watchlist_symbols)

    # 2. Metriken, Top/Flop, Gewichtungen berechnen
    analysis_data = AnalyticsEngine.analyze(portfolio_items, portfolio_data)

    # 3. KI Analysen (Nur wenn Keys existieren, ansonsten Fallbacks)
    portfolio_html = "<p>KI-Analyse übersprungen (Kein API-Key).</p>"
    opportunities_html = "<p>Scouting übersprungen (Kein API-Key).</p>"
    
    if gemini_key:
        ai = AIAssistant(api_key=gemini_key)
        portfolio_html = ai.analyze_portfolio(analysis_data)
        opportunities_html = ai.scout_opportunities(watchlist_items, watchlist_data)

    # 4. E-Mail HTML Bauen
    final_email_html = EmailBuilder.build_html(analysis_data, portfolio_html, opportunities_html)

    # Lokales Backup speichern
    with open("depot_report.html", "w", encoding="utf-8") as f:
        f.write(final_email_html)
    log.info("💾 Report lokal als 'depot_report.html' gespeichert.")

    # 5. Versand
    if args.dry_run:
        log.info("🏃 Dry-Run beendet. Keine E-Mail gesendet.")
    else:
        e_addr = os.environ.get("EMAIL_ADDRESS", "")
        e_pass = os.environ.get("EMAIL_PASSWORD", "")
        e_rec = os.environ.get("EMAIL_RECIPIENT", "")
        
        if all([e_addr, e_pass, e_rec]):
            send_email(final_email_html, e_addr, e_pass, e_rec)
        else:
            log.warning("⚠️ E-Mail Credentials fehlen, überspringe Versand.")

if __name__ == "__main__":
    main()
