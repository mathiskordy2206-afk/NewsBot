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
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg') # Headless mode für Serverbereich

# Logging konfigurieren
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("DepotBot")

# ─── 1. Datenbeschaffung ──────────────────────────────────────────────────────

def load_portfolio():
    """Lädt die Portfolio-Konfiguration."""
    try:
        with open("portfolio.yaml", "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    except Exception as e:
        log.error(f"Fehler beim Laden der portfolio.yaml: {e}")
        return {"portfolio": [], "watchlist": []}

def fetch_market_data(symbols):
    """Holt die 1-Wochen-Performance für die übergebenen Symbole."""
    log.info(f"📈 Hole Marktdaten für {len(symbols)} Symbole...")
    data = {}
    
    # 5 Tage + Wochenende = 1 Woche zurück
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    # Real-time Wechselkurs abrufen
    exchange_rate = 1.0
    try:
        eur_usd = yf.Ticker("EURUSD=X")
        exchange_rate_data = eur_usd.history(period="1d")
        if not exchange_rate_data.empty:
            exchange_rate = exchange_rate_data['Close'].iloc[-1]
            log.info(f"💶 Aktueller EUR/USD Kurs: {exchange_rate}")
    except Exception as e:
        log.warning(f"Konnte Wechselkurs nicht abrufen, nutze 1.0: {e}")
        
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
            
            if not hist.empty:
                start_price = hist['Close'].iloc[0]
                end_price = hist['Close'].iloc[-1]
                perf_pct = ((end_price - start_price) / start_price) * 100
                
                # Basiswissen für AI
                info = ticker.info
                trailing_pe = info.get('trailingPE', 'N/A')
                forward_pe = info.get('forwardPE', 'N/A')
                target_price = info.get('targetMeanPrice', 'N/A')
                recommendation = info.get('recommendationKey', 'N/A')
                ex_dividend_date = info.get('exDividendDate', None)
                if ex_dividend_date:
                    ex_dividend_date = datetime.fromtimestamp(ex_dividend_date).strftime('%d.%m.%Y')
                
                # News abrufen (Top 3 Headlines)
                news = ticker.news
                news_headlines = [n.get('title', 'News Headline') for n in news[:3]] if news else []
                
                
                # Währung prüfen und in Euro umrechnen (yfinance liefert meist USD für US-Werte)
                currency = info.get('currency', 'USD')
                is_usd = currency == 'USD'
                
                sp_eur = start_price / exchange_rate if is_usd else start_price
                ep_eur = end_price / exchange_rate if is_usd else end_price
                
                # Historische Preise für Sparkline extrahieren
                history_prices = hist['Close'].tolist()
                
                data[symbol] = {
                    "start_price": round(sp_eur, 2),
                    "current_price": round(ep_eur, 2),
                    "performance_1w_pct": round(perf_pct, 2),
                    "history_prices": history_prices,
                    "trailing_pe": trailing_pe,
                    "target_price": target_price,
                    "analyst_rating": recommendation,
                    "ex_dividend_date": ex_dividend_date,
                    "news_headlines": news_headlines
                }
            else:
                log.warning(f"Keine Historie für {symbol} gefunden.")
                data[symbol] = {"error": "Keine Daten"}
                
        except Exception as e:
            log.error(f"Fehler bei {symbol}: {e}")
            data[symbol] = {"error": str(e)}
            
    return data

def generate_sparkline(symbol, prices, color="#10b981"):
    """Generiert einen winzigen 1-Wochen Chart (Sparkline) als Base64-PNG."""
    if not prices or len(prices) < 2:
        return ""
    
    fig, ax = plt.subplots(figsize=(1.5, 0.4))
    ax.plot(prices, color=color, linewidth=2)
    ax.axis('off')
    
    # Ränder entfernen
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, transparent=True)
    buf.seek(0)
    plt.close(fig)
    
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{encoded}"

# ─── 2. KI-Analyse ────────────────────────────────────────────────────────────

def call_gemini(prompt: str, api_key: str) -> str:
    """Ruft die Gemini API auf (google.generativeai SDK)."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        # System instructions parameter is supported in newer versions,
        # but prepend to prompt for compatibility
        full_prompt = "Du bist ein hochkarätiger, professioneller Portfolio-Manager und Investment-Analyst. Du erklärst komplexe Marktbewegungen präzise, quantitativ belegt und dennoch verständlich.\n\n" + prompt
        
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        log.error(f"❌ Gemini API Fehler: {e}")
        return ""

def analyze_portfolio(portfolio_data, config, api_key):
    """Lässt Gemini das Portfolio analysieren und baut eine Übersicht."""
    log.info("🧠 Gemini analysiert das Portfolio...")
    
    total_start_value = 0.0
    total_current_value = 0.0
    has_shares = False
    
    best_stock = {"name": "N/A", "perf": -9999.0}
    worst_stock = {"name": "N/A", "perf": 9999.0}
    
    portfolio_text = "MEIN DEPOT (Performance der letzten 7 Tage):\n"
    dividend_events = []
    
    for item in config.get("portfolio", []):
        sym = item["symbol"]
        if sym in portfolio_data and "current_price" in portfolio_data[sym] and "start_price" in portfolio_data[sym]:
            d = portfolio_data[sym]
            buy_in = item.get("buy_in", "N/A")
            shares = item.get("shares", 0)
            
            # Dividendenkalender check
            if d.get("ex_dividend_date"):
                dividend_events.append(f"<li>{item['name']} ({sym}): <b>{d['ex_dividend_date']}</b></li>")
            
            # Hebel-Logik für Morgan Stanley Zertifikat (Ouster 2x Long)
            perf = d['performance_1w_pct']
            if sym == "OUST":
                perf = perf * 2  # 2x Hebel
                
            if perf > best_stock["perf"]:
                best_stock = {"name": item["name"], "symbol": sym, "perf": perf, "prices": d["history_prices"], "news": d["news_headlines"]}
            if perf < worst_stock["perf"]:
                worst_stock = {"name": item["name"], "symbol": sym, "perf": perf, "prices": d["history_prices"], "news": d["news_headlines"]}
                
            # Wöchentliche Performance für das Gesamtdepot berechnen (schon in EUR)
            if shares > 0:
                has_shares = True
                total_current_value += d['current_price'] * float(shares)
                total_start_value += d['start_price'] * float(shares)
            
            sign = "+" if perf > 0 else ""
            portfolio_text += f"- {item['name']} ({sym}): Letzter Preis {d['current_price']} EUR | 1-Wochen-Perf: {sign}{round(perf, 2)}% | KGV: {d['trailing_pe']} | Analysten: {d['analyst_rating']}\n"
            
            if buy_in != "N/A":
                # Gesamtperformance in Prozent
                total_perf = round(((d['current_price'] - float(buy_in)) / float(buy_in)) * 100, 2)
                if sym == "OUST":
                    total_perf = total_perf * 2  # 2x Hebel
                
                total_sign = "+" if total_perf > 0 else ""
                
                # Wenn wir die Stückzahl kennen, absoluten Gewinn in Euro/Dollar berechnen
                if shares > 0:
                    current_value = d['current_price'] * float(shares)
                    buy_value = float(buy_in) * float(shares)
                    profit = round(current_value - buy_value, 2)
                    
                    if sym == "OUST":
                        # Bei Derivaten ist der Wert des Scheins nicht = Kurs der Aktie * 100
                        # Der Nutzer sagte: 100 Stück = 528€ aktuell (also 5.28€/Stk)
                        pass # Für das Zertifikat belassen wir es bei der % Rechnung
                    else:
                        profit_sign = "+" if profit > 0 else ""
                        portfolio_text += f"  (Gesamt-Performance seit Kauf: {total_sign}{total_perf}% -> G/V: {profit_sign}{profit} EUR)\n"
                else:
                    portfolio_text += f"  (Gesamt-Performance seit Kauf bei {buy_in}: {total_sign}{total_perf}%)\n"

    # Portfolio Klumpenrisiko berechnen
    weight_text = ""
    if has_shares and total_current_value > 0:
        weight_text = "PORTFOLIO GEWICHTUNG (Zur Beurteilung von Klumpenrisiken):\n"
        for item in config.get("portfolio", []):
            sym = item["symbol"]
            if sym in portfolio_data and item.get("shares", 0) > 0:
                d = portfolio_data[sym]
                pos_value = d['current_price'] * float(item["shares"])
                weight = (pos_value / total_current_value) * 100
                weight_text += f"- {item['name']}: {round(weight, 1)}% vom Depot\n"

    # HTML Summary generieren
    summary_html = ""
    if has_shares and total_start_value > 0:
        total_perf_pct = ((total_current_value - total_start_value) / total_start_value) * 100
        total_profit = total_current_value - total_start_value
        sign = "+" if total_profit > 0 else ""
        color = "#10b981" if total_profit > 0 else "#ef4444"
        
        # Sparklines generieren
        best_img = generate_sparkline(best_stock["symbol"], best_stock.get("prices", []), color="#10b981")
        worst_img = generate_sparkline(worst_stock["symbol"], worst_stock.get("prices", []), color="#ef4444")
        
        best_img_html = f'<br><img src="{best_img}" style="width:120px;height:auto;margin-top:4px;" />' if best_img else ''
        worst_img_html = f'<br><img src="{worst_img}" style="width:120px;height:auto;margin-top:4px;" />' if worst_img else ''
        
        div_calendar_html = ""
        if dividend_events:
            div_calendar_html = f"<div style='margin-top:15px;padding-top:10px;border-top:1px dashed #cbd5e1;'><p style='margin:0 0 5px 0;font-size:14px;color:#475569;'><strong>📅 Nächste Ex-Dividenden / Termine:</strong></p><ul style='margin:0;padding-left:20px;font-size:13px;color:#64748b;'>" + "".join(dividend_events) + "</ul></div>"
        
        summary_html = f"""
        <div style="background-color:#ffffff;border:1px solid #e2e8f0;border-radius:8px;padding:20px;margin-bottom:25px;box-shadow:0 2px 4px rgba(0,0,0,0.02);">
            <h3 style="margin:0 0 15px 0;font-size:15px;color:#0f172a;text-transform:uppercase;letter-spacing:1px;">Wochen-Überblick</h3>
            <p style="margin:0 0 12px 0;font-size:15px;color:#475569;"><strong>Gesamt-Entwicklung (1W):</strong> <span style="color:{color};font-weight:bold;font-size:16px;">{sign}{round(total_perf_pct, 2)}% ({sign}{round(total_profit, 2)} €)</span></p>
            
            <table width="100%" border="0" cellspacing="0" cellpadding="0">
                <tr>
                    <td width="50%" valign="top">
                        <p style="margin:0 0 8px 0;font-size:14px;color:#475569;"><strong>🚀 Top der Woche:</strong><br>{best_stock['name']} <span style="color:#10b981;">(+{round(best_stock['perf'], 2)}%)</span>{best_img_html}</p>
                    </td>
                    <td width="50%" valign="top">
                        <p style="margin:0;font-size:14px;color:#475569;"><strong>🔻 Flop der Woche:</strong><br>{worst_stock['name']} <span style="color:#ef4444;">({round(worst_stock['perf'], 2)}%)</span>{worst_img_html}</p>
                    </td>
                </tr>
            </table>
            
            {div_calendar_html}
        </div>
        """

    # News-Context für Top/Flop zusammenbauen
    news_context = f"""
AKTUELLE NACHRICHTEN ZU TOP/FLOP (Nutze diese als Kontext für die KI-Erklärung!):
News zu {best_stock['name']} (Top): {', '.join(best_stock.get("news", []))}
News zu {worst_stock['name']} (Flop): {', '.join(worst_stock.get("news", []))}
"""

    prompt = f"""Analysiere die Performance meines Aktiendepots für die letzte Woche auf Deutsch.

{portfolio_text}

{weight_text}

{news_context}

AUFGABE:
1. Erkläre in 1-2 kurzen Absätzen den generellen Markttrend dieser Woche. Binde die "Top"- und "Flop"-News ein, um faktenbasiert zu erklären, *warum* diese Unternehmen gestiegen/gefallen sind.
2. KLUMPENRISIKO: Beurteile kurz meine Depotgewichtung. Warne mich objektiv, wenn ein Sektor oder eine Aktie zu stark gewichtet ist (>20%).
3. HANDLUNGSEMPFEHLUNGEN (WICHTIG!): Gehe meine Aktien durch, aber erwähne NUR die Aktien, bei denen ich aktuell VORSICHTIG sein sollte, die ich VERKAUFEN sollte (z.B. Gewinnmitnahmen/Bewertung zu hoch) oder bei denen ich NACHKAUFEN sollte. 
Gib zu diesen handverlesenen Titeln eine knappe Begründung.
-> IGNORIERE alle Aktien komplett, bei denen die Empfehlung ohnehin nur "Halten" lautet. Zeige mir ausschließlich die "Action Items"!
-> FALLS alle Aktien auf "Halten" stehen und es keine Action Items gibt, schreibe zwingend: "<p>Aktuell gibt es bei deinen Einzelpositionen keinen akuten Handlungsbedarf, alle Positionen können solide gehalten werden.</p>"

Formatiere dein Ergebnis als reines HTML-Snippet.
WICHTIG: Nutze NUR <h3>, <p>, <ul>, <li> Tags! Verbotene Tags: <html>, <head>, <body>, ```html !
Nutze maximal simples Inline-Styling falls etwas hervorgehoben werden soll (zB <strong style="color: green">).
"""
    ai_html = call_gemini(prompt, api_key)
    return summary_html + "\n" + ai_html

def scout_opportunities(watchlist_data, config, api_key):
    """Lässt Gemini neue, unentdeckte Kauftipps generieren."""
    log.info("🎯 Gemini sucht nach neuen Kaufchancen...")
    
    watch_text = "MEINE WATCHLIST:\n"
    for item in config.get("watchlist", []):
        sym = item["symbol"]
        if sym in watchlist_data and "current_price" in watchlist_data[sym]:
            d = watchlist_data[sym]
            watch_text += f"- {item['name']} ({sym}): Preis {d['current_price']} EUR | 1W-Perf: {d['performance_1w_pct']}% | KGV: {d['trailing_pe']} | Analysten: {d['analyst_rating']}\n"

    prompt = f"""Du bist ein Analyst für Hidden Champions und Value/Growth-Aktien.

{watch_text}

AUFGABE:
Identifiziere 3 Aktien, die *jetzt* ein hervorragendes Kaufpotenzial (Buy-Rating) aufweisen.
Das können Aktien von meiner Watchlist sein, MÜSSEN aber nicht. Ich möchte ausdrücklich auch 1-2 *nischenhafte*, fundamental starke Aktien oder spannende Sektoren-Hits präsentiert bekommen, die nicht jeder auf dem Radar (wie Apple/Nvidia) hat.

Für jede der 3 Empfehlungen:
- Nenne Name und Symbol
- Erkläre knackig den fundamentalen Katalysator (Warum jetzt attraktiv? Unterbewertet? Starkes Wachstum? Nischen-Burggraben?)

Formatiere dein Ergebnis als reines HTML-Snippet.
WICHTIG: Nutze NUR <h3>, <p>, <ul>, <li> Tags! Verbotene Tags: <html>, <head>, <body>, ```html !
"""
    return call_gemini(prompt, api_key)

# ─── 3. E-Mail Versand ────────────────────────────────────────────────────────

def build_email_html(portfolio_html, opportunities_html):
    """Baut eine professionelle HTML-E-Mail zusammen."""
    now = datetime.now(timezone(timedelta(hours=1)))
    date_str = now.strftime("%d.%m.%Y")
    
    html = f"""<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" lang="de">
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
<title>Weekly Depot Report</title>
</head>
<body style="margin:0;padding:0;background-color:#f1f5f9;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;">
<table width="100%" border="0" cellspacing="0" cellpadding="0" bgcolor="#f1f5f9">
    <tr>
        <td align="center" style="padding:40px 15px;">
            <table width="100%" max-width="650" border="0" cellspacing="0" cellpadding="0" bgcolor="#ffffff" style="max-width:650px;border-radius:12px;overflow:hidden;box-shadow:0 4px 15px rgba(0,0,0,0.05);border:1px solid #e2e8f0;">
                
                <!-- HEADER -->
                <tr>
                    <td bgcolor="#0f172a" style="padding:40px 30px;background-color:#0f172a;">
                        <h1 style="margin:0 0 5px 0;color:#ffffff;font-size:28px;letter-spacing:-0.5px;">📈 Weekly Depot Review</h1>
                        <p style="margin:0;color:#94a3b8;font-size:16px;">Wochenanalyse vom {date_str}</p>
                    </td>
                </tr>
                
                <!-- PORTFOLIO ANALYSIS -->
                <tr>
                    <td style="padding:35px 30px 15px 30px;">
                        <h2 style="margin:0;color:#0ea5e9;font-size:13px;text-transform:uppercase;letter-spacing:1.5px;border-bottom:2px solid #e0f2fe;padding-bottom:10px;">📊 Deine Depot-Performance</h2>
                    </td>
                </tr>
                <tr>
                    <td style="padding:0 30px 15px 30px;font-size:15px;color:#334155;line-height:1.6;">
                        {portfolio_html}
                    </td>
                </tr>
                
                <!-- NEW OPPORTUNITIES -->
                <tr>
                    <td style="padding:35px 30px 15px 30px;">
                        <h2 style="margin:0;color:#10b981;font-size:13px;text-transform:uppercase;letter-spacing:1.5px;border-bottom:2px solid #d1fae5;padding-bottom:10px;">🎯 Scouting: Starke Kaufchancen</h2>
                    </td>
                </tr>
                <tr>
                    <td style="padding:0 30px 40px 30px;font-size:15px;color:#334155;line-height:1.6;">
                        <div style="background-color:#f0fdf4;border-left:4px solid #10b981;padding:20px;border-radius:0 8px 8px 0;">
                            {opportunities_html}
                        </div>
                    </td>
                </tr>
                
                <!-- FOOTER -->
                <tr>
                    <td bgcolor="#f8fafc" style="padding:25px 30px;border-top:1px solid #e2e8f0;text-align:center;">
                        <p style="margin:0;font-size:12px;color:#94a3b8;">
                            Powered by DepotBot &bull; Gemini 2.5 Flash &bull; yfinance API
                        </p>
                    </td>
                </tr>
            </table>
        </td>
    </tr>
</table>
</body>
</html>"""
    return html

def send_email(html_content, smtp_server="smtp.gmail.com", smtp_port=587):
    """Versendet die HTML-E-Mail."""
    email_address = os.environ.get("EMAIL_ADDRESS", "")
    email_password = os.environ.get("EMAIL_PASSWORD", "")
    email_recipient = os.environ.get("EMAIL_RECIPIENT", "")

    if not all([email_address, email_password, email_recipient]):
        log.warning("⚠️ E-Mail-Konfiguration unvollständig, überspringe E-Mail-Versand")
        return False

    now = datetime.now(timezone(timedelta(hours=1)))
    
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Depot Review - {now.strftime('%d.%m.%Y')}"
    msg["From"] = f"DepotBot <{email_address}>"
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

# ─── 4. Main Workflow ─────────────────────────────────────────────────────────

def main():
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    
    if not gemini_key:
        log.error("❌ GEMINI_API_KEY ist nicht gesetzt!")
        return

    config = load_portfolio()
    
    # Symbole sammeln
    portfolio_symbols = [item["symbol"] for item in config.get("portfolio", [])]
    watchlist_symbols = [item["symbol"] for item in config.get("watchlist", [])]
    
    # Marktdaten holen
    portfolio_data = fetch_market_data(portfolio_symbols)
    watchlist_data = fetch_market_data(watchlist_symbols)
    
    # KI Analyse
    portfolio_html = analyze_portfolio(portfolio_data, config, gemini_key)
    opportunities_html = scout_opportunities(watchlist_data, config, gemini_key)
    
    # HTML bereinigen (falls Gemini den Codeblock-Markdown mitschickt)
    portfolio_html = portfolio_html.replace("```html", "").replace("```", "").strip()
    opportunities_html = opportunities_html.replace("```html", "").replace("```", "").strip()
    
    # Email generieren und senden
    final_email_html = build_email_html(portfolio_html, opportunities_html)
    
    # Für manuelles Testen lokal speichern
    with open("depot_report.html", "w", encoding="utf-8") as f:
        f.write(final_email_html)
        
    send_email(final_email_html)

if __name__ == "__main__":
    main()
