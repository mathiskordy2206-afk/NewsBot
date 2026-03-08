import os
import sys
import json
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, timezone, timedelta
import yfinance as yf
import google.generativeai as genai

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("DeepResearch")

def get_gemini_model(api_key: str):
    genai.configure(api_key=api_key)
    # Using the flash model as requested in previous parts of the project
    return genai.GenerativeModel('gemini-2.5-flash')

def identify_candidates(model) -> list:
    log.info("\n🔍 [Schritt 1] KI durchläuft einen internen Brainstorming- und Evaluierungsprozess für nischige Spitzenkandidaten...\n")
    prompt = """Du bist ein brillanter, unkonventioneller Aktien-Analyst für "Pick-and-Shovel" (Schaufelverkäufer) Unternehmen aus der zweiten und dritten Reihe.
Deine Aufgabe ist es, exakt 3 Unternehmen zu finden, die aktuell als essenzielle, aber oft übersehene B2B-Infrastruktur-Player, Zulieferer oder Dienstleister fungieren, bei denen ein unmittelbares und reales Upside-Potenzial auf absehbare Zeit (6-18 Monate) besteht.

WICHTIGE REGELN:
1. VERBOTENE MEGA-CAPS: Absolutes Veto gegen große Techwerte! Verbote beinhalten (aber nicht nur) NVDA, TSLA, AAPL, MSFT, AMZN, META, GOOGL, AMD, AVGO.
2. DYNAMISCHE STRATEGIE (SEHR WICHTIG): Wähle NICHT immer die gleichen Aktien wie ASML, SNPS, VRT oder SMCI. Gehe in deinem "Kopf" ganz bewusst aktuelle, unterrepräsentierte Nischenthemen durch (z.B. Automatisierungskomponenten, Spezialchemikalien für Batterien, Wasserinfrastruktur in Rechenzentren, Cybersecurity-Nischen, Nischen-Kupfer-Produzenten, hochspezialisierte Medizin-Dienstleister etc.) und picke dir dort gezielt Player heraus.
3. QUALITÄT VOR BEKANNTHEIT: Führe einen internen Screening-Prozess durch. Denke an 10 Mid-Cap-Kandidaten, sortiere die schwachen aus und präsentiere mir NUR die besten 3, die FUNDAMENTAL und REALISTISCH aktuell hervorragende Einstiegschancen (gute Balance aus Wachstum und passabler Bewertung) aufweisen.

Gib als Antwort AUSSCHLIESSLICH ein valides JSON-Array mit den amerikanischen Tickersymbolen zurück, ohne Markdown, ohne ein einziges anderes Wort.
Beispielformat:
["ROK", "XYL", "FSLR"]
"""
    try:
        response = model.generate_content(prompt)
        text = response.text.replace("```json", "").replace("```", "").strip()
        symbols = json.loads(text)
        if not isinstance(symbols, list) or len(symbols) == 0:
            raise ValueError("Erwartetes JSON-Array nicht empfangen.")
        
        # Begrenze auf 3 falls die KI zu viele liefert
        symbols = [str(s).strip().upper() for s in symbols[:3]]
        log.info(f"✅ Identifizierte Kandidaten: {', '.join(symbols)}")
        return symbols
    except Exception as e:
        log.error(f"❌ Fehler bei der Kandidatensuche: {e}")
        log.error(f"Rohausgabe der KI: {response.text if 'response' in locals() else 'N/A'}")
        sys.exit(1)

def fetch_stock_data(symbol: str) -> dict:
    log.info(f"   => Lade Live-Daten für {symbol}...")
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        current_price = info.get('currentPrice', info.get('regularMarketPrice', 'N/A'))
        trailing_pe = info.get('trailingPE', 'N/A')
        forward_pe = info.get('forwardPE', 'N/A')
        recommendation = info.get('recommendationKey', 'N/A')
        target_price = info.get('targetMeanPrice', 'N/A')
        short_name = info.get('shortName', symbol)
        debt_to_equity = info.get('debtToEquity', 'N/A')
        revenue_growth = info.get('revenueGrowth', 'N/A')
        
        # Get recent news
        recent_news = []
        try:
            news_items = ticker.news
            if news_items:
                for n in news_items[:3]:
                    title = n.get('title')
                    if not title and 'content' in n:
                        title = n['content'].get('title')
                    if title:
                        recent_news.append(title)
        except:
            pass
            
        return {
            "symbol": symbol,
            "name": short_name,
            "current_price": current_price,
            "trailing_pe": trailing_pe,
            "forward_pe": forward_pe,
            "analyst_rating": recommendation,
            "target_price": target_price,
            "debt_to_equity": debt_to_equity,
            "revenue_growth": revenue_growth,
            "recent_news": recent_news
        }
    except Exception as e:
        log.warning(f"⚠️ Konnte keine vollständigen Daten für {symbol} laden: {e}")
        return {"symbol": symbol, "error": str(e)}

def critically_analyze_stock(model, stock_data: dict) -> str:
    symbol = stock_data.get("symbol")
    name = stock_data.get("name", symbol)
    
    log.info(f"\n🧠 [Schritt 3] Erstelle Deep Dive und kritische Analyse für {name} ({symbol})...")
    
    prompt = f"""Du bist ein extrem kritischer und skeptischer Investment-Analyst (ein "Gegen-den-Strom"-Denker). Deine Aufgabe ist es, einen schonungslosen, detaillierten Deep-Dive zu einer Aktie zu schreiben, die zuvor als positiv identifiziert wurde.

Aktie: {name} ({symbol})

Aktuelle Live-Finanzdaten von Yahoo Finance:
- Aktueller Preis: {stock_data.get('current_price')}
- KGV (Trailing PE): {stock_data.get('trailing_pe')}
- Zukunfts-KGV (Forward PE): {stock_data.get('forward_pe')}
- Analysten-Konsens: {stock_data.get('analyst_rating')}
- Durchschnittliches Kursziel: {stock_data.get('target_price')}
- Schulden/Eigenkapital-Quote (Debt to Equity): {stock_data.get('debt_to_equity')}
- Umsatzwachstum: {stock_data.get('revenue_growth')}
- Letzte News-Schlagzeilen: {'; '.join(stock_data.get('recent_news', []))}

AUFTRAG:
Schreibe eine ausgewogene, tiefgehende und rationale Analyse für diese Aktie. Halte dich exakt an folgende Struktur:
1. Das "Pick-and-Shovel" Bull Case: Für welchen großen Hype ist dieses Unternehmen im Hintergrund unverzichtbar? Warum hat diese Aktie vermutlich hohes Potenzial für die nächsten 6-18 Monate? (Verwende Fundamentaldaten, um das zu belegen.)
2. Der rationale Realitätscheck: Beleuchte objektiv die wesentlichen Risiken (makroökonomisch, firmenintern oder wettbewerbsbedingt). Ist die aktuelle Bewertung eigentlich viel zu hoch oder gerechtfertigt angesichts des Wachstums? Lass dich nicht von kurzfristigen negativen News ablenken, sondern blicke auf das strukturelle Bild.
3. Risiko-Rendite-Abwägung & Ehrliches Fazit: Wie steht das konkrete Downside-Risiko im Verhältnis zum Upside-Potenzial? Sei hierbei kein starrköpfiger Pessimist, der von jedem Investment abrät, sondern ein realistischer Strategieberater. Lohnt sich ein Investment für einen rational denkenden Investor? Positioniere dich klar (Kaufen, Warten auf Rücksetzer, Hände weg).

Schreibe professionell, analytisch und auf Deutsch. Die 3 Aktien, die du in Schritt 1 übergeben hast, sollen in diesem Schritt nun nicht einfach "niedergemacht" werden! Ich möchte im Fazit eine wirkliche kauf- oder warte-Empfehlung für rationale Anleger – wenn die Aktie nichts taugt, hättest du sie in Schritt 1 gar nicht erst vorschlagen sollen. Zeige mir, *warum* du das Upside für so sicher hältst, belege aber auch, welche rationalen Risken existieren.

Formatiere dein Ergebnis als reines HTML-Snippet.
WICHTIG: Nutze NUR <h3>, <p>, <ul>, <li>, <strong>, <em> Tags! Verbotene Tags: <html>, <head>, <body>, ```html !
"""
    try:
        response = model.generate_content(prompt)
        text = response.text.replace("```html", "").replace("```", "").strip()
        return text
    except Exception as e:
        return f"Fehler bei der KI-Analyse für {symbol}: {e}"

def build_email_html(reports) -> str:
    now = datetime.now(timezone(timedelta(hours=1)))
    date_str = now.strftime("%d.%m.%Y")
    
    reports_html = ""
    for sym, name, report in reports:
        reports_html += f"""
        <div style="background-color:#ffffff;border:1px solid #e2e8f0;border-radius:8px;padding:20px;margin-bottom:25px;box-shadow:0 2px 4px rgba(0,0,0,0.02);">
            <h2 style="margin:0 0 15px 0;color:#0ea5e9;font-size:18px;border-bottom:2px solid #e0f2fe;padding-bottom:10px;">📉 {name} ({sym})</h2>
            <div style="font-size:15px;color:#334155;line-height:1.6;">
                {report}
            </div>
        </div>
        """
        
    html = f"""<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" lang="de">
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
<title>Deep Research Report</title>
</head>
<body style="margin:0;padding:0;background-color:#f1f5f9;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;">
<table width="100%" border="0" cellspacing="0" cellpadding="0" bgcolor="#f1f5f9">
    <tr>
        <td align="center" style="padding:40px 15px;">
            <table width="100%" max-width="650" border="0" cellspacing="0" cellpadding="0" bgcolor="#ffffff" style="max-width:650px;border-radius:12px;overflow:hidden;box-shadow:0 4px 15px rgba(0,0,0,0.05);border:1px solid #e2e8f0;">
                <!-- HEADER -->
                <tr>
                    <td bgcolor="#0f172a" style="padding:40px 30px;background-color:#0f172a;">
                        <h1 style="margin:0 0 5px 0;color:#ffffff;font-size:28px;letter-spacing:-0.5px;">🕵️‍♂️ Deep Research Report</h1>
                        <p style="margin:0;color:#94a3b8;font-size:16px;">Kritische KI-Aktienanalyse vom {date_str}</p>
                    </td>
                </tr>
                <!-- CONTENT -->
                <tr>
                    <td style="padding:35px 30px 10px 30px;">
                        <p style="margin:0 0 20px 0;font-size:15px;color:#475569;">Hier sind die heutigen 3 Kandidaten für ein außerordentliches mittelfristiges Anlagepotenzial – gnadenlos und kritisch hinterfragt:</p>
                        {reports_html}
                    </td>
                </tr>
                <!-- FOOTER -->
                <tr>
                    <td bgcolor="#f8fafc" style="padding:25px 30px;border-top:1px solid #e2e8f0;text-align:center;">
                        <p style="margin:0;font-size:12px;color:#94a3b8;">
                            Powered by DeepResearchBot &bull; Gemini 2.5 Flash &bull; yfinance API
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
    email_address = os.environ.get("EMAIL_ADDRESS", "")
    email_password = os.environ.get("EMAIL_PASSWORD", "")
    email_recipient = os.environ.get("EMAIL_RECIPIENT", "")

    if not all([email_address, email_password, email_recipient]):
        log.warning("⚠️ E-Mail-Konfiguration unvollständig. Überspringe E-Mail-Versand.")
        return False

    now = datetime.now(timezone(timedelta(hours=1)))
    
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Deep Research Report - {now.strftime('%d.%m.%Y')}"
    msg["From"] = f"Deep Research Bot <{email_address}>"
    msg["To"] = email_recipient

    html_part = MIMEText(html_content, "html", "utf-8")
    msg.attach(html_part)

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(email_address, email_password)
            server.send_message(msg)
        log.info(f"✅ E-Mail erfolgreich an {email_recipient} gesendet.")
        return True
    except Exception as e:
        log.error(f"❌ Fehler beim E-Mail-Versand: {e}")
        return False

def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        log.error("❌ FEHLER: Umgebungsvariable GEMINI_API_KEY ist nicht gesetzt.")
        log.error("Bitte setze sie mit: export GEMINI_API_KEY='dein-key'")
        sys.exit(1)
        
    model = get_gemini_model(api_key)
    
    # 1. Kandidaten finden
    symbols = identify_candidates(model)
    
    print("\n" + "="*60)
    print(" DEEP RESEARCH DASHBOARD ")
    print("="*60 + "\n")
    
    all_reports = []
    
    # 2. & 3. Daten abrufen und analysieren
    for sym in symbols:
        data = fetch_stock_data(sym)
        report = critically_analyze_stock(model, data)
        all_reports.append((sym, data.get("name", sym), report))
        
    # Ergebnisse hübsch im Terminal ausgeben (jetzt mit HTML Tags, aber nützlich zum Debuggen)
    print("\n\n" + "#"*60)
    print(" ERGEBNISSE DER KRITISCHEN DEEP-RESEARCH-ANALYSE ")
    print("#"*60 + "\n")
    
    for sym, name, report in all_reports:
        print(f"--- AKTIE: {name} ({sym}) ---")
        print(report)
        print("\n" + "-"*60 + "\n")
        
    # HTML bauen und E-Mail senden
    email_html = build_email_html(all_reports)
    
    # Für manuelles Testen lokal speichern
    with open("deep_research_report.html", "w", encoding="utf-8") as f:
        f.write(email_html)
        
    send_email(email_html)
        
    log.info("\n✅ Deep Research abgeschlossen.")

if __name__ == "__main__":
    main()
