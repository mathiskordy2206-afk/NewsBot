import os
import sys
import json
import logging
import time
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, timezone, timedelta
import requests
import yfinance as yf
import google.generativeai as genai

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("DeepResearch")

def get_gemini_model(api_key: str):
    genai.configure(api_key=api_key)
    # Using the flash model as requested in previous parts of the project
    return genai.GenerativeModel('gemini-2.5-flash')

import re

def identify_candidates(model) -> list:
    log.info("\n🔍 [Schritt 1] KI sucht nach fundamental starken, nicht-mainstream Aktien...\n")
    prompt = """Du bist ein datengetriebener Aktienanalyst für institutionelle Investoren.
Deine Aufgabe ist es, exakt 3 Unternehmen zu finden, die aktuell ein enormes, realistisches Upside-Potenzial für die nächsten 6 bis 18 Monate bieten.

STRIKTE REGELN ZUR AUSWAHL:
1. KEIN MAINSTREAM: Wähle keine Unternehmen, die täglich in den Nachrichten sind (z.B. NVDA, TSLA, AAPL, MSFT, AMZN, META, GOOGL, AMD). Suche in der zweiten Reihe nach hochprofitablen oder stark wachsenden Hidden Champions.
2. DYNAMIK & VIELFALT: Überlege dir 3 völlig unterschiedliche, spannende Sektoren (z.B. Industrieautomatisierung, spezialisierte Software, Medizintechnik, Infrastruktur, Nischen-Chemie) und wähle aus jedem Sektor den attraktivsten Player.
3. FUNDAMENTALE STÄRKE: Das Unternehmen muss einen echten, greifbaren Wachstumstreiber oder Burggraben haben (keine reinen Meme- oder Zockeraktien). Führe im Hintergrund ein breites Screening von mind. 10 Aktien durch und wähle nur die besten 3 aus.

Gib als Antwort AUSSCHLIESSLICH ein valides JSON-Array mit den amerikanischen Tickersymbolen zurück, ohne Markdown, ohne ein einziges drittes Wort.
Beispielformat:
["ROK", "XYL", "FSLR"]
"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            
            # Check for safety filter block
            if not response.parts:
                log.warning(f"⚠️ Versuch {attempt+1}: Leere Antwort generiert (Möglicherweise durch Safety-Filter blockiert).")
                time.sleep(10)
                continue
                
            text = response.text.replace("```json", "").replace("```", "").strip()
            
            # Robust JSON extraction
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                text = match.group(0)
                
            symbols = json.loads(text)
            if not isinstance(symbols, list) or len(symbols) == 0:
                raise ValueError("Erwartetes JSON-Array nicht empfangen.")
            
            # Begrenze auf 3 falls die KI zu viele liefert
            symbols = [str(s).strip().upper() for s in symbols[:3]]
            log.info(f"✅ Identifizierte Kandidaten: {', '.join(symbols)}")
            return symbols
        except Exception as e:
            log.warning(f"⚠️ Versuch {attempt+1} fehlgeschlagen: {e}")
            if hasattr(response, 'text'):
                log.warning(f"Rohausgabe der KI: {response.text}")
            elif hasattr(response, 'candidates') and response.candidates:
                log.warning(f"Finish Reason: {response.candidates[0].finish_reason}")
            time.sleep(10) # Pausiere vor dem nächsten Versuch
            
    log.error("❌ Fehler bei der Kandidatensuche: Maximale Anzahl an Versuchen erreicht.")
    sys.exit(1)

def fetch_stock_data(symbol: str) -> dict:
    log.info(f"   => Lade Live-Daten für {symbol}...")
    try:
        # User-Agent Session für Yahoo Finance erstellen, um Blockaden durch GitHub Actions zu verhindern
        session = requests.Session()
        session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'})
        
        ticker = yf.Ticker(symbol, session=session)
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
Erstelle eine präzise, faktenbasierte und strukturierte Analyse. Halte dich exakt an diesen strukturierten Output und vermeide jegliches ausschweifende "Gequassel":

1. Kurze Vorstellung: Was macht das Unternehmen (in maximal 2 Sätzen) und warum ist sein Geschäftsmodell stark?
2. Wichtige Kennzahlen: Fasse die zur Verfügung gestellten realen Daten (KGV, Umsatzwachstum, Verschuldung etc.) sachlich in 1-2 Sätzen zusammen.
3. Warum diese Aktie Potenzial hat: Erkläre den konkreten Katalysator. Warum ist realistischerweise mit einem Upside in den nächsten 6-18 Monaten zu rechnen? Warum ist *genau jetzt* der richtige Kaufmoment?
4. Kritische Abwägung (Risiko): Beleuchte realistische Risiken. Wo liegen die Gefahren in der Bilanz oder im Marktumfeld?
5. Klare Entscheidung: Wäge die Argumente aus Punkt 3 und 4 gegeneinander ab und fälle ein eindeutiges Urteil (Kaufen oder Abwarten). Kein Schwammig-Reden.

Formatiere dein Ergebnis als reines HTML-Snippet. Nutze <h3> für die Zwischenüberschriften der 5 Punkte und <p> für den Text. 
WICHTIG: Nutze NUR <h3>, <p>, <ul>, <li>, <strong>, <em> Tags! Verbotene Tags: <html>, <head>, <body>, ```html , Markdown!
"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            if not response.parts:
                log.warning(f"⚠️ Versuch {attempt+1}: Leere Antwort generiert (Möglicherweise durch Safety-Filter blockiert).")
                time.sleep(10)
                continue
                
            text = response.text.replace("```html", "").replace("```", "").strip()
            return text
        except Exception as e:
            log.warning(f"⚠️ Versuch {attempt+1} für Analyse von {symbol} fehlgeschlagen: {e}")
            time.sleep(10)
            
    return f"<p style='color:red;'>Fehler bei der KI-Analyse für {symbol} nach mehreren Versuchen. (API Rate Limits oder Safety-Blocker).</p>"

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
    api_key = os.environ.get("GEMINI_RESEARCH_API_KEY")
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY") # Fallback
        
    if not api_key:
        log.error("❌ FEHLER: Umgebungsvariable GEMINI_RESEARCH_API_KEY ist nicht gesetzt.")
        log.error("Bitte setze sie mit: export GEMINI_RESEARCH_API_KEY='dein-key'")
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
        
        # Rate-Limiting für den Gemini Free-Tier verhindern (max 15 RPM)
        # Wir warten hier ein paar Sekunden zwischen den Anfragen.
        log.info("⏳ Kurze Pause für API-Rate-Limits...")
        time.sleep(5)
        
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
