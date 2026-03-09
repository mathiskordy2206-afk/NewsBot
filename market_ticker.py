import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, timezone, timedelta
import yfinance as yf
import logging
import time

# Logging konfigurieren
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("MarketTicker")

# --- Konfiguration ---
# Wichtigste Indizes & ETFs
MARKET_SYMBOLS = {
    "^GSPC": "S&P 500",
    "^NDX": "Nasdaq 100",
    "^DJI": "Dow Jones",
    "^GDAXI": "DAX",
    "URTH": "MSCI World ETF",
    "BTC-USD": "Bitcoin"
}

# Großes Basket für "Ausreißer" am Markt
STOCK_BASKET = [
    "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "GOOGL", "BRK-B", "JPM", "V", 
    "WMT", "JNJ", "PG", "XOM", "UNH", "MA", "HD", "CVX", "MRK", "ABBV",
    "LLY", "AVGO", "COST", "PEP", "KO", "BAC", "TMO", "MCD", "DIS", "ADBE",
    "CRM", "AMD", "NFLX", "INTC", "CSCO", "BA", "NKE", "SBUX", "PFE", "PLTR",
    "COIN", "RHEM.DE", "SAP", "ALV.DE", "SIE.DE", "NOW", "UBER", "CRWD", "SNOW", "ARM"
]

def fetch_market_data():
    log.info("Lade Daten für Indizes und ETFs...")
    market_data = {}
    
    # Indizes einzeln laden
    for sym, name in MARKET_SYMBOLS.items():
        try:
            ticker = yf.Ticker(sym)
            hist = ticker.history(period="5d")
            if len(hist) >= 2:
                current = hist['Close'].iloc[-1]
                prev = hist['Close'].iloc[-2]
                pct_change = ((current - prev) / prev) * 100
                market_data[name] = {"price": current, "pct_change": pct_change, "symbol": sym}
        except Exception as e:
            log.warning(f"Fehler bei {name} ({sym}): {e}")

    log.info("Suche nach Ausreißern (Top Gainers/Losers)...")
    outliers_data = []
    
    # Um yfinance nicht mit 50 Einzelabfragen zu blockieren, laden wir per Tickers "download"
    try:
        # Ticker download mit threads
        data = yf.download(STOCK_BASKET, period="5d", threads=True, progress=False)
        
        # 'Close' column überprüfen (pandas MultiIndex Format)
        if 'Close' in data:
            closes = data['Close']
            for sym in STOCK_BASKET:
                try:
                    if sym in closes.columns:
                        series = closes[sym].dropna()
                        if len(series) >= 2:
                            current = series.iloc[-1]
                            prev = series.iloc[-2]
                            pct_change = ((current - prev) / prev) * 100
                            # Nur Aktien mit starker Bewegung als Ausreißer (>3% oder <-3%) reinnehmen 
                            # (oder einfach alle sortieren)
                            outliers_data.append({"symbol": sym, "pct_change": float(pct_change), "price": float(current)})
                except Exception as e:
                    pass
    except Exception as e:
        log.warning(f"Fehler beim Bulk-Download: {e}")

    # Top 5 Gewinner und Verlierer
    outliers_data.sort(key=lambda x: x["pct_change"])
    biggest_losers = outliers_data[:5]
    biggest_winners = list(reversed(outliers_data[-5:]))
    
    # Allgemeine News laden, z.B. vom S&P 500 Ticker
    news_headlines = []
    try:
        sp500 = yf.Ticker("^GSPC")
        for n in sp500.news[:5]:
            title = n.get("title")
            if not title and "content" in n:
                title = n["content"].get("title")
            if title:
                news_headlines.append(title)
    except Exception:
        pass

    return market_data, biggest_winners, biggest_losers, news_headlines

def generate_ai_summary(market_data, winners, losers, news_headlines, api_key):
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Daten für Prompt aufbereiten
        market_str = "\n".join([f"- {name}: {d['price']:.2f} ({d['pct_change']:.2f}%)" for name, d in market_data.items() if "price" in d])
        winners_str = "\n".join([f"- {w['symbol']}: {w['price']:.2f} (+{w['pct_change']:.2f}%)" for w in winners])
        losers_str = "\n".join([f"- {l['symbol']}: {l['price']:.2f} ({l['pct_change']:.2f}%)" for l in losers])
        news_str = "\n".join([f"- {n}" for n in news_headlines])

        prompt = f"""Du bist ein pointierter, moderner Finanzjournalist (z.B. wie beim Morning Brew oder Markus Koch). 
Fasse den heutigen Handelstag (Wall Street & globale Märkte) nach US-Börsenschluss prägnant zusammen.

HIER SIND DIE DATEN VON HEUTE:
--- INDIZES & ETFs ---
{market_str}

--- TOP GEWINNER DES TAGES ---
{winners_str}

--- TOP VERLIERER DES TAGES ---
{losers_str}

--- HEUTIGE SCHLAGZEILEN ---
{news_str}

AUFGABE:
1. "Der Tag in 3 Sätzen": Eine knackige Einleitung (Was war heute los? Bullish/Bearish? Hauptthema?).
2. "Zahlen-Check": Präsentiere die Indizes übersichtlich (S&P 500, DAX etc.) zusammen mit 1-2 kurzen Sätzen, was dies bedeutet.
3. "Auffallend & Ausreißer": Picke dir max. 3-4 Aktien aus den Gewinnern/Verlierern heraus, die bemerkenswert sind, und erkläre (falls ersichtlich aus den News oder logisch ableitbar aus Tech/Zinsen) *warum* sie sich massiv bewegt haben.
4. "Ausblick": Ein abschließender ermutigender oder nachdenklicher Satz zum morgigen Handelstag.

Formatiere die Antwort in gut strukturiertem HTML (Nutzung von <h3>, <p>, <ul>, <li>, <strong>). Kein Markdown (wie ```html etc.)! Mach es optisch leicht konsumierbar.
"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt)
                if not response.parts:
                    log.warning("Leere Antwort vom Modell.")
                    time.sleep(5)
                    continue
                return response.text.replace("```html", "").replace("```", "").strip()
            except Exception as e:
                log.warning(f"Gemini API Fehler ({attempt+1}/{max_retries}): {e}")
                time.sleep(5)
        
        return "<p>Die KI-Zusammenfassung konnte nach mehreren Versuchen leider nicht generiert werden.</p>"
    except ImportError:
        return "<p>Google Generative AI SDK fehlt.</p>"
    except Exception as e:
        return f"<p>Unerwarteter Fehler: {e}</p>"

def build_email_html(ai_summary):
    now = datetime.now(timezone(timedelta(hours=1)))
    date_str = now.strftime("%d.%m.%Y")
    
    html = f"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8">
<title>Daily Market Ticker</title>
</head>
<body style="margin:0;padding:0;background-color:#f8fafc;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;">
<table width="100%" border="0" cellspacing="0" cellpadding="0" bgcolor="#f8fafc">
    <tr>
        <td align="center" style="padding:40px 15px;">
            <table width="100%" max-width="600" border="0" cellspacing="0" cellpadding="0" bgcolor="#ffffff" style="max-width:600px;border-radius:12px;overflow:hidden;box-shadow:0 10px 25px rgba(0,0,0,0.05);border:1px solid #e2e8f0;">
                
                <!-- HEADER -->
                <tr>
                    <td bgcolor="#1e293b" style="padding:35px 30px;text-align:center;">
                        <h1 style="margin:0;color:#ffffff;font-size:24px;letter-spacing:0.5px;">🔔 Closing Bell Ticker</h1>
                        <p style="margin:8px 0 0 0;color:#94a3b8;font-size:14px;">Börsenschluss-Update &middot; {date_str}</p>
                    </td>
                </tr>

                <!-- CONTENT -->
                <tr>
                    <td style="padding:40px 30px;font-size:15px;color:#334155;line-height:1.6;">
                        {ai_summary}
                    </td>
                </tr>
                
                <!-- FOOTER -->
                <tr>
                    <td bgcolor="#f1f5f9" style="padding:20px 30px;border-top:1px solid #e2e8f0;text-align:center;">
                        <p style="margin:0;font-size:12px;color:#64748b;">
                            Generiert durch Market Ticker Bot &bull; Gemini Flash<br/>
                            Vollautomatisches tägliches Marktsummari!
                        </p>
                    </td>
                </tr>
            </table>
        </td>
    </tr>
</table>
</body>
</html>
"""
    return html

def send_email(html_content, smtp_server="smtp.gmail.com", smtp_port=587):
    email_address = os.environ.get("EMAIL_ADDRESS", "")
    email_password = os.environ.get("EMAIL_PASSWORD", "")
    email_recipient = os.environ.get("EMAIL_RECIPIENT", "")

    if not all([email_address, email_password, email_recipient]):
        log.warning("⚠️ E-Mail-Konfiguration unvollständig, überspringe E-Mail-Versand.")
        return False

    now = datetime.now(timezone(timedelta(hours=1)))
    
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Closing Bell Ticker - {now.strftime('%d.%m.%Y')}"
    msg["From"] = f"Market Ticker <{email_address}>"
    msg["To"] = email_recipient

    html_part = MIMEText(html_content, "html", "utf-8")
    msg.attach(html_part)

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(email_address, email_password)
            server.send_message(msg)
        log.info(f"✅ Ticker erfolgreich an {email_recipient} gesendet.")
        return True
    except Exception as e:
        log.error(f"❌ Fehler beim E-Mail-Versand: {e}")
        return False

def main():
    # Neues API Key Secret für den Ticker verwenden
    api_key = os.environ.get("GEMINI_TICKER_API_KEY", "")
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY", "") # Fallback
        
    if not api_key:
        log.error("❌ GEMINI_TICKER_API_KEY ist nicht gesetzt!")
        return

    market_data, winners, losers, news = fetch_market_data()
    log.info("Lasse KI die Zusammenfassung schreiben...")
    ai_summary = generate_ai_summary(market_data, winners, losers, news, api_key)
    
    html = build_email_html(ai_summary)
    
    with open("market_ticker_report.html", "w", encoding="utf-8") as f:
        f.write(html)
        
    send_email(html)
    log.info("🎉 Ticker Workflow abgeschlossen.")

if __name__ == "__main__":
    main()
