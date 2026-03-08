import os
import sys
import json
import logging
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
    log.info("\n🔍 [Schritt 1] KI sucht nach 3 Aktien mit hohem mittelfristigem Potenzial...\n")
    prompt = """Du bist ein erstklassiger Aktien-Analyst.
Identifiziere exakt 3 Aktien (Unternehmen), die aktuell ein extrem hohes, positives Potenzial für die mittelfristige Zukunft (6 bis 18 Monate) haben.
Sie können aus jedem Sektor stammen, aber es muss fundamentale oder makroökonomische Katalysatoren geben.

Gib als Antwort AUSSCHLIESSLICH ein valides JSON-Array mit den Tickersymbolen zurück, ohne Markdown, ohne weitere Erklärungen.
Beispielformat:
["AAPL", "NVDA", "MC.PA"]
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
Schreibe eine tiefgehende, kritische Analyse für diese Aktie. Halte dich an folgende Struktur:
1. Das "Bull Case" (Der Katalysator): Warum hat diese Aktie vermutlich hohes Potenzial für die nächsten 6-18 Monate?
2. Der kritische Realitätscheck (Das "Bear Case"): Zerreiße das Bull Case. Welche makroökonomischen, firmeninternen oder wettbewerbsbedingten Risiken werden übersehen? Ist die Bewertung (KGV) aktuell eigentlich viel zu hoch?
3. Fazit & Knallhartes Urteil: Ist das wirklich ein gutes Investment für einen rationalen Anleger, oder doch eher eine riskante Wette? Positioniere dich klar (Kaufen, Warten, Hände weg).

Schreibe professionell, analytisch und auf Deutsch. Nutze das bereitgestellte Datenmaterial (z.B. hohe KGV oder fragliches Umsatzwachstum), um deine Skepsis zu belegen. Markdown ist erlaubt. Mach es nicht künstlich lang, aber inhaltlich extrem dicht.
"""
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Fehler bei der KI-Analyse für {symbol}: {e}"

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
        
    # Ergebnisse hübsch ausgeben
    print("\n\n" + "#"*60)
    print(" ERGEBNISSE DER KRITISCHEN DEEP-RESEARCH-ANALYSE ")
    print("#"*60 + "\n")
    
    for sym, name, report in all_reports:
        print(f"--- AKTIE: {name} ({sym}) ---")
        print(report)
        print("\n" + "-"*60 + "\n")
        
    log.info("\n✅ Deep Research abgeschlossen.")

if __name__ == "__main__":
    main()
