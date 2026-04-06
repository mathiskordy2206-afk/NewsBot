import os
import sys
import yaml
import json
import feedparser
import yfinance as yf
from datetime import datetime
import requests
from google import genai
from pydantic import BaseModel, Field

# Model schemas for structured output
class Headline(BaseModel):
    title: str = Field(description="Zusammenfassender deutscher Titel")
    summary: str = Field(description="1-2 kurze Sätze Erklärung zur Implikation auf den Markt")
    link: str = Field(description="URL der Quelle")

class DailySummary(BaseModel):
    sentiment: str = Field(description="Bullish, Bearish, oder Neutral")
    top_headlines: list[Headline] = Field(description="Maximal 7 wichtigste Schlagzeilen")
    market_insight: str = Field(description="Ein analytischer Absatz zu den erwarteten Marktbewegungen (auf Deutsch)")

def load_feeds():
    """Lade die konfigurierten News-Feeds."""
    try:
        with open("feeds.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            return config.get("feeds", [])
    except Exception as e:
        print(f"Fehler beim Laden der Feeds: {e}")
        return []

def gather_news(feeds):
    """Sammle die aktuellsten 10 Artikel aus jedem Feed."""
    articles = []
    for feed in feeds:
        try:
            parsed = feedparser.parse(feed["url"])
            for entry in parsed.entries[:10]:
                title = entry.get("title", "").strip()
                link = entry.get("link", "")
                if title and link:
                    articles.append(f"[{feed['name']}] {title} - {link}")
        except Exception as e:
            print(f"Konnte Feed {feed['name']} nicht laden: {e}")
    return articles

def get_market_data():
    """Hole einen kurzen Markt-Snapshot via yfinance."""
    print("Rufe Marktdaten ab...")
    tickers = {"^GSPC": "S&P 500", "^GDAXI": "DAX", "BTC-USD": "Bitcoin", "^TNX": "U.S. 10Y Yield"}
    market_info = []
    
    for symbol, name in tickers.items():
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2d")
            if len(hist) >= 2:
                last_val = hist['Close'].iloc[-2]
                curr_val = hist['Close'].iloc[-1]
                change = curr_val - last_val
                change_pct = (change / last_val) * 100
                
                if symbol == "^TNX":
                    market_info.append(f"**{name}:** {curr_val:.2f}% ({change*100:+.1f} bps)")
                else:
                    market_info.append(f"**{name}:** {curr_val:,.0f} ({change_pct:+.2f}%)")
        except Exception as e:
            print(f"Fehler bei {name}: {e}")
            
    return "\n".join(market_info)

def generate_report(articles, market_data, api_key):
    """Nutze Gemini, um einen strukturierten Newsletter zu erstellen."""
    print("Analysiere Daten mit Gemini...")
    client = genai.Client(api_key=api_key)
    
    # Text input for Gemini
    news_text = "\n".join(articles[:50]) # Use max 50 for context
    prompt = f"Analysiere die heutigen Finanz- und Wirtschaftsnachrichten.\n\nNACHRICHTEN:\n{news_text}\n"
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config={
            'response_mime_type': 'application/json',
            'response_schema': DailySummary,
            'temperature': 0.2,
        },
    )
    
    data = json.loads(response.text)
    
    now = datetime.now()
    time_greeting = "Morgen" if now.hour < 12 else "Abend"
    
    # Formatiere das Markdown mit Erwähnung für E-Mail-Benachrichtigung
    md = f"## 📰 Daily News Briefing ({now.strftime('%d.%m.%Y, %H:%M')})\n\n"
    md += f"@mathiskordy2206-afk Guten {time_greeting}! Hier ist dein Finanz-Briefing.\n\n"
    md += f"### 📊 Markt-Snapshot\n{market_data}\n\n"
    md += f"**Marktstimmung:** {data.get('sentiment', 'Neutral')}\n\n"
    
    md += "### 🔥 Top Schlagzeilen\n"
    for h in data.get('top_headlines', []):
        md += f"- **[{h['title']}]({h['link']})**\n  {h['summary']}\n\n"
        
    md += f"### 🧠 Market Insight\n{data.get('market_insight', '')}\n\n"
    md += "---\n*Generiert mit DailyNewsAgent*"
    
    return md

def post_github_issue(markdown_content):
    """Postet den Newsletter als GitHub Issue, um eine Benachrichtigung/E-Mail auszulösen."""
    token = os.environ.get("GITHUB_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY")
    
    if not token or not repo:
        print("GITHUB_TOKEN oder GITHUB_REPOSITORY fehlt. Überspringe Issue-Post (Dry-Run Modus).")
        return False
        
    print("Erstelle GitHub Issue...")
    url = f"https://api.github.com/repos/{repo}/issues"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    date_str = datetime.now().strftime('%d.%m.%Y - %H:%M Uhr')
    data = {"title": f"Daily Briefing: {date_str}", "body": markdown_content, "labels": ["newsletter"]}
    
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 201:
        print("Erfolgreich als Issue gepostet!")
        return True
    else:
        print(f"Fehler: {response.status_code} - {response.text}")
        return False

def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Fehler: GEMINI_API_KEY Umgebungsvariable nicht gesetzt!")
        sys.exit(1)
        
    feeds = load_feeds()
    articles = gather_news(feeds)
    
    if not articles:
        print("Keine Artikel gefunden. Beende.")
        sys.exit(1)
        
    market_data = get_market_data()
    report = generate_report(articles, market_data, api_key)
    
    print("\n\n" + "="*50 + "\n" + report + "\n" + "="*50 + "\n")
    
    post_github_issue(report)

if __name__ == "__main__":
    main()
