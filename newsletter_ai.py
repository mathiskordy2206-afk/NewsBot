import os
import sys
import yaml
import json
import feedparser
from datetime import datetime
import requests
from google import genai
from pydantic import BaseModel, Field

# Model schemas for structured output
class GeneralRelease(BaseModel):
    title: str = Field(description="Titel des neuen Produkt- oder Modell-Releases auf Deutsch")
    summary: str = Field(description="Was EXAKT ist neu? Funktionen, Updates oder Launches der letzten Tage.")
    link: str = Field(description="URL der Quelle")

class StudentHack(BaseModel):
    title: str = Field(description="Titel des Tools oder des Hacks")
    hack_description: str = Field(description="Konkretes neues Feature oder Tool-Update für Studenten und wie man es einsetzen kann.")
    link: str = Field(description="URL der Quelle")

class FinanceAI(BaseModel):
    title: str = Field(description="Titel der Entwicklung")
    finance_impact: str = Field(description="Konkreter Anwendungsfall oder neues Tool-Announcement für Banken/Analysten.")
    link: str = Field(description="URL der Quelle")

class DailyAISummary(BaseModel):
    general_releases: list[GeneralRelease] = Field(description="Maximal 4 handfeste neue KI-Releases oder App-Updates (kein Generisches Blabla)")
    student_hacks: list[StudentHack] = Field(description="Maximal 3 Must-Haves/Hacks für Studenten")
    finance_ai: list[FinanceAI] = Field(description="Maximal 3 echte neue Use-Cases/Tools für die Finanzwelt")

def load_feeds():
    """Lade die konfigurierten KI-Feeds."""
    try:
        with open("feeds_ai.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            return config.get("feeds", [])
    except Exception as e:
        print(f"Fehler beim Laden der Feeds: {e}")
        return []

def gather_news(feeds):
    """Sammle die aktuellsten 15 Artikel aus jedem Feed."""
    articles = []
    for feed in feeds:
        try:
            parsed = feedparser.parse(feed["url"])
            for entry in parsed.entries[:15]:
                title = entry.get("title", "").strip()
                link = entry.get("link", "")
                summary = entry.get("summary", "")[:200]
                if title and link:
                    articles.append(f"[{feed['name']}] TITEL: {title} | LINK: {link} | TEXT: {summary}")
        except Exception as e:
            print(f"Konnte Feed {feed['name']} nicht laden: {e}")
    return articles

def generate_report(articles, api_key):
    """Nutze Gemini, um einen strukturierten KI-Newsletter zu erstellen."""
    print("Analysiere KI Daten mit Gemini...")
    client = genai.Client(api_key=api_key)
    
    news_text = "\n".join(articles[:60]) # Increase context
    prompt = f"""Du bist ein Tech & Finance Analyst. Analysiere die folgenden aktuellsten KI-Nachrichten.
    WICHTIGSTE REGEL: Ignoriere allgemeines philosophisches KI-Gerede, Meinungen oder abstrakte Konzepte.
    Extrahiere AUSSCHLIESSLICH explizite Tool-Releases, Update-Ankündigungen, neue KI-Modelle, oder handfeste neue Features, die im letzten Zeitraum veröffentlicht wurden! Der Nutzer darf nichts verpassen!
    
    1. Finde die besten neuen Tool/Modell-Releases (Allgemein).
    2. Finde explizit neue Tools und Update-Funktionen, die JETZT für das Studium/Studenten nutzbar sind.
    3. Finde Berichte über konkrete neue KI-Anwendungen, Plugins oder Plattformen in der Finanzwelt (Banking, Trading, Research).
    
    NACHRICHTEN:
    {news_text}
    """
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config={
            'response_mime_type': 'application/json',
            'response_schema': DailyAISummary,
            'temperature': 0.1,
        },
    )
    
    data = json.loads(response.text)
    now = datetime.now()
    
    md = f"## 🤖 Bi-Weekly AI & Future Briefing ({now.strftime('%d.%m.%Y')})\n\n"
    md += f"@mathiskordy2206-afk Hier sind die neuesten echten Tool-Releases und Modell-Updates der letzten Tage!\n\n"
    
    md += "### 🎓 Neue Tools & Hacks fürs Studium\n"
    if not data.get('student_hacks'):
        md += "_Keine relevanten neuen Studientools in den letzten Tagen._\n\n"
    for h in data.get('student_hacks', []):
        md += f"- **[{h['title']}]({h['link']})**\n  💡 *Neuheit:* {h['hack_description']}\n\n"
        
    md += "### 💼 KI Releases im Finance-Sektor\n"
    if not data.get('finance_ai'):
        md += "_Keine handfesten neuen Tools im Finance-Sektor gefunden._\n\n"
    for f in data.get('finance_ai', []):
        md += f"- **[{f['title']}]({f['link']})**\n  📈 *Finance Impact:* {f['finance_impact']}\n\n"

    md += "### 🚀 Weltweite Tool & Modell Updates\n"
    for g in data.get('general_releases', []):
        md += f"- **[{g['title']}]({g['link']})**\n  {g['summary']}\n\n"
        
    md += "---\n*Generiert mit DailyNewsAgent - AI Module*"
    
    return md

def post_github_issue(markdown_content):
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
    
    date_str = datetime.now().strftime('%d.%m.%Y')
    data = {"title": f"🤖 AI & Future Briefing: {date_str}", "body": markdown_content, "labels": ["ai-newsletter"]}
    
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 201:
        print("Erfolgreich als AI-Issue gepostet!")
        return True
    else:
        print(f"Fehler: {response.status_code} - {response.text}")
        return False

def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Fehler: GEMINI_API_KEY nicht gesetzt!")
        sys.exit(1)
        
    feeds = load_feeds()
    articles = gather_news(feeds)
    
    if not articles:
        print("Keine KI Artikel gefunden. Beende.")
        sys.exit(1)
        
    report = generate_report(articles, api_key)
    
    print("\n\n" + "="*50 + "\n" + report + "\n" + "="*50 + "\n")
    
    post_github_issue(report)

if __name__ == "__main__":
    main()
