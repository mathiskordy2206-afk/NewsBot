import os
import sys
import yaml
import json
import feedparser
from datetime import datetime
import requests
from google import genai
from pydantic import BaseModel, Field

class AIRelease(BaseModel):
    title: str = Field(description="Name des neuen Tools, Modells oder Features")
    short_description: str = Field(description="Was genau ist in den letzten Tagen erschienen? (Was ist neu?)")
    new_features: str = Field(description="Was kann das Tool konkret? (Welche Features bietet es?)")
    use_case: str = Field(description="Wofür wird das in der Praxis (Beruf, Studium, Alltag) gebraucht?")
    link: str = Field(description="URL der News-Quelle")

class DailyAISummary(BaseModel):
    recent_releases: list[AIRelease] = Field(description="Die 5 bis 7 absolut wichtigsten, echten Releases und konkreten KI-Updates der letzten Tage")

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
    
    prompt = f"""Du bist ein Analyst für künstliche Intelligenz. Analysiere die folgenden aktuellsten Nachrichten.
    Wir suchen AUSSCHLIESSLICH nach echten, brandneuen Releases der letzten Tage: neue Tools, neue KI-Modelle, oder große neue Features bestehender Softwares.
    Ignoriere kategorisch alles, was nur philosophische Debatten, Meinungen, Prognosen oder allgemeine Diskussionen sind.
    
    Für jedes echte Release, extrahiere:
    - Den genauen Namen
    - Was exakt neu erschienen ist.
    - Was das Tool/Modell technisch kann.
    - Wofür ein Mensch (Beruf, Studium, Alltag) dieses Tool konkret braucht.
    
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
    
    md = f"## 🤖 AI Release Radar ({now.strftime('%d.%m.%Y')})\n\n"
    md += f"@mathiskordy2206-afk Hier ist dein Digest der brandheißesten KI-Releases der letzten Tage!\n\n"
    
    releases = data.get('recent_releases', [])
    if not releases:
        md += "_In den analysierten Quellen wurden keine konkreten Tool-Releases für die letzten Tage gefunden._\n\n"
    
    for r in releases:
        md += f"### 🔥 [{r['title']}]({r['link']})\n"
        md += f"**Was ist neu?** {r['short_description']}\n\n"
        md += f"**Was kann es?** {r['new_features']}\n\n"
        md += f"**Wofür braucht man es?** {r['use_case']}\n\n"
        md += "---\n"
        
    md += "\n*Generiert mit DailyNewsAgent - AI Release Module*"
    
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
