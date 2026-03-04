#!/usr/bin/env python3
"""
NewsBot – Automatisierter täglicher Finanz-Newsletter
=====================================================
Crawlt RSS-Feeds, verarbeitet sie mit Gemini Flash (Zusammenfassung)
und Claude Opus (Deep Analysis bei signifikanten Events), und gibt
einen formatierten Markdown-Newsletter aus.

Ausgabekanäle: GitHub Issue oder E-Mail (SMTP).
"""

import os
import sys
import json
import logging
import smtplib
import argparse
import hashlib
from datetime import datetime, timezone, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher

import yaml
import requests
import feedparser

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("NewsBot")

# ─── Konstanten ───────────────────────────────────────────────────────────────

FEED_TIMEOUT = 15  # Sekunden
MAX_ARTICLES_PER_FEED = 10
SIMILARITY_THRESHOLD = 0.7  # Für Deduplizierung
MAX_HEADLINES = 8
MAX_DEEP_DIVES = 3
MAX_WIRTSCHAFT_KOMPAKT = 6

# ─── Feed Crawling ───────────────────────────────────────────────────────────


def load_feeds(config_path: str = "feeds.yaml") -> list[dict]:
    """Lädt die Feed-Konfiguration aus der YAML-Datei."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, config_path)
    with open(full_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config.get("feeds", [])


def fetch_single_feed(feed_config: dict) -> list[dict]:
    """
    Ruft einen einzelnen RSS-Feed ab und gibt eine Liste
    normalisierter Artikel zurück.
    """
    name = feed_config["name"]
    url = feed_config["url"]
    category = feed_config["category"]
    priority = feed_config.get("priority", 2)

    try:
        log.info(f"📡 Abrufen: {name}")
        parsed = feedparser.parse(url, request_headers={"User-Agent": "NewsBot/1.0"})

        if parsed.bozo and not parsed.entries:
            log.warning(f"⚠️  Feed fehlerhaft oder down: {name} ({parsed.bozo_exception})")
            return []

        articles = []
        for entry in parsed.entries[:MAX_ARTICLES_PER_FEED]:
            # Datum parsen
            published = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
            elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
                published = datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)

            # Nur Artikel der letzten 24 Stunden
            if published:
                age = datetime.now(timezone.utc) - published
                if age > timedelta(hours=36):
                    continue

            summary = ""
            if hasattr(entry, "summary"):
                summary = entry.summary
            elif hasattr(entry, "description"):
                summary = entry.description

            # HTML-Tags entfernen (einfach)
            import re
            summary = re.sub(r"<[^>]+>", "", summary).strip()

            articles.append({
                "title": entry.get("title", "Kein Titel"),
                "link": entry.get("link", ""),
                "summary": summary[:500],
                "source": name,
                "category": category,
                "priority": priority,
                "published": published.isoformat() if published else None,
            })

        log.info(f"✅ {name}: {len(articles)} Artikel gefunden")
        return articles

    except Exception as e:
        log.error(f"❌ Fehler bei {name}: {e}")
        return []


def deduplicate_articles(articles: list[dict]) -> list[dict]:
    """Entfernt doppelte Artikel basierend auf Titel-Ähnlichkeit."""
    unique = []
    seen_titles = []

    for article in articles:
        title = article["title"].lower()
        is_duplicate = False
        for seen in seen_titles:
            if SequenceMatcher(None, title, seen).ratio() > SIMILARITY_THRESHOLD:
                is_duplicate = True
                break
        if not is_duplicate:
            unique.append(article)
            seen_titles.append(title)

    log.info(f"🔄 Deduplizierung: {len(articles)} → {len(unique)} Artikel")
    return unique


def crawl_all_feeds(feeds: list[dict]) -> list[dict]:
    """Ruft alle Feeds parallel ab und dedupliziert die Ergebnisse."""
    all_articles = []

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(fetch_single_feed, f): f for f in feeds}
        for future in as_completed(futures):
            feed = futures[future]
            try:
                articles = future.result()
                all_articles.extend(articles)
            except Exception as e:
                log.error(f"❌ Thread-Fehler bei {feed['name']}: {e}")

    # Sortieren: Priorität 1 zuerst, dann nach Datum
    all_articles.sort(key=lambda a: (a["priority"], a.get("published") or ""))

    return deduplicate_articles(all_articles)


# ─── KI-Verarbeitung ─────────────────────────────────────────────────────────


def call_gemini(prompt: str, api_key: str) -> str:
    """Ruft die Gemini API auf (google-genai SDK)."""
    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        return response.text
    except Exception as e:
        log.error(f"❌ Gemini API Fehler: {e}")
        return ""


def call_claude(prompt: str, api_key: str) -> str:
    """Ruft die Claude API auf für Deep Analysis."""
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
    except Exception as e:
        log.error(f"❌ Claude API Fehler: {e}")
        return ""


def process_with_gemini(articles: list[dict], api_key: str) -> dict:
    """
    Nutzt Gemini Flash für schnelle Kategorisierung und Zusammenfassung.
    Gibt ein strukturiertes Dict mit den Newsletter-Sektionen zurück.
    """
    # Artikel nach Kategorie gruppieren
    by_category = {"weltgeschehen": [], "wirtschaft": [], "finanzen": []}
    for a in articles:
        cat = a.get("category", "wirtschaft")
        if cat in by_category:
            by_category[cat].append(a)

    # ── Artikel-Daten für den Prompt aufbereiten ──
    article_text = ""
    for cat, cat_articles in by_category.items():
        article_text += f"\n=== {cat.upper()} ===\n"
        for a in cat_articles[:15]:
            article_text += f"- [{a['source']}] {a['title']}\n  {a['summary'][:200]}\n"

    prompt = f"""Du bist ein erfahrener Finanzjournalist. Analysiere die folgenden Nachrichtenartikel
und erstelle einen strukturierten Newsletter auf Deutsch.

ARTIKEL:
{article_text}

AUFGABE:
Erstelle ein JSON-Objekt mit folgender Struktur (NUR valides JSON, kein Markdown drum herum):
{{
    "headlines": [
        {{"title": "Kurze Headline", "summary": "1-2 Sätze", "source": "Quelle", "link": "URL"}},
        ... (maximal {MAX_HEADLINES} Headlines, die wichtigsten zuerst)
    ],
    "deep_dives": [
        {{"title": "Thema", "analysis": "3-5 Sätze tiefere Analyse", "source": "Quelle", "link": "URL"}},
        ... (maximal {MAX_DEEP_DIVES}, nur Finanz/Banken-Themen)
    ],
    "wirtschaft_kompakt": [
        {{"title": "Thema", "summary": "1-2 Sätze", "source": "Quelle"}},
        ... (maximal {MAX_WIRTSCHAFT_KOMPAKT})
    ],
    "market_sentiment": "bullish/bearish/neutral",
    "significant_events": true/false,
    "significant_event_summary": "Falls true: Was ist passiert und warum ist es signifikant?"
}}

WICHTIG:
- Antworte NUR mit dem JSON-Objekt, kein anderer Text
- "significant_events" = true nur bei: Zinsänderungen, Bankpleiten, Crash >3%, geopolitische Krisen mit Marktauswirkung
- Alle Texte auf Deutsch
"""

    log.info("🤖 Gemini Flash verarbeitet Artikel...")
    response = call_gemini(prompt, api_key)

    # JSON aus der Antwort extrahieren
    try:
        # Versuche direkt zu parsen
        result = json.loads(response)
    except json.JSONDecodeError:
        # Versuche JSON aus Markdown-Code-Block zu extrahieren
        import re
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response)
        if json_match:
            try:
                result = json.loads(json_match.group(1))
            except json.JSONDecodeError:
                log.error("❌ Konnte JSON nicht parsen, nutze Fallback")
                result = _fallback_structure(articles)
        else:
            log.error("❌ Kein JSON in Gemini-Antwort, nutze Fallback")
            result = _fallback_structure(articles)

    return result


def _fallback_structure(articles: list[dict]) -> dict:
    """Fallback-Struktur falls KI-Verarbeitung fehlschlägt."""
    headlines = []
    deep_dives = []
    wirtschaft = []

    for a in articles:
        item = {"title": a["title"], "summary": a["summary"][:150], "source": a["source"], "link": a.get("link", "")}
        if a["category"] == "finanzen" and len(deep_dives) < MAX_DEEP_DIVES:
            deep_dives.append({**item, "analysis": a["summary"][:300]})
        elif a["category"] == "wirtschaft" and len(wirtschaft) < MAX_WIRTSCHAFT_KOMPAKT:
            wirtschaft.append(item)
        elif len(headlines) < MAX_HEADLINES:
            headlines.append(item)

    return {
        "headlines": headlines,
        "deep_dives": deep_dives,
        "wirtschaft_kompakt": wirtschaft,
        "market_sentiment": "neutral",
        "significant_events": False,
        "significant_event_summary": "",
    }


def deep_analysis_with_claude(gemini_result: dict, api_key: str) -> str:
    """
    Nutzt Claude für eine tiefgreifende Analyse, wenn signifikante
    Marktbewegungen erkannt wurden.
    """
    if not gemini_result.get("significant_events", False):
        log.info("ℹ️  Keine signifikanten Events – Claude Deep Analysis übersprungen")
        return ""

    event_summary = gemini_result.get("significant_event_summary", "")
    deep_dives = gemini_result.get("deep_dives", [])

    context = "DEEP DIVE ARTIKEL:\n"
    for dd in deep_dives:
        context += f"- {dd['title']}: {dd.get('analysis', dd.get('summary', ''))}\n"

    prompt = f"""Du bist ein Senior-Analyst bei einer großen Investmentbank.

SIGNIFIKANTES MARKTEREIGNIS:
{event_summary}

{context}

AUFGABE:
Erstelle eine tiefgreifende Analyse (auf Deutsch) mit folgenden Punkten:
1. **Was ist passiert?** – Klare Darstellung des Ereignisses
2. **Warum ist es bedeutend?** – Einordnung in den größeren Kontext
3. **Kausalkette:** Welche Dominoeffekte könnten folgen?
4. **Sektoren im Fokus:** Welche Branchen/Unternehmen sind betroffen?
5. **Einschätzung:** Kurzfristige vs. langfristige Auswirkungen

Formatiere als fließenden Text, keine Bullet-Points. Maximal 300 Wörter.
Professionell, aber verständlich.
"""

    log.info("🧠 Claude Deep Analysis wird durchgeführt...")
    return call_claude(prompt, api_key)


def generate_market_outlook(gemini_result: dict, api_key: str) -> str:
    """Generiert mit Gemini einen kurzen Marktausblick."""
    sentiment = gemini_result.get("market_sentiment", "neutral")
    headlines = gemini_result.get("headlines", [])
    deep_dives = gemini_result.get("deep_dives", [])

    headline_text = "\n".join([f"- {h['title']}" for h in headlines[:5]])
    dive_text = "\n".join([f"- {d['title']}" for d in deep_dives])

    prompt = f"""Basierend auf der heutigen Nachrichtenlage, erstelle einen kurzen Marktausblick auf Deutsch.

STIMMUNG: {sentiment}

TOP-HEADLINES:
{headline_text}

FINANZ-THEMEN:
{dive_text}

Schreibe maximal 4-5 Sätze. Professionell, objektiv, zukunftsgerichtet.
Was sollte ein Anleger heute im Blick behalten?
Antworte nur mit dem Text, keine Überschriften oder Formatierung.
"""

    return call_gemini(prompt, api_key)


# ─── Markdown-Generierung ────────────────────────────────────────────────────


def build_newsletter_markdown(
    gemini_result: dict,
    claude_analysis: str,
    market_outlook: str,
) -> str:
    """Baut den finalen Newsletter im Markdown-Format."""
    now = datetime.now(timezone(timedelta(hours=1)))  # MEZ
    date_str = now.strftime("%d. %B %Y")
    # Deutsche Monatsnamen
    months_de = {
        "January": "Januar", "February": "Februar", "March": "März",
        "April": "April", "May": "Mai", "June": "Juni",
        "July": "Juli", "August": "August", "September": "September",
        "October": "Oktober", "November": "November", "December": "Dezember",
    }
    for en, de in months_de.items():
        date_str = date_str.replace(en, de)

    sentiment = gemini_result.get("market_sentiment", "neutral")
    sentiment_emoji = {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡"}.get(sentiment, "🟡")

    md = []
    md.append(f"# 📰 NewsBot – Finanz-Newsletter")
    md.append(f"**{date_str}** | Marktstimmung: {sentiment_emoji} {sentiment.capitalize()}")
    md.append("")
    md.append("---")
    md.append("")

    # ── Headline-Ticker ──
    md.append("## 🗞️ Headline-Ticker")
    md.append("")
    headlines = gemini_result.get("headlines", [])
    for i, h in enumerate(headlines, 1):
        link = h.get("link", "")
        title = h["title"]
        if link:
            md.append(f"**{i}.** [{title}]({link})")
        else:
            md.append(f"**{i}.** {title}")
        md.append(f"   {h.get('summary', '')} *({h.get('source', 'N/A')})*")
        md.append("")

    md.append("---")
    md.append("")

    # ── Deep Dive Finanzen/Banken ──
    md.append("## 🏦 Deep Dive – Finanzen & Banken")
    md.append("")

    deep_dives = gemini_result.get("deep_dives", [])
    if deep_dives:
        for dd in deep_dives:
            link = dd.get("link", "")
            title = dd["title"]
            if link:
                md.append(f"### [{title}]({link})")
            else:
                md.append(f"### {title}")
            md.append("")
            md.append(dd.get("analysis", dd.get("summary", "")))
            md.append(f"*Quelle: {dd.get('source', 'N/A')}*")
            md.append("")
    else:
        md.append("*Heute keine signifikanten Finanz-Themen für einen Deep Dive.*")
        md.append("")

    # ── Claude Deep Analysis (falls vorhanden) ──
    if claude_analysis:
        md.append("#### 🧠 KI Deep Analysis (signifikantes Marktereignis)")
        md.append("")
        md.append(claude_analysis)
        md.append("")

    md.append("---")
    md.append("")

    # ── Wirtschaft Kompakt ──
    md.append("## 📊 Wirtschaft Kompakt")
    md.append("")
    wirtschaft = gemini_result.get("wirtschaft_kompakt", [])
    if wirtschaft:
        for w in wirtschaft:
            md.append(f"- **{w['title']}** – {w.get('summary', '')} *({w.get('source', 'N/A')})*")
    else:
        md.append("*Keine Wirtschaftsmeldungen verfügbar.*")
    md.append("")

    md.append("---")
    md.append("")

    # ── Marktausblick ──
    md.append("## 🔮 Marktausblick")
    md.append("")
    if market_outlook:
        md.append(market_outlook)
    else:
        md.append("*Kein Marktausblick verfügbar.*")
    md.append("")

    md.append("---")
    md.append("")
    md.append(f"*Generiert von [NewsBot](https://github.com/mathiskordy/NewsBot) am {date_str} um {now.strftime('%H:%M')} Uhr MEZ*")
    md.append("")
    md.append("*⚡ Powered by Gemini Flash & Claude Opus | 📡 RSS Feeds | 🤖 GitHub Actions*")

    return "\n".join(md)


# ─── Output-Kanäle ───────────────────────────────────────────────────────────


def post_github_issue(markdown: str, repo: str, token: str) -> bool:
    """Erstellt ein neues GitHub Issue mit dem Newsletter."""
    now = datetime.now(timezone(timedelta(hours=1)))
    date_str = now.strftime("%d.%m.%Y")

    url = f"https://api.github.com/repos/{repo}/issues"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }
    data = {
        "title": f"📰 Newsletter – {date_str}",
        "body": markdown,
        "labels": ["newsletter"],
    }

    try:
        # Stelle sicher, dass das Label existiert
        label_url = f"https://api.github.com/repos/{repo}/labels"
        requests.post(
            label_url,
            headers=headers,
            json={"name": "newsletter", "color": "0075ca", "description": "Automatischer Newsletter"},
        )
    except Exception:
        pass  # Label existiert möglicherweise schon

    try:
        resp = requests.post(url, headers=headers, json=data, timeout=30)
        if resp.status_code == 201:
            issue_url = resp.json().get("html_url", "")
            log.info(f"✅ GitHub Issue erstellt: {issue_url}")
            return True
        else:
            log.error(f"❌ GitHub Issue Fehler: {resp.status_code} – {resp.text}")
            return False
    except Exception as e:
        log.error(f"❌ GitHub Issue Fehler: {e}")
        return False


def build_newsletter_html(
    gemini_result: dict,
    claude_analysis: str,
    market_outlook: str,
) -> str:
    """Baut den Newsletter als professionelle HTML-E-Mail."""
    now = datetime.now(timezone(timedelta(hours=1)))
    date_str = now.strftime("%d.%m.%Y")
    months_de = {
        "January": "Januar", "February": "Februar", "March": "März",
        "April": "April", "May": "Mai", "June": "Juni",
        "July": "Juli", "August": "August", "September": "September",
        "October": "Oktober", "November": "November", "December": "Dezember",
    }
    long_date = now.strftime("%d. %B %Y")
    for en, de in months_de.items():
        long_date = long_date.replace(en, de)
    weekdays_de = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]
    weekday = weekdays_de[now.weekday()]

    sentiment = gemini_result.get("market_sentiment", "neutral")
    sentiment_config = {
        "bullish": {"label": "Bullish", "color": "#10b981", "bg": "#ecfdf5", "icon": "🟢"},
        "bearish": {"label": "Bearish", "color": "#ef4444", "bg": "#fef2f2", "icon": "🔴"},
        "neutral": {"label": "Neutral", "color": "#f59e0b", "bg": "#fffbeb", "icon": "🟡"},
    }.get(sentiment, {"label": "Neutral", "color": "#f59e0b", "bg": "#fffbeb", "icon": "🟡"})

    def short_source(link, source_name):
        """Erzeugt einen kurzen, klickbaren Quellen-Link."""
        if link:
            return f'<a href="{link}" style="color:#6366f1;text-decoration:none;font-weight:500;">{source_name} →</a>'
        return f'<span style="color:#94a3b8;">{source_name}</span>'

    # ── Headlines bauen ──
    headlines_html = ""
    for i, h in enumerate(gemini_result.get("headlines", []), 1):
        title = h["title"]
        link = h.get("link", "")
        summary = h.get("summary", "")
        source = h.get("source", "")
        title_html = f'<a href="{link}" style="color:#1e293b;text-decoration:none;">{title}</a>' if link else title
        headlines_html += f"""
        <tr>
            <td style="padding:12px 16px;border-bottom:1px solid #f1f5f9;">
                <table cellpadding="0" cellspacing="0" border="0" width="100%"><tr>
                    <td style="width:32px;vertical-align:top;padding-top:2px;">
                        <span style="display:inline-block;width:26px;height:26px;border-radius:50%;background:linear-gradient(135deg,#6366f1,#8b5cf6);color:#fff;text-align:center;line-height:26px;font-size:12px;font-weight:700;">{i}</span>
                    </td>
                    <td style="padding-left:12px;">
                        <div style="font-size:15px;font-weight:600;color:#1e293b;line-height:1.4;">{title_html}</div>
                        <div style="font-size:13px;color:#64748b;margin-top:4px;line-height:1.5;">{summary}</div>
                        <div style="margin-top:6px;">{short_source(link, source)}</div>
                    </td>
                </tr></table>
            </td>
        </tr>"""

    # ── Deep Dives bauen ──
    deep_dives_html = ""
    for dd in gemini_result.get("deep_dives", []):
        title = dd["title"]
        link = dd.get("link", "")
        analysis = dd.get("analysis", dd.get("summary", ""))
        source = dd.get("source", "")
        title_html = f'<a href="{link}" style="color:#1e293b;text-decoration:none;">{title}</a>' if link else title
        deep_dives_html += f"""
        <div style="background:#f8fafc;border-radius:10px;padding:20px;margin-bottom:12px;border-left:4px solid #6366f1;">
            <div style="font-size:16px;font-weight:700;color:#1e293b;margin-bottom:8px;">{title_html}</div>
            <div style="font-size:14px;color:#475569;line-height:1.6;">{analysis}</div>
            <div style="margin-top:10px;">{short_source(link, source)}</div>
        </div>"""

    if not deep_dives_html:
        deep_dives_html = '<div style="color:#94a3b8;font-style:italic;padding:16px;">Heute keine signifikanten Finanz-Themen für einen Deep Dive.</div>'

    # ── Claude Analysis ──
    claude_html = ""
    if claude_analysis:
        claude_html = f"""
        <div style="background:linear-gradient(135deg,#fef3c7,#fde68a);border-radius:10px;padding:20px;margin-top:16px;border:1px solid #f59e0b;">
            <div style="font-size:14px;font-weight:700;color:#92400e;margin-bottom:10px;">🧠 KI Deep Analysis – Signifikantes Marktereignis</div>
            <div style="font-size:14px;color:#78350f;line-height:1.7;">{claude_analysis}</div>
        </div>"""

    # ── Wirtschaft Kompakt ──
    wirtschaft_html = ""
    for w in gemini_result.get("wirtschaft_kompakt", []):
        title = w["title"]
        summary = w.get("summary", "")
        source = w.get("source", "")
        wirtschaft_html += f"""
        <tr>
            <td style="padding:10px 16px;border-bottom:1px solid #f1f5f9;">
                <div style="font-size:14px;"><span style="font-weight:600;color:#1e293b;">{title}</span> <span style="color:#64748b;">– {summary}</span></div>
                <div style="font-size:12px;color:#94a3b8;margin-top:3px;">{source}</div>
            </td>
        </tr>"""

    if not wirtschaft_html:
        wirtschaft_html = '<tr><td style="padding:16px;color:#94a3b8;font-style:italic;">Keine Wirtschaftsmeldungen verfügbar.</td></tr>'

    # ── Marktausblick ──
    outlook_html = market_outlook if market_outlook else "Kein Marktausblick verfügbar."

    # ── Gesamtes HTML zusammenbauen ──
    html = f"""<!DOCTYPE html>
<html lang="de">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1.0"></head>
<body style="margin:0;padding:0;background:#f1f5f9;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif;">
<table cellpadding="0" cellspacing="0" border="0" width="100%" style="background:#f1f5f9;">
<tr><td align="center" style="padding:24px 16px;">
<table cellpadding="0" cellspacing="0" border="0" width="600" style="max-width:600px;background:#ffffff;border-radius:16px;overflow:hidden;box-shadow:0 4px 24px rgba(0,0,0,0.08);">

    <!-- ══ HEADER ══ -->
    <tr><td style="background:linear-gradient(135deg,#1e1b4b,#312e81,#4338ca);padding:32px 28px;">
        <div style="font-size:28px;font-weight:800;color:#ffffff;letter-spacing:-0.5px;">📰 NewsBot</div>
        <div style="font-size:14px;color:#c7d2fe;margin-top:4px;">{weekday}, {long_date}</div>
        <div style="margin-top:16px;">
            <span style="display:inline-block;background:{sentiment_config['bg']};color:{sentiment_config['color']};padding:6px 14px;border-radius:20px;font-size:13px;font-weight:600;">
                {sentiment_config['icon']} Marktstimmung: {sentiment_config['label']}
            </span>
        </div>
    </td></tr>

    <!-- ══ HEADLINE-TICKER ══ -->
    <tr><td style="padding:28px 28px 8px 28px;">
        <div style="font-size:11px;font-weight:700;color:#6366f1;text-transform:uppercase;letter-spacing:1.5px;">🗞️ Headline-Ticker</div>
    </td></tr>
    <tr><td style="padding:0 12px 20px 12px;">
        <table cellpadding="0" cellspacing="0" border="0" width="100%">
            {headlines_html}
        </table>
    </td></tr>

    <!-- ══ DEEP DIVE ══ -->
    <tr><td style="padding:8px 28px;">
        <div style="font-size:11px;font-weight:700;color:#6366f1;text-transform:uppercase;letter-spacing:1.5px;">🏦 Deep Dive – Finanzen &amp; Banken</div>
    </td></tr>
    <tr><td style="padding:12px 28px 20px 28px;">
        {deep_dives_html}
        {claude_html}
    </td></tr>

    <!-- ══ WIRTSCHAFT KOMPAKT ══ -->
    <tr><td style="padding:8px 28px;">
        <div style="font-size:11px;font-weight:700;color:#6366f1;text-transform:uppercase;letter-spacing:1.5px;">📊 Wirtschaft Kompakt</div>
    </td></tr>
    <tr><td style="padding:0 12px 20px 12px;">
        <table cellpadding="0" cellspacing="0" border="0" width="100%" style="background:#f8fafc;border-radius:10px;overflow:hidden;">
            {wirtschaft_html}
        </table>
    </td></tr>

    <!-- ══ MARKTAUSBLICK ══ -->
    <tr><td style="padding:8px 28px;">
        <div style="font-size:11px;font-weight:700;color:#6366f1;text-transform:uppercase;letter-spacing:1.5px;">🔮 Marktausblick</div>
    </td></tr>
    <tr><td style="padding:12px 28px 28px 28px;">
        <div style="background:linear-gradient(135deg,#eef2ff,#e0e7ff);border-radius:10px;padding:20px;border:1px solid #c7d2fe;">
            <div style="font-size:14px;color:#3730a3;line-height:1.7;">{outlook_html}</div>
        </div>
    </td></tr>

    <!-- ══ FOOTER ══ -->
    <tr><td style="background:#f8fafc;padding:20px 28px;border-top:1px solid #e2e8f0;">
        <div style="font-size:12px;color:#94a3b8;text-align:center;line-height:1.6;">
            Generiert von <strong style="color:#6366f1;">NewsBot</strong> am {long_date} um {now.strftime('%H:%M')} Uhr MEZ<br>
            ⚡ Gemini Flash &amp; Claude Opus · 📡 RSS Feeds · 🤖 GitHub Actions
        </div>
    </td></tr>

</table>
</td></tr>
</table>
</body>
</html>"""

    return html


def send_email(
    markdown: str,
    gemini_result: dict = None,
    claude_analysis: str = "",
    market_outlook: str = "",
    smtp_server: str = "smtp.gmail.com",
    smtp_port: int = 587,
) -> bool:
    """Sendet den Newsletter als professionelle HTML-E-Mail."""
    email_address = os.environ.get("EMAIL_ADDRESS", "")
    email_password = os.environ.get("EMAIL_PASSWORD", "")
    email_recipient = os.environ.get("EMAIL_RECIPIENT", "")

    if not all([email_address, email_password, email_recipient]):
        log.warning("⚠️  E-Mail-Konfiguration unvollständig, überspringe E-Mail-Versand")
        return False

    now = datetime.now(timezone(timedelta(hours=1)))
    date_str = now.strftime("%d.%m.%Y")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"=?utf-8?Q?=F0=9F=93=B0?= NewsBot – Finanz-Briefing {date_str}"
    msg["From"] = email_address
    msg["To"] = email_recipient

    # Plain-Text Fallback (wird angezeigt wenn HTML nicht geht)
    msg.attach(MIMEText(markdown, "plain", "utf-8"))

    # HTML-Version (professionell gestylt) – wird immer angehängt
    if gemini_result:
        html_content = build_newsletter_html(gemini_result, claude_analysis, market_outlook)
    else:
        # Minimales HTML-Fallback aus Markdown
        html_content = f"<html><body><pre>{markdown}</pre></body></html>"

    html_part = MIMEText(html_content, "html", "utf-8")
    html_part.replace_header("Content-Type", "text/html; charset=utf-8")
    msg.attach(html_part)

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(email_address, email_password)
            server.sendmail(email_address, email_recipient, msg.as_string())
        log.info(f"✅ E-Mail gesendet an {email_recipient}")
        return True
    except Exception as e:
        log.error(f"❌ E-Mail Fehler: {e}")
        return False


# ─── Hauptprogramm ───────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="NewsBot – Täglicher Finanz-Newsletter")
    parser.add_argument("--dry-run", action="store_true", help="Nur Feeds abrufen, keine KI-Calls")
    parser.add_argument("--output", choices=["github", "email", "both", "stdout"], default="github",
                        help="Ausgabekanal (Standard: github)")
    parser.add_argument("--feeds", default="feeds.yaml", help="Pfad zur Feed-Konfiguration")
    args = parser.parse_args()

    log.info("🚀 NewsBot startet...")
    log.info(f"⏰ {datetime.now(timezone(timedelta(hours=1))).strftime('%d.%m.%Y %H:%M')} MEZ")

    # ── 1. Feeds laden und crawlen ──
    feeds = load_feeds(args.feeds)
    log.info(f"📋 {len(feeds)} Feeds konfiguriert")

    articles = crawl_all_feeds(feeds)
    log.info(f"📰 {len(articles)} Artikel nach Deduplizierung")

    if not articles:
        log.error("❌ Keine Artikel gefunden – Newsletter kann nicht erstellt werden")
        sys.exit(1)

    if args.dry_run:
        log.info("🏃 Dry-Run Modus – Stoppe hier")
        log.info(f"Erste 5 Artikel:")
        for a in articles[:5]:
            log.info(f"  [{a['category']}] {a['title']} ({a['source']})")
        return

    # ── 2. API-Keys prüfen ──
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    claude_key = os.environ.get("ANTHROPIC_API_KEY", "")

    if not gemini_key:
        log.error("❌ GEMINI_API_KEY nicht gesetzt!")
        sys.exit(1)

    # ── 3. Gemini-Verarbeitung ──
    gemini_result = process_with_gemini(articles, gemini_key)

    # ── 4. Claude Deep Analysis (nur bei signifikanten Events) ──
    claude_analysis = ""
    if claude_key and gemini_result.get("significant_events", False):
        claude_analysis = deep_analysis_with_claude(gemini_result, claude_key)
    elif not claude_key:
        log.info("ℹ️  ANTHROPIC_API_KEY nicht gesetzt – Claude Deep Analysis deaktiviert")

    # ── 5. Marktausblick generieren ──
    market_outlook = generate_market_outlook(gemini_result, gemini_key)

    # ── 6. Newsletter zusammenbauen ──
    newsletter = build_newsletter_markdown(gemini_result, claude_analysis, market_outlook)

    log.info(f"📝 Newsletter generiert ({len(newsletter)} Zeichen)")

    # ── 7. Ausgabe ──
    success = False

    if args.output in ("github", "both"):
        repo = os.environ.get("GITHUB_REPOSITORY", "")
        token = os.environ.get("GITHUB_TOKEN", "")
        if repo and token:
            success = post_github_issue(newsletter, repo, token) or success
        else:
            log.warning("⚠️  GITHUB_REPOSITORY oder GITHUB_TOKEN nicht gesetzt")

    if args.output in ("email", "both"):
        success = send_email(
            newsletter,
            gemini_result=gemini_result,
            claude_analysis=claude_analysis,
            market_outlook=market_outlook,
        ) or success

    if args.output == "stdout":
        print(newsletter)
        success = True

    if not success and args.output != "stdout":
        log.warning("⚠️  Kein Ausgabekanal erfolgreich – gebe auf stdout aus")
        print(newsletter)

    log.info("✅ NewsBot fertig!")


if __name__ == "__main__":
    main()
