# 📰 NewsBot – Automatisierter Finanz-Newsletter

Dein persönlicher KI-gestützter Finanz-Newsletter, der jeden Morgen um **08:00 Uhr MEZ** automatisch generiert wird.

## ✨ Features

- **15+ RSS-Feeds** aus Weltgeschehen, Wirtschaft und Finanzen (Reuters, Bloomberg, FT, Handelsblatt, u.v.m.)
- **Gemini Flash** für schnelle Zusammenfassung und Kategorisierung (kostenlos im Free-Tier)
- **Claude Deep Analysis** bei signifikanten Marktbewegungen (Zinsänderungen, Crashs, etc.)
- **Automatischer täglicher Lauf** via GitHub Actions (kostenlos)
- **Deduplizierung** – keine doppelten Nachrichten
- **Zwei Ausgabekanäle**: GitHub Issue oder E-Mail

## 🏗️ Newsletter-Struktur

| Sektion | Inhalt |
|---------|--------|
| 🗞️ Headline-Ticker | Top 5-8 Schlagzeilen des Tages |
| 🏦 Deep Dive Finanzen | 2-3 tiefere Analysen aus dem Bankensektor |
| 📊 Wirtschaft Kompakt | Kurzüberblick Wirtschaftsmeldungen |
| 🔮 Marktausblick | KI-generierter Ausblick für den Tag |

## 🚀 Setup

### 1. API-Keys besorgen

| Key | Wo? | Kosten |
|-----|-----|--------|
| Gemini API | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) | Kostenlos (Free-Tier) |
| Anthropic API | [console.anthropic.com](https://console.anthropic.com/) | Pay-per-use (optional) |

### 2. GitHub Secrets einrichten

Gehe zu **Settings → Secrets and variables → Actions** in deinem Repository und füge hinzu:

| Secret | Beschreibung | Pflicht? |
|--------|-------------|----------|
| `GEMINI_API_KEY` | Dein Google Gemini API Key | ✅ Ja |
| `ANTHROPIC_API_KEY` | Dein Anthropic/Claude API Key | ❌ Optional |
| `EMAIL_ADDRESS` | Gmail-Adresse für Versand | ❌ Nur für E-Mail |
| `EMAIL_PASSWORD` | Gmail App-Passwort | ❌ Nur für E-Mail |
| `EMAIL_RECIPIENT` | Empfänger-Adresse | ❌ Nur für E-Mail |

> **Hinweis:** `GITHUB_TOKEN` wird automatisch von GitHub Actions bereitgestellt.

### 3. Workflow aktivieren

Der Workflow läuft automatisch jeden Tag um 07:00 UTC (08:00 MEZ).

**Zum Testen:** Gehe zu **Actions → Daily Newsletter → Run workflow** und klicke auf den grünen Button.

## 🧪 Lokaler Test

```bash
# Dependencies installieren
pip install -r requirements.txt

# Nur Feeds testen (keine API-Keys nötig)
python newsletter.py --dry-run

# Vollständiger Newsletter auf stdout
GEMINI_API_KEY=dein_key python newsletter.py --output stdout

# Newsletter als GitHub Issue
GEMINI_API_KEY=dein_key GITHUB_TOKEN=dein_token GITHUB_REPOSITORY=user/repo python newsletter.py --output github
```

## 📡 Feeds anpassen

Bearbeite `feeds.yaml`, um Feeds hinzuzufügen oder zu entfernen:

```yaml
feeds:
  - name: "Mein Feed"
    url: "https://example.com/rss"
    category: "finanzen"  # weltgeschehen, wirtschaft, finanzen
    priority: 1            # 1=hoch, 3=niedrig
```

## 📁 Projektstruktur

```
NewsBot/
├── newsletter.py                  # Hauptskript
├── feeds.yaml                     # RSS-Feed-Konfiguration
├── requirements.txt               # Python-Dependencies
├── README.md                      # Diese Datei
└── .github/workflows/
    └── daily_news.yml             # GitHub Actions Workflow
```
