# SET50 Futures Calendar Spread Analysis

Automated analysis tool for SET50 futures calendar spreads with Telegram notifications.

## Features

- Analyzes multiple calendar spread pairs (G26-H26, H26-M26, U26-Z26)
- Calculates theoretical pricing using Cost-of-Carry model
- Identifies mispricing opportunities
- Generates analysis charts
- Sends automated reports to Telegram

## Local Setup

1. Clone the repository
2. Create virtual environment: `python -m venv venv`
3. Activate: `source venv/bin/activate`
4. Install: `pip install -r requirements.txt`
5. Copy `.env.example` to `.env` and add your Telegram credentials
6. Run: `python main.py`

## Deploy to Railway

1. Push to GitHub
2. Go to railway.app → New Project → Deploy from GitHub
3. Add environment variables:
   - `TELEGRAM_API_TOKEN`
   - `TELEGRAM_CHAT_ID`
4. Set cron schedule if needed (e.g., `0 9 * * 1-5` for 9 AM weekdays)
