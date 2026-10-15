# AI Smart Travel and Budget Planner

Full-stack travel planner built with FastAPI and SQLite. It combines Gemini-generated itineraries with expense tracking, budget management and an ML spending forecast.

## Features
- **Itinerary planning** - Gemini generates day-by-day plans, packing lists and recommendations
- **Expense tracking** - log expenses by category and date
- **Budget management** - spent / remaining / percent used, with a category breakdown
- **Recommendations** - AI budget tips tailored to your trip
- **Forecasting** - scikit-learn linear regression on cumulative spend predicts budget overruns
- **Dashboards** - Chart.js charts update after every change

## Tech stack
Python, FastAPI, SQLite, scikit-learn, Gemini API, Chart.js

## Run locally
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then add your GEMINI_API_KEY
uvicorn main:app --reload
```
Open http://127.0.0.1:8000 (API docs at /docs). Get a free Gemini key at https://aistudio.google.com/apikey.
