"""AI Smart Travel and Budget Planner - FastAPI + SQLite + Gemini."""
import json, os, sqlite3
from datetime import date
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.linear_model import LinearRegression

load_dotenv()
DB = Path(__file__).parent / "planner.db"
CATEGORIES = ["Stay", "Food", "Transport", "Activities", "Shopping", "Other"]
app = FastAPI(title="AI Smart Travel and Budget Planner")


def db():
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    return con


with db() as con:
    con.executescript("""
    CREATE TABLE IF NOT EXISTS trips(id INTEGER PRIMARY KEY, name TEXT, destination TEXT,
        days INTEGER, budget REAL, start_date TEXT);
    CREATE TABLE IF NOT EXISTS expenses(id INTEGER PRIMARY KEY, trip_id INTEGER, category TEXT,
        amount REAL, note TEXT, spent_on TEXT);
    CREATE TABLE IF NOT EXISTS plans(trip_id INTEGER PRIMARY KEY, content TEXT);
    """)


class TripIn(BaseModel):
    name: str
    destination: str
    days: int = Field(gt=0, le=60)
    budget: float = Field(gt=0)
    start_date: str = Field(default_factory=lambda: date.today().isoformat())


class ExpenseIn(BaseModel):
    category: str
    amount: float = Field(gt=0)
    note: str = ""
    spent_on: str = Field(default_factory=lambda: date.today().isoformat())


def get_trip(trip_id: int):
    with db() as con:
        row = con.execute("SELECT * FROM trips WHERE id=?", (trip_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Trip not found")
    return dict(row)


# ---- Module 1: trips ----
@app.post("/api/trips")
def create_trip(t: TripIn):
    with db() as con:
        cur = con.execute("INSERT INTO trips(name,destination,days,budget,start_date) VALUES(?,?,?,?,?)",
                          (t.name, t.destination, t.days, t.budget, t.start_date))
        return {"id": cur.lastrowid}


@app.get("/api/trips")
def list_trips():
    with db() as con:
        return [dict(r) for r in con.execute("SELECT * FROM trips ORDER BY id DESC")]


@app.delete("/api/trips/{trip_id}")
def delete_trip(trip_id: int):
    with db() as con:
        for table in ("expenses", "plans"):
            con.execute(f"DELETE FROM {table} WHERE trip_id=?", (trip_id,))
        con.execute("DELETE FROM trips WHERE id=?", (trip_id,))
    return {"ok": True}


# ---- Module 2: expense tracking ----
@app.post("/api/trips/{trip_id}/expenses")
def add_expense(trip_id: int, e: ExpenseIn):
    get_trip(trip_id)
    if e.category not in CATEGORIES:
        raise HTTPException(400, f"Category must be one of {CATEGORIES}")
    with db() as con:
        con.execute("INSERT INTO expenses(trip_id,category,amount,note,spent_on) VALUES(?,?,?,?,?)",
                    (trip_id, e.category, e.amount, e.note, e.spent_on))
    return {"ok": True}


@app.get("/api/trips/{trip_id}/expenses")
def list_expenses(trip_id: int):
    with db() as con:
        rows = con.execute("SELECT * FROM expenses WHERE trip_id=? ORDER BY spent_on DESC, id DESC", (trip_id,))
        return [dict(r) for r in rows]


@app.delete("/api/expenses/{expense_id}")
def delete_expense(expense_id: int):
    with db() as con:
        con.execute("DELETE FROM expenses WHERE id=?", (expense_id,))
    return {"ok": True}


# ---- Module 3: budget management ----
@app.get("/api/trips/{trip_id}/summary")
def summary(trip_id: int):
    trip = get_trip(trip_id)
    with db() as con:
        rows = con.execute("SELECT category, SUM(amount) s FROM expenses WHERE trip_id=? GROUP BY category",
                           (trip_id,)).fetchall()
    by_cat = {r["category"]: r["s"] for r in rows}
    spent = sum(by_cat.values())
    return {"budget": trip["budget"], "spent": spent, "remaining": trip["budget"] - spent,
            "percent_used": round(spent / trip["budget"] * 100, 1), "by_category": by_cat}


# ---- ML expense forecasting ----
@app.get("/api/trips/{trip_id}/forecast")
def forecast(trip_id: int):
    trip = get_trip(trip_id)
    with db() as con:
        rows = con.execute("SELECT spent_on, SUM(amount) a FROM expenses WHERE trip_id=? GROUP BY spent_on "
                           "ORDER BY spent_on", (trip_id,)).fetchall()
    if not rows:
        return {"points": [], "projected_total": 0, "overrun": False, "overrun_by": 0}
    first = date.fromisoformat(rows[0]["spent_on"])
    x = np.array([(date.fromisoformat(r["spent_on"]) - first).days + 1 for r in rows])
    y = np.cumsum([r["a"] for r in rows])
    if len(x) >= 3:  # linear trend on cumulative spend
        model = LinearRegression().fit(x.reshape(-1, 1), y)
        projected = float(model.predict([[trip["days"]]])[0])
    else:            # too little data: use average daily rate
        projected = float(y[-1] / x[-1] * trip["days"])
    projected = max(projected, float(y[-1]))
    points = [{"day": int(d), "spent": float(s)} for d, s in zip(x, y)]
    return {"points": points, "projected_total": round(projected, 2), "days": trip["days"],
            "overrun": projected > trip["budget"], "overrun_by": round(max(projected - trip["budget"], 0), 2)}


# ---- Module 4: Gemini itinerary, packing list, recommendations ----
def call_gemini(prompt: str) -> dict:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise HTTPException(400, "Set GEMINI_API_KEY in your .env file")

    model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()
    print("ACTUAL GEMINI MODEL:", repr(model))

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    try:
        r = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            params={"key": key},
            timeout=60,
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "responseMimeType": "application/json"
                }
            }
        )
    except requests.RequestException as exc:
        raise HTTPException(502, f"Could not connect to Gemini: {exc}")

    if r.status_code != 200:
        print("GEMINI API ERROR:", r.status_code, r.text)
        raise HTTPException(
            502,
            f"Gemini API error ({r.status_code}): {r.text[:500]}"
        )

    try:
        data = r.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(text)
    except (ValueError, KeyError, IndexError, TypeError):
        print("UNEXPECTED GEMINI RESPONSE:", r.text[:1000])
        raise HTTPException(
            502,
            "Gemini returned an unexpected response. Check the backend terminal."
        )


@app.post("/api/trips/{trip_id}/plan")
def generate_plan(trip_id: int):
    t = get_trip(trip_id)
    s = summary(trip_id)
    prompt = (f"Plan a {t['days']}-day trip to {t['destination']} with a total budget of {t['budget']} "
              f"(already spent {s['spent']}). Reply as JSON: "
              '{"itinerary":[{"day":1,"title":"","activities":[""],"estimated_cost":0}],'
              '"packing_list":[""],"budget_tips":[""],"recommendations":[""]}. '
              "Keep itinerary costs within budget.")
    plan = call_gemini(prompt)
    with db() as con:
        con.execute("INSERT OR REPLACE INTO plans(trip_id,content) VALUES(?,?)", (trip_id, json.dumps(plan)))
    return plan


@app.get("/api/trips/{trip_id}/plan")
def get_plan(trip_id: int):
    with db() as con:
        row = con.execute("SELECT content FROM plans WHERE trip_id=?", (trip_id,)).fetchone()
    return json.loads(row["content"]) if row else None


@app.get("/api/categories")
def categories():
    return CATEGORIES


app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")
