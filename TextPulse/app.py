
from flask import Flask, render_template, request, jsonify
import joblib
import sqlite3
import re
from datetime import datetime

app = Flask(__name__)

# Load the trained TextPulse model
model = joblib.load("models/textpulse_model.pkl")

DATABASE = "textpulse.db"

# Possible stress factors and related keywords
stress_keywords = {
    "Academic Pressure": [
        "exam", "assignment", "study", "college",
        "deadline", "marks", "homework", "project"
    ],
    "Work Pressure": [
        "work", "office", "job", "meeting",
        "manager", "workload", "career"
    ],
    "Financial Stress": [
        "money", "debt", "loan", "bills",
        "salary", "financial", "expenses"
    ],
    "Sleep Problems": [
        "sleep", "insomnia", "tired", "fatigue",
        "exhausted", "awake"
    ]
}

recommendations = {
    "Academic Pressure": [
        "Create a simple study schedule.",
        "Divide large assignments into smaller tasks.",
        "Take short breaks between study sessions."
    ],
    "Work Pressure": [
        "Prioritize your important tasks.",
        "Break large work into smaller steps.",
        "Discuss your workload with someone you trust."
    ],
    "Financial Stress": [
        "Prepare a monthly budget.",
        "List essential and non-essential expenses.",
        "Consider reliable financial guidance."
    ],
    "Sleep Problems": [
        "Maintain a regular sleep schedule.",
        "Reduce screen use before bedtime.",
        "Try a relaxing activity before sleeping."
    ]
}

def get_db():
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    connection = get_db()

    connection.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entered_text TEXT NOT NULL,
            result TEXT NOT NULL,
            factors TEXT,
            created_at TEXT NOT NULL
        )
    """)

    connection.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id INTEGER NOT NULL,
            response TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (analysis_id) REFERENCES analyses(id)
        )
    """)

    connection.commit()
    connection.close()


def is_greeting(text):
    cleaned = re.sub(r"[^\w\s]", "", text.lower()).strip()

    greetings = {
        "hi", "hello", "hey", "good morning",
        "good afternoon", "good evening",
        "hi there", "hello there"
    }

    return cleaned in greetings


def identify_factors(text):
    lower_text = text.lower()
    found = []

    for factor, keywords in stress_keywords.items():
        matched = [
            word for word in keywords
            if re.search(r"\b" + re.escape(word) + r"\b", lower_text)
        ]

        if matched:
            found.append({
                "name": factor,
                "indicators": matched
            })

    return found


def generate_action_plan(text, factors):
    lower_text = text.lower()
    tasks = []

    if "Academic Pressure" in factors:
        if "assignment" in lower_text or "homework" in lower_text:
            tasks.append("List your pending assignments and their deadlines.")

        if "exam" in lower_text or "study" in lower_text:
            tasks.append("Create a short study schedule for your upcoming exams.")

        tasks.append("Divide your academic work into smaller manageable steps.")

    if "Work Pressure" in factors:
        tasks.append("List your pending work and identify the most urgent task.")
        tasks.append("Organize your work into smaller achievable steps.")

    if "Financial Stress" in factors:
        tasks.append("Write down your essential expenses and monthly budget.")

    if "Sleep Problems" in factors:
        tasks.append("Plan a consistent time for sleeping and waking up.")

    if not tasks:
        tasks = [
            "Write down what is worrying you.",
            "Identify one small step you can take next.",
            "Take a short break and revisit the situation."
        ]

    # Remove duplicate tasks while retaining their order
    return list(dict.fromkeys(tasks))


def analyze_text(text):
    if not text:
        return {
            "status": "invalid",
            "message": "Please enter some text."
        }

    if is_greeting(text):
        return {
            "status": "greeting",
            "message": (
                "Hello! Welcome to TextPulse. "
                "Please describe your situation so I can "
                "analyze possible stress-related patterns."
            )
        }

    if len(text.split()) < 4:
        return {
            "status": "invalid",
            "message": (
                "Please describe your situation in at least "
                "four words to get a meaningful analysis."
            )
        }

    # SVM model prediction
    prediction = int(model.predict([text])[0])

    if prediction == 1:
        result = "Stress-related patterns detected."
    else:
        result = "No strong stress-related pattern detected."

    # Identify possible factors separately from the ML prediction
    detected = identify_factors(text)
    factors = [item["name"] for item in detected]

    tips = []
    for factor in factors:
        tips.extend(recommendations[factor])

    if prediction == 1 and not factors:
        tips = [
            "Take a short break.",
            "Try a relaxing breathing exercise.",
            "Talk to someone you trust."
        ]

    action_plan = generate_action_plan(text, factors)

    return {
        "status": "success",
        "result": result,
        "prediction": prediction,
        "factors": detected,
        "recommendations": list(dict.fromkeys(tips)),
        "action_plan": action_plan
    }


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()

    if len(text) > 5000:
        return jsonify({
            "status": "invalid",
            "message": "Please enter text shorter than 5000 characters."
        }), 400

    result = analyze_text(text)

    # Greetings and invalid inputs are not saved as stress analyses
    if result["status"] != "success":
        return jsonify(result)

    connection = get_db()

    cursor = connection.execute("""
        INSERT INTO analyses
        (entered_text, result, factors, created_at)
        VALUES (?, ?, ?, ?)
    """, (
        text,
        result["result"],
        ", ".join(item["name"] for item in result["factors"]),
        datetime.now().isoformat(timespec="seconds")
    ))

    result["analysis_id"] = cursor.lastrowid

    connection.commit()
    connection.close()

    return jsonify(result)


@app.route("/api/history", methods=["GET"])
def history():
    connection = get_db()

    rows = connection.execute("""
        SELECT * FROM analyses ORDER BY id DESC LIMIT 50
    """).fetchall()

    connection.close()

    return jsonify([
        dict(row) for row in rows
    ])


@app.route("/api/feedback", methods=["POST"])
def save_feedback():
    data = request.get_json(silent=True) or {}

    analysis_id = data.get("analysis_id")
    response = data.get("response")

    if not isinstance(analysis_id, int) or response not in [
        "correct", "incorrect"
    ]:
        return jsonify({
            "status": "invalid",
            "message": "Invalid feedback details."
        }), 400

    connection = get_db()

    exists = connection.execute(
        "SELECT id FROM analyses WHERE id = ?",
        (analysis_id,)
    ).fetchone()

    if not exists:
        connection.close()
        return jsonify({
            "status": "invalid",
            "message": "Analysis not found."
        }), 404

    connection.execute("""
        INSERT INTO feedback (analysis_id, response, created_at)
        VALUES (?, ?, ?)
    """, (
        analysis_id,
        response,
        datetime.now().isoformat(timespec="seconds")
    ))

    connection.commit()
    connection.close()

    return jsonify({
        "status": "success",
        "message": "Thank you for your feedback!"
    })


if __name__ == "__main__":
    initialize_database()
    app.run(debug=True)
