
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained TextPulse model
model = joblib.load("models/textpulse_model.pkl")


# Identify possible stress factors
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


# Recommendations for different stress factors
recommendations = {
    "Academic Pressure": [
        "Prepare a simple study schedule.",
        "Divide large tasks into smaller steps.",
        "Take short breaks while studying."
    ],

    "Work Pressure": [
        "Prioritize your important tasks.",
        "Take short breaks during work.",
        "Discuss your workload with someone you trust."
    ],

    "Financial Stress": [
        "Prepare a monthly budget.",
        "List essential and non-essential expenses.",
        "Consider seeking reliable financial guidance."
    ],

    "Sleep Problems": [
        "Maintain a regular sleep schedule.",
        "Avoid screens before bedtime.",
        "Try a relaxing activity before sleeping."
    ]
}


@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    entered_text = ""
    factors = []
    tips = []

    if request.method == "POST":

        entered_text = request.form.get("text", "").strip()

        if len(entered_text) < 15:
            result = "Please enter a meaningful sentence with more details."

        else:

            # Predict stress using the trained SVM model
            prediction = model.predict([entered_text])[0]

            if prediction == 1:

                result = "Stress-related patterns detected."

                lower_text = entered_text.lower()

                # Identify possible stress factors
                for factor, keywords in stress_keywords.items():

                    if any(word in lower_text for word in keywords):

                        factors.append(factor)
                        tips.extend(recommendations[factor])

                # General recommendations if no factor is identified
                if not factors:

                    factors.append("General Stress")

                    tips = [
                        "Take a short break.",
                        "Try deep breathing.",
                        "Talk to someone you trust."
                    ]

            else:

                result = "No strong stress-related pattern detected."

    return render_template(
        "index.html",
        result=result,
        entered_text=entered_text,
        factors=factors,
        tips=tips
    )


if __name__ == "__main__":
    app.run(debug=True)
