
from flask import Flask, jsonify, request, render_template
import requests
import re

app = Flask(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:1.5b"

TOTAL_QUESTIONS = 25
QUESTIONS_PER_LEVEL = 5

DIFFICULTY_LEVELS = {
    1: "Very Easy",
    2: "Easy",
    3: "Medium",
    4: "Hard",
    5: "Expert"
}


def get_difficulty(question_number):
    level = min(
        ((question_number - 1) // QUESTIONS_PER_LEVEL) + 1,
        5
    )
    return {
        "level": level,
        "name": DIFFICULTY_LEVELS[level]
    }


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "AI Interviewer API is running!",
        "status": "success"
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})


@app.route("/generate-questions", methods=["POST"])
def generate_questions():
    data = request.get_json(silent=True) or {}

    role = data.get("role", "")
    skills = data.get("skills", [])

    if not isinstance(role, str) or not role.strip():
        return jsonify({"error": "Role is required"}), 400

    if not isinstance(skills, list):
        return jsonify({"error": "Skills must be a list"}), 400

    role = role.strip()
    skills = [
        skill.strip()
        for skill in skills
        if isinstance(skill, str) and skill.strip()
    ]

    difficulty_plan = "\n".join(
        f"Questions {start}-{min(start + 4, TOTAL_QUESTIONS)}: "
        f"Level {level} - {DIFFICULTY_LEVELS[level]}"
        for level, start in enumerate(range(1, TOTAL_QUESTIONS + 1, 5), start=1)
    )

    prompt = f"""
You are an expert technical interviewer.

Generate exactly {TOTAL_QUESTIONS} technical interview questions
for the job role: {role}

Candidate skills: {", ".join(skills) if skills else role}

Follow this difficulty progression:
{difficulty_plan}

Requirements:
- Generate 5 questions for each difficulty level.
- Keep questions relevant to the role and candidate skills.
- Level 1: basic definitions and fundamentals.
- Level 2: simple explanations and basic applications.
- Level 3: practical problem-solving and implementation.
- Level 4: complex scenarios and deeper technical reasoning.
- Level 5: advanced architecture, optimization, and expert scenarios.
- Increase difficulty gradually between levels.
- Do not repeat questions.
- Return only numbered questions, one per line.
- Do not add headings or explanations.
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=300
        )
        response.raise_for_status()

        questions_text = response.json().get("response", "").strip()

        if not questions_text:
            return jsonify({
                "error": "The AI returned an empty response"
            }), 502

        questions = []

        for line in questions_text.splitlines():
            line = line.strip()
            line = re.sub(
                r"^\s*(?:question\s*)?\d+\s*[\.\):\-]\s*",
                "",
                line,
                flags=re.IGNORECASE
            )
            line = line.strip().strip('"').strip("'").strip("*").strip()

            if line:
                questions.append(line)

        questions = questions[:TOTAL_QUESTIONS]

        if not questions:
            return jsonify({
                "error": "No usable questions were generated"
            }), 502

        question_details = [
            {
                "question_number": index,
                "difficulty": get_difficulty(index)
            }
            for index in range(1, len(questions) + 1)
        ]

        return jsonify({
            "role": role,
            "skills": skills,
            "questions": questions,
            "question_details": question_details,
            "total_questions": len(questions)
        })

    except requests.RequestException as error:
        return jsonify({
            "error": f"Could not connect to Ollama: {error}"
        }), 503


@app.route("/interview", methods=["GET"])
def interview():
    return render_template("interview.html")


if __name__ == "__main__":
    app.run(debug=True)
