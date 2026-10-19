

from flask import Flask, jsonify, request
import requests
import re

app = Flask(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:1.5b"


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

    prompt = f"""
You are an expert technical interviewer.
Generate exactly 3 interview questions for the role: {role}
Candidate skills: {", ".join(skills) if skills else role}
Return only the numbered questions, one per line.
Do not add headings or explanations.
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=180
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
            line = re.sub(r"^\d+[\.\)]\s*", "", line)
            line = line.strip().strip('"').strip("'").strip("*").strip()

            if line:
                questions.append(line)

        questions = questions[:3]

        if not questions:
            return jsonify({
                "error": "No usable questions were generated"
            }), 502

        return jsonify({
            "role": role,
            "skills": skills,
            "questions": questions
        })

    except requests.RequestException as error:
        return jsonify({
            "error": f"Could not connect to Ollama: {error}"
        }), 503


if __name__ == "__main__":
    app.run(debug=True)
