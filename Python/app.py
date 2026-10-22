from flask import Flask, jsonify, request, render_template
import os
import random
import re
import requests

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "template")
)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:1.5b"

TOTAL_QUESTIONS = 25
MAX_RETRIES = 5
QUESTION_CATEGORIES = {
    "Role and skills": 10,
    "HR and behavioral": 5,
    "Project experience": 5,
    "Casual conversation": 5
}

DIFFICULTY_LEVELS = {
    1: "Very Easy",
    2: "Easy",
    3: "Medium",
    4: "Hard",
    5: "Expert"
}

JAVA_TOPICS = [
    "Java fundamentals and syntax",
    "Data types, variables, and type casting",
    "Operators and control flow",
    "Classes, objects, and constructors",
    "Encapsulation and access modifiers",
    "Inheritance and method overriding",
    "Polymorphism and abstraction",
    "Interfaces and abstract classes",
    "Method overloading and overriding",
    "Strings and StringBuilder",
    "Arrays and multidimensional arrays",
    "Collections Framework and List",
    "Set, Map, and HashMap",
    "Generics and type safety",
    "Exception handling",
    "Checked and unchecked exceptions",
    "Java 8 lambda expressions",
    "Functional interfaces and streams",
    "Optional and functional programming",
    "Multithreading and thread safety",
    "Synchronization and concurrency",
    "JVM, JRE, JDK, and bytecode",
    "Garbage collection and memory management",
    "Java I/O and file handling",
    "JDBC and database connectivity",
    "Enums, records, and nested classes",
    "equals, hashCode, and toString contracts",
    "Immutability and defensive copying",
    "Design patterns and SOLID principles",
    "Performance, debugging, and optimization"
]


def normalize_question(question):
    question = question.lower().strip()
    question = re.sub(
        r"^(?:question\s*)?\d+\s*[.):\-]\s*",
        "",
        question,
        flags=re.IGNORECASE
    )
    question = re.sub(r"[^\w\s]", "", question)
    return re.sub(r"\s+", " ", question).strip()


def parse_questions(text):
    questions = []
    seen = set()

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        # Remove numbering such as "1.", "1)", "1:", or "Question 1:"
        line = re.sub(
            r"^(?:question\s*)?\d+\s*[.):\-]\s*",
            "",
            line,
            flags=re.IGNORECASE
        )
        line = line.strip().strip('"').strip("'").strip("*").strip()

        # Keep only actual question lines.
        if not line or not line.endswith("?"):
            continue

        normalized = normalize_question(line)
        if normalized and normalized not in seen:
            seen.add(normalized)
            questions.append(line)

    return questions


def generate_from_ollama(prompt, token_limit):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "30m",
            "options": {
                "num_predict": token_limit,
                "temperature": 0.3
            }
        },
        timeout=300
    )
    response.raise_for_status()
    return response.json().get("response", "").strip()


def generate_technical_questions(role, skills, topic_text):
    questions = []
    seen = set()
    difficulty_guidance = {
        1: "Test core definitions and basic concepts with direct, introductory questions.",
        2: "Test straightforward applications, simple examples, and basic comparisons.",
        3: "Test practical problem-solving, multi-step application, or common debugging.",
        4: "Test complex real-world scenarios, debugging, design decisions, and trade-offs.",
        5: "Test advanced architecture, internals, scale, performance, and nuanced trade-offs."
    }
    difficulty_plan = [
        (level, DIFFICULTY_LEVELS[level], difficulty_guidance[level])
        for level in DIFFICULTY_LEVELS
        for _ in range(
            QUESTION_CATEGORIES["Role and skills"] // len(DIFFICULTY_LEVELS)
        )
    ]

    for _ in range(MAX_RETRIES + 1):
        remaining_plan = difficulty_plan[len(questions):]
        missing = len(remaining_plan)
        if missing <= 0:
            return [
                {
                    "question": question,
                    "category": "Role and skills",
                    "difficulty": {
                        "level": level,
                        "name": level_name
                    }
                }
                for question, (level, level_name, _) in zip(
                    questions,
                    difficulty_plan
                )
            ]

        prompt_plan = "\n".join(
            f"Question {index}: {level_name} - {guidance}"
            for index, (_, level_name, guidance) in enumerate(
                remaining_plan,
                start=1
            )
        )

        prompt = f"""
You are a technical interviewer assessing a candidate for a specific job.

Job role: {role}
Candidate skills: {", ".join(skills) if skills else role}
Relevant topics: {topic_text}

Generate exactly {missing} unique technical interview questions.
Each question must be directly related to the job role and candidate skills.
Use this difficulty plan in the exact order shown; do not group or skip levels:
{prompt_plan}

The questions must become progressively more challenging according to the plan.
Do not ask HR, project-history, or casual conversation questions.
Do not include answers, headings, or introductions.
Output only numbered questions, one per line. Every question must end with a question mark.

Already generated questions; do not repeat or rephrase them:
{chr(10).join(questions)}
"""
        generated_text = generate_from_ollama(
            prompt,
            min(1800, max(300, missing * 80))
        )
        for question in parse_questions(generated_text):
            normalized = normalize_question(question)
            if normalized and normalized not in seen:
                seen.add(normalized)
                questions.append(question)
                if len(questions) == len(difficulty_plan):
                    break

    if len(questions) != len(difficulty_plan):
        app.logger.warning(
            "Question model returned %d unique technical questions; expected %d.",
            len(questions),
            len(difficulty_plan)
        )
        return None
    return [
        {
            "question": question,
            "category": "Role and skills",
            "difficulty": {
                "level": level,
                "name": level_name
            }
        }
        for question, (level, level_name, _) in zip(questions, difficulty_plan)
    ]


def generate_mixed_questions(role, skills, topic_text):
    role_questions = generate_technical_questions(role, skills, topic_text)
    if role_questions is None:
        return None

    skill = random.choice(skills) if skills else role
    project_questions = [
        (
            1,
            f"Can you briefly describe a project where you used {skill} and your contribution?"
        ),
        (
            2,
            f"How did you apply {skill} to meet a specific requirement in a project?"
        ),
        (
            3,
            f"What challenge did you face using {skill} in a project, and how did you test your solution?"
        ),
        (
            4,
            f"What alternatives to {skill} did you consider for a project, and how did you weigh the trade-offs?"
        ),
        (
            5,
            f"If a project using {skill} had to scale significantly, what would you redesign first and why?"
        )
    ]
    hr_questions = [
        "What motivates you to do your best work?",
        "Can you describe a time you worked through a disagreement with a teammate?",
        "How do you respond to constructive feedback?",
        "How do you prioritize when several important tasks have competing deadlines?",
        "What strengths would you bring to this team, and how have you demonstrated them?"
    ]
    casual_questions = [
        "What hobby or activity do you enjoy outside of work?",
        "What is a place you have visited that you would recommend?",
        "What kind of music, books, or shows have you been enjoying lately?",
        "What is something fun you would like to try this year?",
        "What is a small thing that has made you smile recently?"
    ]

    nontechnical_questions = [
        {"question": question, "category": "HR and behavioral"}
        for question in hr_questions
    ]
    nontechnical_questions.extend(
        {
            "question": question,
            "category": "Project experience",
            "difficulty": {
                "level": level,
                "name": DIFFICULTY_LEVELS[level]
            }
        }
        for level, question in project_questions
    )
    nontechnical_questions.extend(
        {"question": question, "category": "Casual conversation"}
        for question in casual_questions
    )
    categorized_questions = role_questions.copy()
    random.shuffle(nontechnical_questions)
    for question in nontechnical_questions:
        categorized_questions.insert(
            random.randrange(len(categorized_questions) + 1),
            question
        )
    return categorized_questions


@app.route("/", methods=["GET"])
def home():
    return render_template("interview.html")


@app.route("/interview", methods=["GET"])
def interview():
    return render_template("interview.html")


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

    is_java_role = (
        "java" in role.lower()
        or any("java" in skill.lower() for skill in skills)
    )
    topics = skills + JAVA_TOPICS if is_java_role else skills
    topic_text = ", ".join(topics) if topics else role

    try:
        questions = generate_mixed_questions(role, skills, topic_text)

        if questions is None or len(questions) != TOTAL_QUESTIONS:
            return jsonify({
                "error": (
                    "The model did not return all 10 unique technical questions "
                    "in the expected format. Click Try Again to generate a new set."
                ),
                "required_count": TOTAL_QUESTIONS,
                "retry": True
            }), 502

        question_details = [
            {
                "question_number": index,
                "difficulty": questions[index - 1].get("difficulty"),
                "category": questions[index - 1]["category"]
            }
            for index in range(1, TOTAL_QUESTIONS + 1)
        ]

        return jsonify({
            "role": role,
            "skills": skills,
            "questions": [item["question"] for item in questions],
            "question_details": question_details,
            "total_questions": len(questions)
        })

    except requests.RequestException as error:
        app.logger.exception("Ollama request failed")
        return jsonify({
            "error": (
                "Could not connect to Ollama. Make sure Ollama is running "
                f"and model '{MODEL}' is available. Details: {error}"
            ),
            "retry": True
        }), 503


if __name__ == "__main__":
    app.run(debug=True, port=5000)
