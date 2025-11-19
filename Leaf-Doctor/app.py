import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from huggingface_hub import InferenceClient
from huggingface_hub.utils import HfHubHTTPError
from werkzeug.utils import secure_filename

import CNN


load_dotenv()

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

disease_info = pd.read_csv(BASE_DIR / "disease_info.csv", encoding="cp1252")


def trim_text(value: str, max_chars: int = 220) -> str:
    clean = " ".join(str(value).split())
    if len(clean) <= max_chars:
        return clean
    head = clean[:max_chars].rsplit(" ", 1)[0]
    return f"{head}…"


DISEASE_CATALOG = [
    {
        "name": row.disease_name,
        "summary": trim_text(row.description),
        "steps": [
            step.strip()
            for step in str(getattr(row, "Possible_Steps", getattr(row, "Possible Steps", ""))).split("\n")
            if step.strip()
        ][:2],
    }
    for row in disease_info.itertuples(index=False)
]

BUILDERS = [
    {
        "name": "Nikunj Agarwal",
        "role": "Full-stack & ML Engineer",
        "linkedin": "https://www.linkedin.com/in/nikunj-agarwal-d4rkm4773r/",
    },
    {
        "name": "Sahithi Kokkula",
        "role": "ML Engineer & Product Designer",
        "linkedin": "https://www.linkedin.com/in/sahithi-kokkula-iitbhu/",
    },
]

model = CNN.CNN(39)
model.load_state_dict(torch.load(BASE_DIR / "plant_disease_model_v1.pt", map_location="cpu"))
model.eval()

HF_MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"
HF_API_KEY = os.environ.get("HUGGINGFACE_API_KEY")
HF_CLIENT = InferenceClient(token=HF_API_KEY) if HF_API_KEY else None


def predict_index(image_path: Path) -> int:
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image).view((-1, 3, 224, 224))
    output = model(input_data).detach().numpy()
    return int(np.argmax(output))


def build_prediction_payload(idx: int) -> dict:
    steps = [
        step.strip()
        for step in str(disease_info["Possible Steps"][idx]).split("\n")
        if step.strip()
    ]
    return {
        "disease": disease_info["disease_name"][idx],
        "description": disease_info["description"][idx],
        "steps": steps,
    }


def fetch_advice(prompt: str, disease: str | None = None) -> dict:
    if HF_CLIENT is None:
        return {
            "message": "Advisor service unavailable: Hugging Face API key missing on the server.",
            "source": "config",
        }

    messages = [
        {
            "role": "system",
            "content": "You are an agronomy expert. Provide concise, actionable crop-care guidance."
        },
        {
            "role": "user",
            "content": f"Disease focus: {disease or 'unknown disease'}\nQuestion: {prompt}\nConstraints: <=180 words, include organic + chemical options with dosages when relevant."
        }
    ]

    try:
        response = HF_CLIENT.chat_completion(
            messages=messages,
            model=HF_MODEL_ID,
            max_tokens=220,
            temperature=0.2,
            top_p=0.85,
        )
        generated = response.choices[0].message.content.strip()
        if not generated:
            raise ValueError("Empty advisor response.")
        return {"message": generated, "source": HF_MODEL_ID}
    except HfHubHTTPError as exc:
        status = getattr(exc.response, "status_code", "unknown")
        return {
            "message": f"Advisor service error ({status}). Please retry in a moment.",
            "source": HF_MODEL_ID,
        }
    except Exception as exc:  # pylint: disable=broad-except
        return {
            "message": f"Advisor service is currently unavailable ({exc}). Please retry later.",
            "source": "error",
        }


app = Flask(__name__)


@app.route("/")
def home_page():
    return render_template("home.html")


@app.route("/learn")
def learn_page():
    return render_template("learn.html", diseases=DISEASE_CATALOG, builders=BUILDERS, total=len(DISEASE_CATALOG))


@app.route("/predict", methods=["POST"])
def predict_route():
    image = request.files.get("image")
    if image is None or image.filename == "":
        return jsonify({"error": "Leaf image is required."}), 400

    filename = secure_filename(image.filename)
    save_path = UPLOAD_DIR / filename
    image.save(save_path)
    try:
        idx = predict_index(save_path)
        payload = build_prediction_payload(idx)
        payload["diagnosis_id"] = idx
        return jsonify(payload)
    finally:
        try:
            if save_path.exists():
                save_path.unlink()
        except OSError:
            pass


@app.route("/advisor", methods=["POST"])
def advisor_route():
    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt", "").strip()
    disease = data.get("disease")

    if not prompt:
        return jsonify({"error": "Please provide a question for the advisor."}), 400

    advice = fetch_advice(prompt, disease)
    return jsonify(advice)


if __name__ == "__main__":
    app.run(debug=True)
