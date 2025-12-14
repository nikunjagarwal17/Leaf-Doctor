import sys

# CI-aware model check: skip model load in CI if missing
MODEL_PATH = Path(__file__).parent / "plant_disease_model_v1.pt"
IS_CI = os.environ.get("GITHUB_ACTIONS") == "true"

if not MODEL_PATH.exists():
    if IS_CI:
        print("CI detected: Skipping model load, running minimal app for health check.")
        from flask import Flask
        app = Flask(__name__)

        @app.route("/")
        def health():
            return "CI health check OK", 200

        if __name__ == "__main__":
            app.run(debug=True, port=5000)
        sys.exit(0)
    else:
        print(f"ERROR: Model file not found at {MODEL_PATH}. Please upload the model to start the app.")
        sys.exit(1)
import hashlib
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from huggingface_hub import InferenceClient
from werkzeug.utils import secure_filename

import CNN
from agent import AgentOrchestrator, create_agent_tools


try:  
    from sentence_transformers import SentenceTransformer
except Exception:  
    SentenceTransformer = None

try:
    from groq import Groq
except Exception:
    Groq = None

try:
    import faiss
except Exception:
    faiss = None


load_dotenv()


AGENT: AgentOrchestrator = None

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MAX_UPLOADS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


# Embedding cache configuration
EMBEDDINGS_CACHE_FILE = BASE_DIR / "embeddings_cache.pkl"
FAISS_INDEX_FILE = BASE_DIR / "faiss_index.bin"
CSV_FILE = BASE_DIR / "disease_info.csv"

# Global FAISS index
FAISS_INDEX = None
DOCUMENT_STORE: list[dict] = []  # Stores document metadata (name, text)


def get_csv_hash() -> str:
    """Calculate MD5 hash of the CSV file to detect changes."""
    with open(CSV_FILE, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def load_embedder():
    """Load sentence transformer if available."""
    if SentenceTransformer is None:
        return None
    try:
        return SentenceTransformer(EMBED_MODEL_NAME)
    except Exception:
        return None


def load_cached_faiss_index() -> tuple[list[dict], str] | None:
    """Load FAISS index and document store from cache if CSV hasn't changed."""
    
    if not EMBEDDINGS_CACHE_FILE.exists() or not FAISS_INDEX_FILE.exists():
        return None
    
    if faiss is None:
        print("FAISS not available, falling back to in-memory search")
        return None
    
    try:
        # Load document store with metadata
        with open(EMBEDDINGS_CACHE_FILE, "rb") as f:
            cache = pickle.load(f)
        
        # Validate cache structure and CSV hash
        if isinstance(cache, dict) and cache.get("csv_hash") == get_csv_hash():
            # Load FAISS index
            FAISS_INDEX = faiss.read_index(str(FAISS_INDEX_FILE))
            print(f"✅ Loaded FAISS index from cache ({FAISS_INDEX.ntotal} vectors)")
            return cache["documents"], cache["csv_hash"]
    except Exception as e:
        print(f"FAISS cache load failed: {e}")
    return None


def save_faiss_index(documents: list[dict], embeddings_matrix: np.ndarray, csv_hash: str) -> None:
    """Save FAISS index and document store to cache."""
    
    if faiss is None:
        return
    
    try:
        # Save FAISS index
        faiss.write_index(FAISS_INDEX, str(FAISS_INDEX_FILE))
        
        # Save document metadata (without embeddings - they're in FAISS)
        cache = {
            "csv_hash": csv_hash,
            "documents": documents
        }
        with open(EMBEDDINGS_CACHE_FILE, "wb") as f:
            pickle.dump(cache, f)
        
        print(f"✅ Saved FAISS index ({FAISS_INDEX.ntotal} vectors) and document cache")
    except Exception as e:
        print(f"FAISS cache save failed: {e}")


def build_faiss_index(embedder) -> list[dict]:
    """Build FAISS index from disease information CSV."""
    
    if embedder is None:
        return []
    
    # Try loading from cache first
    cached = load_cached_faiss_index()
    if cached:
        DOCUMENT_STORE = cached[0]
        return DOCUMENT_STORE
    
    # Generate fresh embeddings and build FAISS index
    print("🔨 Building FAISS index (CSV changed or no cache)...")
    csv_hash = get_csv_hash()
    
    documents: list[dict] = []
    embeddings_list: list[np.ndarray] = []
    
    for row in disease_info.itertuples(index=False):
        steps = [
            step.strip()
            for step in str(getattr(row, "Possible_Steps", getattr(row, "Possible Steps", ""))).split("\n")
            if step.strip()
        ]
        text = (
            f"{row.disease_name}. Description: {row.description}. "
            f"Best practices: {'; '.join(steps) if steps else 'No steps listed.'}"
        )
        vector = embedder.encode(text, normalize_embeddings=True)
        
        documents.append({"name": row.disease_name, "text": text})
        embeddings_list.append(vector)
    
    # Convert to numpy matrix
    embeddings_matrix = np.array(embeddings_list, dtype=np.float32)
    embedding_dim = embeddings_matrix.shape[1]
    
    if faiss is not None:
        # Create FAISS index - using IndexFlatIP for inner product (cosine similarity with normalized vectors)
        FAISS_INDEX = faiss.IndexFlatIP(embedding_dim)
        FAISS_INDEX.add(embeddings_matrix)
        print(f"✅ Created FAISS index with {FAISS_INDEX.ntotal} vectors (dim={embedding_dim})")
        
        # Save to cache
        save_faiss_index(documents, embeddings_matrix, csv_hash)
    else:
        # Fallback: store embeddings in documents for in-memory search
        print("⚠️ FAISS not available, using in-memory fallback")
        for i, doc in enumerate(documents):
            doc["embedding"] = embeddings_list[i]
    
    DOCUMENT_STORE = documents
    return documents

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
model.to(DEVICE)
model.eval()


GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL_ID = os.environ.get("GROQ_MODEL_ID", "llama-3.3-70b-versatile")
GROQ_CLIENT = Groq(api_key=GROQ_API_KEY) if (Groq and GROQ_API_KEY) else None

HF_MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"
HF_API_KEY = os.environ.get("HUGGINGFACE_API_KEY")
HF_CLIENT = InferenceClient(token=HF_API_KEY) if HF_API_KEY else None

EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDER = None

# Lazy-load models and build FAISS index
if EMBEDDER is None:
    EMBEDDER = load_embedder()
    build_faiss_index(EMBEDDER)


def retrieve_context(query: str, disease: str | None = None, top_k: int = 3) -> list[dict]:
    """Return top-k context snippets using FAISS vector search."""
    
    if EMBEDDER is None or not DOCUMENT_STORE:
        return []

    prompt = f"{disease or 'unknown disease'} - {query}".strip()
    query_vec = EMBEDDER.encode(prompt, normalize_embeddings=True).astype(np.float32)
    
    # Reshape for FAISS (needs 2D array)
    query_vec = query_vec.reshape(1, -1)
    
    if FAISS_INDEX is not None and faiss is not None:
        # Use FAISS for efficient similarity search
        scores, indices = FAISS_INDEX.search(query_vec, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(DOCUMENT_STORE) and idx >= 0:
                doc = DOCUMENT_STORE[idx]
                results.append({
                    "name": doc["name"],
                    "text": doc["text"],
                    "score": float(scores[0][i])
                })
        return results
    else:
        # Fallback to in-memory search if FAISS not available
        scored = []
        for item in DOCUMENT_STORE:
            if "embedding" in item:
                score = float(np.dot(query_vec.flatten(), item["embedding"]))
                scored.append({"name": item["name"], "text": item["text"], "score": score})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]


def sanitize_context(context: list[dict]) -> list[dict]:
    """Remove non-serializable fields such as the raw embedding vectors."""
    clean = []
    for item in context or []:
        clean.append(
            {
                "name": item.get("name"),
                "text": item.get("text"),
                "score": float(item.get("score", 0.0)),
            }
        )
    return clean


def format_context(context: list[dict]) -> str:
    """Create a compact context block for the LLM."""
    if not context:
        return "No embedded context available."
    return "\n".join([f"- {c['text']} (relevance: {c['score']:.2f})" for c in context])


def _infer_logits(image_path: Path) -> np.ndarray: #"""Run the CNN on a single image and return logits."""
    
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(input_data)
    return output.squeeze(0).detach().cpu().numpy()


def aggregate_predictions(image_paths: list[Path]) -> dict: # """Ensemble logits across multiple images to boost confidence."""
    
    logits_stack = []
    per_image = []

    for path in image_paths:
        logits = _infer_logits(path)
        probs = F.softmax(torch.tensor(logits), dim=-1).numpy()
        pred_idx = int(np.argmax(logits))
        logits_stack.append(logits)
        per_image.append(
            {
                "file": path.name,
                "predicted_idx": pred_idx,
                "disease": disease_info["disease_name"][pred_idx],
                "confidence": float(probs[pred_idx]),
            }
        )

    mean_logits = np.mean(logits_stack, axis=0)
    mean_probs = F.softmax(torch.tensor(mean_logits), dim=-1).numpy()
    top_idx = int(np.argmax(mean_probs))
    top_confidence = float(mean_probs[top_idx])
    ranked = [
        {"disease": disease_info["disease_name"][i], "probability": float(prob)}
        for i, prob in enumerate(mean_probs)
    ]
    ranked.sort(key=lambda x: x["probability"], reverse=True)

    return {
        "top_idx": top_idx,
        "confidence": top_confidence,
        "per_image": per_image,
        "ranked": ranked[:3],
    }


def build_prediction_payload(ensemble: dict) -> dict:
    idx = ensemble["top_idx"]
    steps = [
        step.strip()
        for step in str(disease_info["Possible Steps"][idx]).split("\n")
        if step.strip()
    ]
    return {
        "disease": disease_info["disease_name"][idx],
        "description": disease_info["description"][idx],
        "steps": steps,
        "diagnosis_id": idx,
        "confidence": round(ensemble["confidence"], 4),
        "alternatives": ensemble["ranked"],
        "per_image": ensemble["per_image"],
    }


def fetch_advice(prompt: str, disease: str | None = None) -> dict:
    """Route advisor calls: Groq (Priority 1) -> HuggingFace (Priority 2)."""
    context = retrieve_context(prompt, disease=disease)
    clean_context = sanitize_context(context)
    context_block = format_context(context)

    # Build a unified instruction for LLM
    system_text = (
        "You are an agronomy expert that answers with concise, actionable guidance. "
        "Prioritize organic approaches, follow with chemical controls including active ingredients and dosages. "
        "Keep answers under 180 words. Structure in short paragraphs or bullets."
    )
    user_text = (
        f"Disease focus: {disease or 'unknown disease'}\n"
        f"Question: {prompt}\n"
        f"Context snippets:\n{context_block}\n"
        "Add a short execution checklist at the end."
    )

    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]

    # Priority 1: Try Groq API
    if GROQ_CLIENT is not None:
        try:
            response = GROQ_CLIENT.chat.completions.create(
                messages=messages,
                model=GROQ_MODEL_ID,
                max_tokens=240,
                temperature=0.25,
                top_p=0.9,
            )
            generated = response.choices[0].message.content.strip()
            if generated:
                return {
                    "message": generated,
                    "source": f"Groq ({GROQ_MODEL_ID})",
                    "reasoning": context_block,
                    "context": clean_context,
                }
        except Exception as groq_exc:
            # Fall through to HuggingFace
            pass

    # Priority 2: Try HuggingFace API
    if HF_CLIENT is not None:
        try:
            response = HF_CLIENT.chat_completion(
                messages=messages,
                model=HF_MODEL_ID,
                max_tokens=240,
                temperature=0.25,
                top_p=0.9,
            )
            generated = response.choices[0].message.content.strip()
            if generated:
                return {
                    "message": generated,
                    "source": f"HuggingFace ({HF_MODEL_ID})",
                    "reasoning": context_block,
                    "context": clean_context,
                }
        except Exception as hf_exc:
            return {
                "message": f"LLM service error: {hf_exc}. Please retry later.",
                "source": "error",
                "reasoning": context_block,
                "context": clean_context,
            }

    # No LLM available
    return {
        "message": "No LLM service configured. Please set GROQ_API_KEY or HUGGINGFACE_API_KEY.",
        "source": "none",
        "reasoning": context_block,
        "context": clean_context,
    }



def agent_cnn_predict(image_paths: list[Path]) -> dict:
    """
    Tool wrapper for CNN prediction - used by the agent.
    Returns standardized output for the agent to process.
    """
    if not image_paths:
        return {"error": "No images provided", "confidence": 0}
    
    ensemble = aggregate_predictions(image_paths)
    result = build_prediction_payload(ensemble)
    return result


def agent_retrieve_context(query: str, disease: str | None = None) -> dict:
    """
    Tool wrapper for knowledge retrieval - used by the agent.
    Returns context with metadata for agent reflection.
    """
    context = retrieve_context(query, disease=disease)
    clean = sanitize_context(context)
    formatted = format_context(context)
    
    return {
        "context": clean,
        "formatted": formatted,
        "query_used": f"{disease or 'unknown'} - {query}",
        "num_results": len(clean)
    }


def agent_generate_advice(prompt: str, disease: str | None = None, 
                          retrieved_context: list[dict] = None,
                          formatted_context: str = None) -> dict:
    """
    Tool wrapper for LLM advice generation - used by the agent.
    Uses Groq (Priority 1) -> HuggingFace (Priority 2).
    Can use pre-retrieved context for efficiency.
    """
    # Use pre-retrieved context if available, otherwise retrieve
    if retrieved_context:
        clean_context = retrieved_context
        context_block = formatted_context if formatted_context else format_context(retrieved_context)
    else:
        context = retrieve_context(prompt, disease=disease)
        clean_context = sanitize_context(context)
        context_block = format_context(context)

    # Build instruction for LLM
    system_text = (
        "You are an agronomy expert that answers with concise, actionable guidance. "
        "Prioritize organic approaches, follow with chemical controls including active ingredients and dosages. "
        "Keep answers under 180 words. Structure in short paragraphs or bullets."
    )
    user_text = (
        f"Disease focus: {disease or 'unknown disease'}\n"
        f"Question: {prompt}\n"
        f"Context snippets:\n{context_block}\n"
        "Add a short execution checklist at the end."
    )

    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]

    # Priority 1: Try Groq API
    if GROQ_CLIENT is not None:
        try:
            response = GROQ_CLIENT.chat.completions.create(
                messages=messages,
                model=GROQ_MODEL_ID,
                max_tokens=240,
                temperature=0.25,
                top_p=0.9,
            )
            generated = response.choices[0].message.content.strip()
            if generated:
                return {
                    "message": generated,
                    "source": f"Groq ({GROQ_MODEL_ID})",
                    "reasoning": context_block,
                    "context": clean_context,
                }
        except Exception:
            # Fall through to HuggingFace
            pass

    # Priority 2: Try HuggingFace API
    if HF_CLIENT is not None:
        try:
            response = HF_CLIENT.chat_completion(
                messages=messages,
                model=HF_MODEL_ID,
                max_tokens=240,
                temperature=0.25,
                top_p=0.9,
            )
            generated = response.choices[0].message.content.strip()
            if generated:
                return {
                    "message": generated,
                    "source": f"HuggingFace ({HF_MODEL_ID})",
                    "reasoning": context_block,
                    "context": clean_context,
                }
        except Exception as hf_exc:
            return {
                "message": f"LLM service error: {hf_exc}. Please retry later.",
                "source": "error",
                "reasoning": context_block,
                "context": clean_context,
            }

    # No LLM available
    return {
        "message": "No LLM service configured. Please set GROQ_API_KEY or HUGGINGFACE_API_KEY.",
        "source": "none",
        "reasoning": context_block,
        "context": clean_context,
    }


def initialize_agent() -> AgentOrchestrator:
    """Initialize the agentic orchestrator with all available tools."""
    tools = create_agent_tools(
        cnn_predict_fn=agent_cnn_predict,
        retrieve_fn=agent_retrieve_context,
        generate_advice_fn=agent_generate_advice
    )
    return AgentOrchestrator(tools)


# Initialize the global agent
AGENT = initialize_agent()


app = Flask(__name__)


@app.route("/")
def home_page():
    return render_template("home.html")


@app.route("/learn")
def learn_page():
    return render_template("learn.html", diseases=DISEASE_CATALOG, builders=BUILDERS, total=len(DISEASE_CATALOG))


@app.route("/predict", methods=["POST"])
def predict_route():
    """
    Agentic prediction endpoint.
    The agent autonomously plans and executes:
    1. Image classification
    2. Confidence checking with self-correction
    3. Knowledge retrieval for context
    """
    images = request.files.getlist("images") or []
    images = [img for img in images if img and img.filename]
    if not images:
        single = request.files.get("image")
        if single and single.filename:
            images = [single]
    if not images:
        return jsonify({"error": "At least one leaf image is required."}), 400
    if len(images) > MAX_UPLOADS:
        return jsonify({"error": f"Please upload up to {MAX_UPLOADS} images for one diagnosis."}), 400

    note = (request.form.get("note") or "").strip()
    use_agent = request.form.get("agentic", "true").lower() == "true"
    
    saved_paths: list[Path] = []
    for image in images:
        filename = secure_filename(image.filename)
        save_path = UPLOAD_DIR / filename
        image.save(save_path)
        saved_paths.append(save_path)

    try:
        if use_agent and AGENT:
            # Use the agentic system for autonomous processing
            user_input = {
                "image_paths": saved_paths,
                "note": note,
                "prompt": note if note else ""
            }
            
            # Agent analyzes intent and plans execution
            intent = AGENT.analyze_intent(user_input)
            plan = AGENT.plan_execution(intent, user_input)
            
            # Execute with self-correction
            result = AGENT.execute_plan(plan)
            
            # Extract diagnosis from agent result
            payload = result.get("diagnosis", {})
            payload["agentic"] = True
            payload["reasoning_chain"] = result.get("reasoning_chain", [])
            payload["agent_goal"] = result.get("goal", "")
            
            # Add note context if present
            if note:
                payload["note"] = note
                if result.get("context"):
                    payload["note_context"] = result["context"]
            
            payload["image_count"] = len(saved_paths)
            
            # Add any quality check results
            if result.get("prediction_check"):
                payload["quality_check"] = result["prediction_check"]
            
            return jsonify(payload)
        else:
            # Fallback to non-agentic processing
            ensemble = aggregate_predictions(saved_paths)
            payload = build_prediction_payload(ensemble)
            if note:
                note_context = sanitize_context(retrieve_context(note))
                payload["note"] = note
                payload["note_context"] = note_context
            payload["image_count"] = len(saved_paths)
            payload["agentic"] = False
            return jsonify(payload)
    finally:
        for path in saved_paths:
            try:
                if path.exists():
                    path.unlink()
            except OSError:
                pass


@app.route("/advisor", methods=["POST"])
def advisor_route():
    """
    Agentic advisor endpoint.
    The agent autonomously plans and executes:
    1. Knowledge retrieval with quality checking
    2. LLM advice generation
    3. Self-correction if response quality is low
    """
    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt", "").strip()
    disease = data.get("disease")
    use_agent = data.get("agentic", True)

    if not prompt:
        return jsonify({"error": "Please provide a question for the advisor."}), 400

    if use_agent and AGENT:
        # Use the agentic system
        user_input = {
            "prompt": prompt,
            "disease": disease
        }
        
        # Agent analyzes intent and plans execution
        intent = AGENT.analyze_intent(user_input)
        plan = AGENT.plan_execution(intent, user_input)
        
        # Execute with self-correction
        result = AGENT.execute_plan(plan)
        
        # Extract advice from agent result
        advice = result.get("advice", {})
        advice["agentic"] = True
        advice["reasoning_chain"] = result.get("reasoning_chain", [])
        advice["agent_goal"] = result.get("goal", "")
        advice["agent_success"] = result.get("success", False)
        
        # Add quality checks
        if result.get("retrieval_check"):
            advice["retrieval_quality"] = result["retrieval_check"]
        if result.get("response_check"):
            advice["response_quality"] = result["response_check"]
        
        return jsonify(advice)
    else:
        # Fallback to non-agentic processing
        advice = fetch_advice(prompt, disease)
        advice["agentic"] = False
        return jsonify(advice)


if __name__ == "__main__":
    print("🌿 Leaf Doctor")
    print("=" * 40)
    print(f"Base directory: {BASE_DIR}")
    print(f"Device: {DEVICE}")
    print(f"Groq API: {'Configured' if GROQ_CLIENT else 'Not configured'}")
    print(f"HuggingFace API: {'Configured' if HF_CLIENT else 'Not configured'}")
    print(f"FAISS Index: {'Loaded (' + str(FAISS_INDEX.ntotal) + ' vectors)' if FAISS_INDEX else 'Not available'}")
    print(f"Document Store: {len(DOCUMENT_STORE)} diseases")
    print("=" * 40)
    app.run(debug=True, port=5000)
