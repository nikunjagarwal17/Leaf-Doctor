# Leaf Doctor

A web-based plant disease detection system that identifies diseases from leaf images and provides treatment recommendations through an AI-powered chat interface.

## Features

**Disease Detection**  
Upload 1-4 leaf images for CNN-based classification across 39 disease categories. Multi-image uploads use ensemble voting for improved accuracy.

**Agentic Advisor**  
An autonomous agent orchestrates the diagnosis pipeline—analyzing intent, planning tool execution, and self-correcting based on confidence thresholds. Supports follow-up questions about treatments, fertilizers, and prevention.

**RAG-Powered Responses**  
Retrieves relevant context from the disease knowledge base using semantic search (Sentence Transformers), then generates grounded advice via Groq LLM with HuggingFace as fallback.

**Embedding Caching**  
Embeddings are cached to disk with MD5 validation. The cache auto-regenerates only when the CSV knowledge base changes.

## Quick Start

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Configure environment
# Create .env with:
#   GROQ_API_KEY=your_groq_key
#   HUGGINGFACE_API_KEY=your_hf_key (optional fallback)

# Run
python app.py
```

Open `http://127.0.0.1:5000` and upload a leaf image.

## Docker

```bash
docker compose up --build
```

## Project Structure

```
app.py                  # Flask server, routes, LLM integration
agent.py                # Agentic orchestrator with self-correction
CNN.py                  # PyTorch CNN architecture (39 classes)
disease_info.csv        # Knowledge base (diseases, descriptions, treatments)
plant_disease_model_v1.pt  # Trained model weights
templates/              # HTML templates
static/                 # CSS, JS, uploads
```

## Tech Stack

- PyTorch CNN for image classification
- Sentence Transformers for semantic embeddings
- Groq API (llama-3.3-70b) as primary LLM
- HuggingFace Inference API (zephyr-7b) as fallback
- Flask for backend

## Authors

- Nikunj Agarwal – [LinkedIn](https://www.linkedin.com/in/nikunj-agarwal-d4rkm4773r/)
- Sahithi Kokkula – [LinkedIn](https://www.linkedin.com/in/sahithi-kokkula-iitbhu/)
