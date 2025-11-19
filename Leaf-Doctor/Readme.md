# Leaf Doctor
Leaf Doctor is a friendly web assistant that spots plant diseases from a leaf photo and walks farmers through the next steps. Everything runs through a single chatbot-style interface, so you can diagnose, read curated tips, and ask questions without jumping between tabs.

## What it can do
- **Leaf upload + CNN diagnosis** – drop in a photo and the onboard PyTorch model predicts one of 39 diseases with a short overview and prevention checklist.
- **Advisor chat** – send follow-up questions to a Hugging Face Zephyr model (via your API key) for organic/chemical recommendations, dosages, and cultural practices.
- **Supplement shortcut** – tap “Get supplement suggestion” to auto-ask the advisor for fertilizers, fungicides, or pesticides.
- **Learn More page** – browse every supported disease and meet the builders (Nikunj Agarwal & Sahithi Kokkula) with quick LinkedIn links.
- **Clean storage & Docker** – uploads are deleted after each request, and the repo ships with a Dockerfile + docker-compose for easy container runs.

## Quick start
1. **Install deps**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Add your secrets**  
   Create a `.env` file with `HUGGINGFACE_API_KEY=<your token>`. The key powers the advisor and supplement features.
3. **Run the app**
   ```bash
   python app.py
   ```
   Visit `http://127.0.0.1:5000`, upload a leaf image, and start chatting.

## Docker option
```bash
docker compose up --build
```
This uses the gunicorn server and mounts `static/uploads` so the app can keep saving temporary files while running in a container.

## Files to know
- `app.py` – Flask server, CNN inference, advisor endpoints.
- `templates/` + `static/` – chatbot UI, Learn page, styles, and client JS.
- `disease_info.csv` – knowledge base used for both predictions and the Learn page.
- `plant_disease_model_v1.pt` – PyTorch weights (keep in the project root).

That’s it—keep the `.env` file private, swap in updated model weights when you retrain, and you’re good to deploy. Happy diagnosing! 🌱
