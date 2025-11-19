# Plant Disease Detection

This repository contains **Leaf Doctor**, a plant disease detection system built with PyTorch and Flask.

- A CNN model trained on leaf images to classify 39 plant diseases.
- A web app in `Leaf-Doctor/` that lets you upload a leaf photo, get a diagnosis, and read simple prevention/treatment tips.
- A "Learn" page listing all supported diseases in an easy-to-scan format.

## Quick start (web app)

1. Go to the Flask app folder:
	```bash
	cd "Leaf-Doctor"
	```
2. Create a virtual environment and install dependencies:
	```bash
	python -m venv .venv
	.venv\Scripts\activate  # on Windows
	pip install -r requirements.txt
	```
3. Add your Hugging Face key in `.env`:
	```env
	HUGGINGFACE_API_KEY=your_hf_token_here
	```
4. Run the app:
	```bash
	python app.py
	```
5. Open `http://127.0.0.1:5000` in your browser to use the chatbot and Learn page.

## Training notebook

The training pipeline for the CNN model lives in:

- `Model/Plant Disease Detection Code.ipynb`

It shows how the dataset is loaded, the CNN is defined, and how the final weights file is saved.

For more details about the web app itself, check `Leaf-Doctor/Readme.md`.
