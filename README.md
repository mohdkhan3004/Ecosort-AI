# EcoSort AI — Smart Waste Classification Assistant

> An academic AI + ML project aligned with **UN SDG 12: Responsible Consumption and Production**

---

## Project Overview

EcoSort AI is a small, functional web application that helps users identify common household and office waste items and understand how to dispose of them responsibly. It combines a real trained machine-learning model with a Google Gemini AI advisor to deliver both a classification result and practical, personalised disposal guidance.

**Problem statement:** Improper waste disposal — mixing recyclables with general waste, contaminating compost bins, or sending recyclables to landfill — contributes to unnecessary resource waste, pollution, and increased carbon emissions. Most people lack quick, accessible guidance for everyday waste items.

---

## Objectives

- Demonstrate a working ML text-classification pipeline (TF-IDF + Logistic Regression).
- Provide AI-generated disposal advice through the Gemini API.
- Support three simplified waste categories: **Organic**, **Recyclable**, and **General Waste**.
- Align with UN SDG 12 by promoting responsible consumption habits.
- Serve as a clear, understandable academic project for college-level demonstration.

---

## Main Features

| Feature | Description |
|---|---|
| ML Classifier | TF-IDF vectoriser + Logistic Regression trained on ~80 labelled waste examples |
| Class Probabilities | Confidence scores displayed for each category |
| Gemini AI Advisor | Disposal explanation, practical steps, eco-tip, and safety note |
| Fallback Guide | Static disposal guide shown when Gemini API is unavailable |
| Sample Items | Five clickable examples to try instantly |
| SDG 12 Alignment | Project description and disclaimer on every page |

---

## Technology Stack

| Layer | Technology |
|---|---|
| Web interface | Streamlit |
| ML | scikit-learn (TF-IDF + Logistic Regression) |
| Data | pandas, CSV |
| AI advisor | Google Gemini API (`google-generativeai`) |
| Environment | python-dotenv |
| Language | Python 3.10+ |

---

## Project Folder Structure

```
EcoSort-AI/
├── app.py              ← Streamlit application (UI + AI advisor)
├── train_model.py      ← ML training and evaluation script
├── data/
│   └── waste_samples.csv  ← Labelled dataset (~80 examples)
├── model.pkl           ← Saved trained model (generated after training)
├── requirements.txt    ← Python dependencies
├── .env.example        ← API key template (safe to commit)
├── .gitignore          ← Excludes .env, model.pkl, cache
├── README.md           ← This file
├── PROJECT_REPORT.md   ← Academic mini-project report
├── CONCEPT_NOTE.md     ← Project concept note
├── LEAN_CANVAS.md      ← Lean Canvas
├── PRESENTATION.md     ← Slide-by-slide presentation content
└── TESTING.md          ← Test checklist and results
```

---

## Prerequisites

- Python 3.10 or later ([python.org](https://www.python.org/downloads/))
- `pip` (comes with Python)
- A Google Gemini API key (optional — the ML classifier works without it)
  - Free tier available at <https://aistudio.google.com/app/apikey>

---

## Installation (Windows)

### 1. Open a terminal and navigate to the project folder

```powershell
cd path\to\EcoSort-AI
```

### 2. Create and activate a virtual environment

```powershell
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` at the start of the prompt.

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

---

## Dataset and ML Training

The CSV dataset is already included at `data/waste_samples.csv`. You must train the model once before running the app.

```powershell
python train_model.py
```

Expected output:

```
 EcoSort AI — Model Training
-----------------------------------
Step 1: Loading dataset ...
  Loaded 80 samples across 3 categories.
  Category distribution:
  ...
Step 2: Training and evaluating model ...
  Training samples : 64
  Test samples     : 16

==================================================
  Test Accuracy: XX.X%
==================================================

  Classification Report:
  ...

Step 3: Saving model ...
  Model saved to: ...\model.pkl

 Training complete. Run the app with:
  streamlit run app.py
```

The trained model is saved as `model.pkl` in the project folder.

---

## Gemini API Key Setup

1. Copy the example environment file:

```powershell
copy .env.example .env
```

2. Open `.env` in any text editor and replace the placeholder:

```
GEMINI_API_KEY=your_actual_key_here
```

3. Save the file. **Never commit `.env` to version control.**

> The app works without a key — it shows a static fallback disposal guide instead of AI advice.

---

## Running the Application

```powershell
streamlit run app.py
```

The app opens automatically in your browser at `http://localhost:8501`.

---

## Example Inputs and Expected Categories

| Input | Expected Category |
|---|---|
| banana peel | Organic |
| coffee grounds | Organic |
| cardboard box | Recyclable |
| glass bottle | Recyclable |
| aluminum can | Recyclable |
| used tissue | General Waste |
| broken ceramic mug | General Waste |
| dirty diaper | General Waste |
| vegetable scraps | Organic |
| clean plastic bottle | Recyclable |

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `Model not found` error in app | Run `python train_model.py` first |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` inside the venv |
| Streamlit not found | Make sure the virtual environment is activated |
| Gemini advice not showing | Check that `.env` contains a valid `GEMINI_API_KEY` |
| Low model accuracy | The dataset is small (~80 items); this is expected for academic purposes |
| App shows port in use | Run `streamlit run app.py --server.port 8502` |

---

## Limitations

- The dataset contains ~80 items — this is intentionally small for academic demonstration.
- Disposal categories are simplified educational labels, not universal recycling rules.
- Local waste-management rules vary significantly by municipality and country.
- The ML model may misclassify unusual or ambiguous inputs not seen during training.
- Gemini AI advice is general and not tailored to any specific location.

---

## Future Improvements

- Expand the dataset to several hundred items for better accuracy.
- Add image-based waste recognition (computer vision).
- Integrate location-aware disposal rules via a local-authority API.
- Support multiple languages.
- Add a feedback loop so users can flag incorrect classifications.

---

## Academic-Use Disclaimer

This project was built solely for academic and educational purposes. It is not a certified waste-management tool. Always follow the waste-disposal guidelines provided by your local municipality. Predictions are informational only and should not replace professional environmental guidance.

---

*EcoSort AI · Academic Project · SDG 12 Responsible Consumption and Production*
