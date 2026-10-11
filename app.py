"""
app.py — EcoSort AI: Smart Waste Classification Assistant
Streamlit web interface that combines:
  • ML classifier (TF-IDF + Logistic Regression) for waste categorisation
  • Google Gemini AI for disposal advice

Run after training the model:
    streamlit run app.py
"""

import os
import pickle
import textwrap

import streamlit as st
from dotenv import load_dotenv

# Load .env for local development (no-op in production if already set)
load_dotenv()

# ─── Constants ───────────────────────────────────────────────────────────────

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

# ─── Auto-train if model is missing (e.g. on first cloud deployment) ─────────

def _auto_train():
    """Train and save model.pkl if it doesn't exist yet (runs on cloud startup)."""
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    data_path = os.path.join(os.path.dirname(__file__), "data", "waste_samples.csv")
    df = pd.read_csv(data_path)
    df = df.dropna(subset=["item", "category"])
    valid = {"Organic", "Recyclable", "General Waste"}
    df = df[df["category"].isin(valid)]

    X = df["item"].to_numpy(dtype=str)
    y = df["category"].to_numpy(dtype=str)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])
    pipeline.fit(X, y)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)

CATEGORY_EMOJI = {
    "Organic": "🌱",
    "Recyclable": "♻️",
    "General Waste": "🗑️",
}

CATEGORY_COLOR = {
    "Organic": "#2d6a2d",
    "Recyclable": "#1a5276",
    "General Waste": "#7d3c0a",
}

SAMPLE_ITEMS = [
    "banana peel",
    "glass bottle",
    "used tissue",
    "cardboard box",
    "coffee grounds",
]

# Fallback disposal guides shown when Gemini API is unavailable
FALLBACK_GUIDE = {
    "Organic": textwrap.dedent("""\
        **Category: Organic Waste**
        - Place in a compost bin or green-waste bin if your locality provides one.
        - Do not mix with recyclables — organic matter can contaminate recycling streams.
        - Home composting is an excellent option for fruit and vegetable scraps.
        - Tip: Avoid composting meat or dairy at home as it can attract pests.

        *(Note: This is a static fallback guide. AI-generated advice is unavailable right now.)*"""),
    "Recyclable": textwrap.dedent("""\
        **Category: Recyclable Waste**
        - Rinse containers to remove food residue before placing them in the recycling bin.
        - Check local guidelines — not all plastics are accepted everywhere.
        - Flatten cardboard boxes to save bin space.
        - Glass, aluminium, and clean paper are widely accepted recyclables.

        *(Note: This is a static fallback guide. AI-generated advice is unavailable right now.)*"""),
    "General Waste": textwrap.dedent("""\
        **Category: General Waste**
        - Place in the general (landfill) bin.
        - Avoid putting hazardous materials (batteries, chemicals) in general waste — check local drop-off points.
        - Try to reduce single-use items to minimise general waste.
        - Some items in this category may be reducible through better product choices.

        *(Note: This is a static fallback guide. AI-generated advice is unavailable right now.)*"""),
}

# ─── Model loading ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading ML model …")
def load_model():
    """Load the trained scikit-learn pipeline from disk.
    If model.pkl is missing (e.g. fresh cloud deployment), train it first.
    """
    if not os.path.exists(MODEL_PATH):
        _auto_train()
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


# ─── ML classification ────────────────────────────────────────────────────────

def classify_waste(model, user_input: str) -> dict:
    """
    Run the ML pipeline on user_input.
    Returns a dict with keys: category, confidence, probabilities.
    """
    prediction = model.predict([user_input])[0]
    proba = model.predict_proba([user_input])[0]
    classes = model.classes_
    prob_dict = {cls: float(p) for cls, p in zip(classes, proba)}
    confidence = prob_dict[prediction]
    return {
        "category": prediction,
        "confidence": confidence,
        "probabilities": prob_dict,
    }


# ─── Gemini AI advisor ────────────────────────────────────────────────────────

def get_gemini_advice(waste_description: str, ml_category: str) -> str | None:
    """
    Call the Gemini API for disposal advice.
    Returns the AI response text, or None on any failure.
    API key is read from the GEMINI_API_KEY environment variable only.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None  # Caller will show fallback

    try:
        import google.generativeai as genai  # noqa: PLC0415
    except ImportError:
        return None

    try:
        genai.configure(api_key=api_key)

        prompt = textwrap.dedent(f"""\
            You are an environmental education assistant helping users understand waste disposal.

            The user described this waste item: "{waste_description}"
            The ML classifier predicted the category: "{ml_category}"

            Please provide a helpful, concise response with these four parts:
            1. A one-sentence explanation of what "{ml_category}" waste means.
            2. Two or three practical steps for disposing of this item responsibly.
            3. One environmentally responsible suggestion (reuse, reduce, or better alternative).
            4. A brief safety or contamination warning if relevant to this item.

            Keep your response under 200 words. Do not contradict or change the ML-predicted
            category of "{ml_category}". Remind the user that local disposal rules may vary.
        """)

        # Try the newest stable model; fall back gracefully if unavailable
        try:
            model_name = "gemini-1.5-flash"
            gem_model = genai.GenerativeModel(model_name)
            response = gem_model.generate_content(prompt)
        except Exception:
            model_name = "gemini-pro"
            gem_model = genai.GenerativeModel(model_name)
            response = gem_model.generate_content(prompt)

        return response.text.strip()

    except Exception as exc:
        # Log to console (not UI) for debugging; never expose key details
        print(f"[EcoSort AI] Gemini API error: {type(exc).__name__}: {exc}")
        return None


# ─── Streamlit UI ─────────────────────────────────────────────────────────────

def render_category_badge(category: str, confidence: float):
    emoji = CATEGORY_EMOJI.get(category, "❓")
    color = CATEGORY_COLOR.get(category, "#333")
    pct = f"{confidence * 100:.1f}%"
    st.markdown(
        f"""
        <div style="
            background:{color}18;
            border-left: 5px solid {color};
            padding: 12px 16px;
            border-radius: 6px;
            margin-bottom: 8px;
        ">
            <span style="font-size:1.5rem;">{emoji}</span>
            <strong style="font-size:1.15rem; color:{color}; margin-left:8px;">{category}</strong>
            <span style="font-size:0.85rem; color:#555; margin-left:12px;">
                Model confidence: <strong>{pct}</strong>
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_probabilities(probabilities: dict):
    """Show a compact probability bar for each category."""
    st.markdown("**Class probabilities:**")
    for cat, prob in sorted(probabilities.items(), key=lambda x: -x[1]):
        emoji = CATEGORY_EMOJI.get(cat, "")
        color = CATEGORY_COLOR.get(cat, "#888")
        bar_pct = int(prob * 100)
        st.markdown(
            f"""
            <div style="margin-bottom:4px;">
                <span style="width:130px;display:inline-block;font-size:0.85rem;">{emoji} {cat}</span>
                <div style="display:inline-block;width:55%;background:#e0e0e0;
                            border-radius:4px;height:12px;vertical-align:middle;">
                    <div style="width:{bar_pct}%;background:{color};
                                border-radius:4px;height:12px;"></div>
                </div>
                <span style="font-size:0.82rem;color:#444;margin-left:6px;">{prob*100:.1f}%</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_ml_metrics_section():
    """Show a compact model-evaluation note (metrics are computed during training)."""
    with st.expander("📊 ML Model Info", expanded=False):
        st.markdown(
            """
            | Detail | Value |
            |---|---|
            | Algorithm | Logistic Regression |
            | Features | TF-IDF (1–2 grams) |
            | Dataset size | 97 labelled examples |
            | Train / Test split | 80 / 20 (stratified) |
            | Categories | Organic · Recyclable · General Waste |

            Run `python train_model.py` to see the full classification report and accuracy score.
            """,
            unsafe_allow_html=False,
        )


def main():
    st.set_page_config(
        page_title="EcoSort AI",
        page_icon="♻️",
        layout="centered",
    )

    # ── Header ──────────────────────────────────────────────────────────────
    st.markdown(
        """
        <h1 style="text-align:center; margin-bottom:0;">
            ♻️ EcoSort AI
        </h1>
        <p style="text-align:center; color:#555; margin-top:4px; font-size:1.05rem;">
            Smart Waste Classification Assistant
        </p>
        <p style="text-align:center; font-size:0.88rem; color:#777;">
            🌍 Aligned with <strong>UN SDG 12</strong> — Responsible Consumption & Production
        </p>
        <hr style="border:none;border-top:1px solid #ddd;margin:12px 0 20px;">
        """,
        unsafe_allow_html=True,
    )

    # ── Model loading (auto-trains on first run if model.pkl missing) ────────
    try:
        model = load_model()
    except Exception as e:
        st.error(f"⚠️ Could not load or train the model: {e}")
        st.stop()

    # ── Sample items ─────────────────────────────────────────────────────────
    st.markdown("**Try a sample item:**")
    cols = st.columns(len(SAMPLE_ITEMS))
    selected_sample = ""
    for i, sample in enumerate(SAMPLE_ITEMS):
        if cols[i].button(sample, key=f"sample_{i}"):
            selected_sample = sample

    # ── Input ────────────────────────────────────────────────────────────────
    st.markdown("")  # spacing
    user_input = st.text_input(
        "Describe a waste item:",
        value=selected_sample,
        placeholder="e.g. banana peel, glass bottle, used tissue …",
        max_chars=200,
    )

    classify_btn = st.button("🔍 Classify Waste", type="primary", use_container_width=True)

    # ── Classification & advice ──────────────────────────────────────────────
    if classify_btn:
        cleaned = user_input.strip()
        if not cleaned:
            st.warning("Please enter a waste item description before classifying.")
            st.stop()

        with st.spinner("Classifying …"):
            result = classify_waste(model, cleaned)

        st.markdown("### 🏷️ Classification Result")
        render_category_badge(result["category"], result["confidence"])
        render_probabilities(result["probabilities"])

        st.markdown("---")
        st.markdown("### 💡 Disposal Advice")

        advice = get_gemini_advice(cleaned, result["category"])

        if advice:
            st.markdown(
                """
                <p style="font-size:0.78rem; color:#555; margin-bottom:8px;">
                    ✨ <em>Generated by Google Gemini AI based on the ML prediction</em>
                </p>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(advice)
        else:
            api_key_present = bool(os.environ.get("GEMINI_API_KEY", "").strip())
            if not api_key_present:
                st.info(
                    "ℹ️ **Gemini API key not configured.** "
                    "Set `GEMINI_API_KEY` in your `.env` file to enable AI advice. "
                    "Showing static disposal guide instead."
                )
            else:
                st.warning(
                    "⚠️ **Gemini API unavailable.** "
                    "Showing static disposal guide instead."
                )
            st.markdown(FALLBACK_GUIDE.get(result["category"], "No guide available."))

    # ── ML model section ─────────────────────────────────────────────────────
    st.markdown("")
    render_ml_metrics_section()

    # ── Disclaimer ───────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="background:#f7f7f7;border:1px solid #ddd;border-radius:6px;
                    padding:10px 14px;margin-top:20px;font-size:0.82rem;color:#555;">
            <strong>⚠️ Disclaimer:</strong> EcoSort AI provides <em>educational guidance only</em>.
            Actual recycling and disposal rules vary by city, region, and local authority.
            Always follow the waste-management guidelines provided by your local municipality.
            This project was built for academic demonstration purposes.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<p style='text-align:center;font-size:0.75rem;color:#aaa;margin-top:18px;'>"
        "EcoSort AI · Academic Project · SDG 12</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
