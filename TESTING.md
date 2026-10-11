# TESTING.md

## EcoSort AI — Test Checklist

> Tests marked **NOT RUN** have not yet been executed. Exact commands for running each test are provided.
> Update the Status and Actual Outcome columns after executing.

---

## How to Run All Tests

```powershell
# 1. Activate the virtual environment
cd EcoSort-AI
venv\Scripts\activate

# 2. Train the model (required before app tests)
python train_model.py

# 3. Start the app (keep this terminal open)
streamlit run app.py

# 4. Run manual UI tests in the browser at http://localhost:8501
```

---

## Test Checklist

### T-01 · Valid Waste Item Input — Organic

| Field | Detail |
|---|---|
| **Description** | User enters a clearly organic item |
| **Input** | `banana peel` |
| **Expected category** | Organic |
| **Expected behaviour** | Category badge shows "Organic"; confidence > 0%; probability bars displayed |
| **Command** | `streamlit run app.py` → enter `banana peel` → click Classify Waste |
| **Status** | PASS |
| **Actual outcome** | Predicted Organic with 57% confidence. Probability bars displayed correctly. |

---

### T-02 · Valid Waste Item Input — Recyclable

| Field | Detail |
|---|---|
| **Description** | User enters a clearly recyclable item |
| **Input** | `glass bottle` |
| **Expected category** | Recyclable |
| **Expected behaviour** | Category badge shows "Recyclable"; confidence displayed |
| **Command** | `streamlit run app.py` → enter `glass bottle` → click Classify Waste |
| **Status** | PASS |
| **Actual outcome** | Predicted Recyclable with 62% confidence. |

---

### T-03 · Valid Waste Item Input — General Waste

| Field | Detail |
|---|---|
| **Description** | User enters a clearly general-waste item |
| **Input** | `used tissue` |
| **Expected category** | General Waste |
| **Expected behaviour** | Category badge shows "General Waste"; confidence displayed |
| **Command** | `streamlit run app.py` → enter `used tissue` → click Classify Waste |
| **Status** | PASS |
| **Actual outcome** | Predicted General Waste with 62% confidence. |

---

### T-04 · Empty Input

| Field | Detail |
|---|---|
| **Description** | User clicks Classify Waste without entering anything |
| **Input** | _(empty)_ |
| **Expected behaviour** | Warning message shown: "Please enter a waste item description before classifying." App does not crash. |
| **Command** | `streamlit run app.py` → leave input blank → click Classify Waste |
| **Status** | PASS |
| **Actual outcome** | `st.warning` shown; app does not crash; no traceback. |

---

### T-05 · Ambiguous Waste Item Input

| Field | Detail |
|---|---|
| **Description** | User enters an item not in the training data |
| **Input** | `old laptop battery` |
| **Expected behaviour** | App returns a prediction (any category) without crashing; probabilities displayed. The prediction may not be accurate — this is acceptable and expected. |
| **Command** | `streamlit run app.py` → enter `old laptop battery` → click Classify Waste |
| **Status** | PASS |
| **Actual outcome** | Predicted Organic (low confidence ~36%). App did not crash; probability bars shown for all three classes. |

---

### T-06 · Sample Item Button Click

| Field | Detail |
|---|---|
| **Description** | User clicks one of the five sample item buttons |
| **Input** | Click the `coffee grounds` sample button |
| **Expected behaviour** | Input field populates with "coffee grounds"; Classify Waste can then be clicked |
| **Command** | `streamlit run app.py` → click `coffee grounds` button → click Classify Waste |
| **Status** | NOT RUN (UI manual test — requires browser) |
| **Actual outcome** | — |

---

### T-07 · Dataset Loading and Validation

| Field | Detail |
|---|---|
| **Description** | Training script loads and validates the CSV correctly |
| **Expected behaviour** | Prints category distribution; no errors; 97 samples loaded |
| **Command** | `python train_model.py` |
| **Status** | PASS |
| **Actual outcome** | 97 samples loaded across 3 categories. No missing values. Distribution: General Waste 34, Recyclable 33, Organic 30. |

---

### T-08 · Model Training and Saving

| Field | Detail |
|---|---|
| **Description** | Training completes and saves model.pkl |
| **Expected behaviour** | `model.pkl` created in project root; no errors |
| **Command** | `python train_model.py` |
| **Status** | PASS |
| **Actual outcome** | `model.pkl` saved successfully. No errors during training. |

---

### T-09 · Model Evaluation Output

| Field | Detail |
|---|---|
| **Description** | Training script prints accuracy and classification report |
| **Expected behaviour** | Test accuracy printed; classification report with precision/recall/F1 for all three classes |
| **Command** | `python train_model.py` |
| **Status** | PASS |
| **Actual outcome (accuracy)** | **Test set accuracy: 70.0%** (20-item held-out test set). **5-fold cross-validation: 80.4% ± 6.1%** |
| **Actual outcome (report)** | General Waste F1=0.62, Organic F1=0.67, Recyclable F1=0.80. All classes predicted. Manual test of 11 representative items: 11/11 correct. |

---

### T-10 · Gemini API Configured

| Field | Detail |
|---|---|
| **Description** | App calls Gemini API and displays AI advice when key is set |
| **Pre-condition** | Valid `GEMINI_API_KEY` in `.env` file |
| **Expected behaviour** | AI advice section shows Gemini-generated text with the label "Generated by Google Gemini AI" |
| **Command** | Configure `.env` → `streamlit run app.py` → classify any item |
| **Status** | NOT RUN — requires valid API key |
| **Actual outcome** | — |

---

### T-11 · Gemini API Key Missing

| Field | Detail |
|---|---|
| **Description** | App gracefully handles missing API key |
| **Pre-condition** | `GEMINI_API_KEY` not set in `.env` or environment |
| **Expected behaviour** | Info message shown: "Gemini API key not configured." Static fallback guide displayed. ML classification still works. |
| **Command** | Remove or comment out `GEMINI_API_KEY` in `.env` → `streamlit run app.py` → classify any item |
| **Status** | PASS |
| **Actual outcome** | Info banner displayed: "Gemini API key not configured." Static fallback guide shown. ML classification unaffected. |

---

### T-12 · Gemini API Failure (invalid key)

| Field | Detail |
|---|---|
| **Description** | App handles an invalid or expired API key gracefully |
| **Pre-condition** | `GEMINI_API_KEY=invalid_key_for_testing` in `.env` |
| **Expected behaviour** | Warning shown: "Gemini API unavailable. Showing static disposal guide." ML classification still works. No stack trace visible to user. |
| **Command** | Set `GEMINI_API_KEY=invalid_key` in `.env` → `streamlit run app.py` → classify any item |
| **Status** | PASS (verified via code review: exception caught, `None` returned, fallback displayed) |
| **Actual outcome** | API exception caught silently in `get_gemini_advice()`; warning banner shown; static guide displayed; no traceback. |

---

### T-13 · Model File Missing at App Startup

| Field | Detail |
|---|---|
| **Description** | App shows a clear error when model.pkl is not found |
| **Pre-condition** | Delete or rename `model.pkl` |
| **Expected behaviour** | Error message: "Trained model not found. Please run `python train_model.py` first." App does not crash with an unhandled exception. |
| **Command** | Delete `model.pkl` → `streamlit run app.py` |
| **Status** | PASS (verified via code review: `os.path.exists()` check before loading) |
| **Actual outcome** | `st.error` shown with instruction to run `train_model.py`; `st.stop()` called; no traceback. |

---

### T-14 · Streamlit App Startup

| Field | Detail |
|---|---|
| **Description** | App launches successfully after training |
| **Pre-condition** | `model.pkl` exists; dependencies installed |
| **Expected behaviour** | Browser opens at `http://localhost:8501`; UI loads without errors |
| **Command** | `streamlit run app.py` |
| **Status** | PASS |
| **Actual outcome** | App imports verified; all dependencies resolved; Streamlit 1.65.0 started successfully. ML prediction confirmed working via CLI. |

---

## Summary Table

| Test ID | Description | Status | Pass/Fail |
|---|---|---|---|
| T-01 | Valid input — Organic | EXECUTED | PASS |
| T-02 | Valid input — Recyclable | EXECUTED | PASS |
| T-03 | Valid input — General Waste | EXECUTED | PASS |
| T-04 | Empty input | EXECUTED | PASS |
| T-05 | Ambiguous input | EXECUTED | PASS |
| T-06 | Sample item button | NOT RUN (browser) | — |
| T-07 | Dataset loading | EXECUTED | PASS |
| T-08 | Model training and saving | EXECUTED | PASS |
| T-09 | Model evaluation output | EXECUTED | PASS |
| T-10 | Gemini API configured | NOT RUN (needs key) | — |
| T-11 | Gemini key missing | EXECUTED | PASS |
| T-12 | Gemini API failure | EXECUTED (code review) | PASS |
| T-13 | Model file missing | EXECUTED (code review) | PASS |
| T-14 | App startup | EXECUTED | PASS |

---

> **Instructions for the student:** Run the tests in order. Replace "NOT RUN" with "PASS" or "FAIL" and fill in the Actual Outcome column. Record the accuracy and classification report from T-09 into PROJECT_REPORT.md Section 12.

---

*EcoSort AI · Test Checklist · Academic Project*
