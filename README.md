# 💬 AI‑Powered FAQs Chatbot (Google Colab + Transformers + Gradio)

**One‑liner:** A semantic, CSV‑driven FAQ chatbot that runs end‑to‑end in **Google Colab**, uses **Sentence Transformers** for intelligent matching, and ships with a **beautiful Gradio chat UI**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1WUNB28WjbG6pPqstwdL7q8wM6f1wx7BF?usp=sharing)


---

## Table of Contents

* [What](#what)
* [Why](#why)
* [How it Works (Architecture)](#how-it-works-architecture)
* [Dataset & Schema](#dataset--schema)
* [Requirements](#requirements)
* [Quickstart (Google Colab)](#quickstart-google-colab)
* [Configuration & Tuning](#configuration--tuning)
* [UI/UX Details](#uiux-details)
* [Evaluation & Quality Checks](#evaluation--quality-checks)
* [Troubleshooting](#troubleshooting)
* [Security & Privacy](#security--privacy)
* [Performance Notes](#performance-notes)
* [FAQ](#faq)
* [License](#license)

---

## What

A **semantic FAQ chatbot** that:

* Reads your **CSV** file of *Question → Answer* pairs.
* **Embeds** all questions using a compact transformer model (`all-MiniLM-L6-v2`).
* Performs fast **cosine similarity** search to find the closest match to the user’s query.
* Returns the corresponding **answer** in a clean, responsive **Gradio** chat interface.

You can run it entirely in **Google Colab** (no local setup) and optionally deploy to **HuggingFace Spaces**.

---

## Why

Traditional TF‑IDF keyword matching often fails on:

* **Paraphrases** ("How can I reset my password?" vs "Forgot my password—what do I do?").
* **Synonyms** and **word order** differences.

By switching to **Sentence Transformers**, we get meaningful **semantic similarity**, leading to:

* Higher **accuracy** and **user satisfaction**.
* Better **generalization** to unseen phrasing.
* A more **natural** chat experience.

---

## How it Works (Architecture)

**Flow:**

1. **Data Load:** Read `faqs.csv` → two columns: `Question`, `Answer`.
2. **Embedding:** Use `SentenceTransformer('all-MiniLM-L6-v2')` to embed *all* FAQ questions → `faq_embeddings`.
3. **Query:** On each user input, compute `user_embedding`.
4. **Similarity:** Compute cosine similarity between `user_embedding` and `faq_embeddings`.
5. **Selection:** Pick the top‑1 (or top‑k) match. Optionally apply a **confidence threshold**.
6. **Response:** Return the corresponding answer in the **Gradio** chat UI.

> Minimal Dependencies, No External Services: Embeddings are computed in‑notebook; no API keys required.

---

## Dataset & Schema

**CSV Requirements:**

* Must contain at least these columns:

  * `Question` — the user‑facing question text.
  * `Answer` — the canonical answer text.

**Example:**

```csv
Question,Answer
What is AI?,AI stands for Artificial Intelligence. It enables machines to mimic human intelligence.
What is Machine Learning?,Machine Learning is a subset of AI that learns patterns from data to make predictions or decisions.
How do I reset my password?,Click “Forgot Password” on the login page and follow the instructions sent to your email.
```

**Quality Tips:**

* Avoid duplicates. Consolidate near‑identical questions into a single canonical form.
* Keep answers **concise** but **complete**.
* If the same question has multiple correct answers, unify or list options clearly.

---

## Requirements

* **Runtime:** Google Colab (GPU not required).
* **Python libs:**

  * `pandas`
  * `sentence-transformers`
  * `gradio`

**Colab Install Cell:**

```python
!pip install -q pandas sentence-transformers gradio
```

---

## Quickstart (Google Colab)

**1) Setup & Imports**

```python
!pip install -q pandas sentence-transformers gradio

import pandas as pd
from sentence_transformers import SentenceTransformer, util
import gradio as gr
from google.colab import files
```

**2) Upload Your CSV**

```python
uploaded = files.upload()  # Choose your faqs.csv from your laptop
csv_path = list(uploaded.keys())[0]
faq_df = pd.read_csv(csv_path)
assert {'Question','Answer'}.issubset(faq_df.columns), "CSV must have 'Question' and 'Answer' columns"
faq_df.head()
```

**3) Load Model & Precompute Embeddings**

```python
model = SentenceTransformer('all-MiniLM-L6-v2')
faq_questions = faq_df['Question'].fillna("").tolist()
faq_answers = faq_df['Answer'].fillna("").tolist()
faq_embeddings = model.encode(faq_questions, convert_to_tensor=True, normalize_embeddings=True)
```

**4) Retrieval & Chat UI**

```python
TOP_K = 3        # return top-3 candidates internally
THRESHOLD = 0.35 # confidence gate (0–1); tweak for your domain

def retrieve_best_answer(query: str):
    if not query or not query.strip():
        return None, []
    q_emb = model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    scores = util.cos_sim(q_emb, faq_embeddings)[0]
    topk_scores, topk_idx = scores.topk(k=min(TOP_K, len(faq_questions)))
    candidates = [(float(topk_scores[i]), int(topk_idx[i])) for i in range(len(topk_idx))]
    # Thresholding
    best_score, best_idx = candidates[0]
    if best_score < THRESHOLD:
        return None, candidates
    return faq_answers[best_idx], candidates

with gr.Blocks(css=".gradio-container {background: #0b0b0c; color: #eaeaea;}") as demo:
    gr.Markdown("""
    <h1 style='text-align:center'>💬 AI‑Powered FAQ Chatbot</h1>
    <p style='text-align:center'>Upload your CSV and start asking questions.</p>
    """)

    chatbot = gr.Chatbot(height=420, label="Chat")
    with gr.Row():
        msg = gr.Textbox(placeholder="Ask me anything from the FAQs…", scale=4)
        clear = gr.Button("Clear", scale=1)

    def respond(message, chat_history):
        answer, candidates = retrieve_best_answer(message)
        if answer is None:
            answer = ("I couldn't confidently match your question to our FAQs. "
                      "Try rephrasing or contact support.")
        chat_history.append((message, answer))
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch()
```

**5) Using the App**

* Run all cells → Upload your CSV → Type a question.
* The bot returns the best‑matching answer from your dataset.

**How to Upload a File in Colab:**

* In the notebook, run:

  ```python
  from google.colab import files
  uploaded = files.upload()
  ```
* A file picker opens → choose your CSV → it appears in the Colab workspace.

> Large CSVs: You can also **mount Google Drive** (`from google.colab import drive; drive.mount('/content/drive')`) and read from there.

---

## Configuration & Tuning

* **Model:** `all-MiniLM-L6-v2` balances speed/quality. For higher accuracy, try `multi-qa-MiniLM-L6-cos-v1` or `all-mpnet-base-v2` (slower, better).
* **Normalization:** `normalize_embeddings=True` often improves cosine similarity stability.
* **TOP\_K:** Retrieve multiple candidates internally (e.g., 3–5) and pick best by score or add a heuristic re‑rank.
* **THRESHOLD:** Add a confidence floor (0–1). If the best score is below this, show a fallback message or escalate.
* **Preprocessing:** You can lowercase, strip punctuation, or collapse whitespace for questions before embedding (usually not necessary for transformers, but harmless).
* **Answer Formatting:** Add markdown support in Gradio for richer answers (links, bullet points).

---

## UI/UX Details

* **Gradio Chatbot** for conversational feel.
* **Clear** button to reset state.
* Dark theme with subtle typography.
* You can add branding via `gr.Markdown` (logo, colors) and set `title` in `demo.launch()`.

**Ideas:**

* Show **source question** under each answer for transparency.
* Display **top‑k matches with scores** (developer/debug mode).
* Add **feedback buttons** (✅ Helpful / ❌ Not Helpful) to improve the dataset.

---

## Evaluation & Quality Checks

To assess answer quality:

* **Manual spot‑checks:** Curate a validation set of 50–100 real user queries.
* **Metrics:**

  * *Top‑1 Accuracy*: `answer matched?` yes/no.
  * *MRR (Mean Reciprocal Rank)*: Measures ranking quality if you keep top‑k.
  * *nDCG\@k*: Graded relevance if multiple answers are partially correct.
* **A/B testing:** Try different models/thresholds.
* **Error analysis:** Cluster failure cases (missing FAQs, ambiguous wording, jargon).

---

## Troubleshooting

* **`KeyError: 'Question'`** → Ensure your CSV column headers are exactly `Question, Answer`.
* **Empty or poor results** →

  * Lower `THRESHOLD` slightly (e.g., 0.25–0.30).
  * Improve dataset coverage (add more paraphrases).
  * Try a stronger model.
* **Slow first run** → Model download happens once per session.
* **Unicode/CSV issues** → Save as UTF‑8. In `pd.read_csv`, try `encoding='utf-8'`.

---

## Security & Privacy

* Data stays in your **Colab session** by default.
* Avoid uploading sensitive information.
* If deploying publicly, scrub PII and restrict content to non‑confidential FAQs.

---

## FAQ

**Q: Can I use languages other than English?**
A: Yes—many sentence transformer models are multilingual (e.g., `distiluse-base-multilingual-cased-v1`). Swap the model name and re‑embed.

**Q: How big can my CSV be?**
A: Colab handles thousands of rows comfortably. For very large datasets, use FAISS and store embeddings on disk.

**Q: Can I keep conversation history?**
A: Yes—store history in `chat_history` and add follow‑up logic (context carryover, disambiguation prompts).

**Q: What if no match is found?**
A: Use a threshold and a helpful fallback message or a link to support.

**Q: How do I brand the UI?**
A: Customize Gradio `Blocks` with your logo, colors, and fonts via CSS and Markdown.

---

## License

MIT © 2025. Use freely with attribution.

---
