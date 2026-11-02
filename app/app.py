import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix

RESULTS = Path(__file__).resolve().parent.parent / "results"
LABELS = {0: "Negative", 1: "Positive"}

st.set_page_config(page_title="Sentiment Dashboard", layout="wide")
st.title("Social Media Sentiment Dashboard")
st.caption("Traditional ML vs BERT on 50,000 Sentiment140 tweets")


@st.cache_data
def load_results():
    trad = pd.read_csv(RESULTS / "results_traditional.csv")
    bert = pd.read_csv(RESULTS / "results_bert.csv")
    return pd.concat([trad, bert], ignore_index=True)


@st.cache_data
def load_predictions():
    return pd.read_csv(RESULTS / "bert_predictions.csv")


results = load_results()
preds = load_predictions()

# ---------- Section 1: model comparison ----------
st.header("1. Model comparison")
col1, col2 = st.columns(2)

with col1:
    st.dataframe(
        results.set_index("Model").style.format("{:.2%}"),
        width="stretch",
    )

with col2:
    st.bar_chart(results.set_index("Model")[["Accuracy", "F1"]], stack=False)

# ---------- Section 2: confusion matrix ----------
st.header("2. BERT confusion matrix")
cm = confusion_matrix(preds["actual"], preds["predicted"])

fig, ax = plt.subplots(figsize=(4, 3))
ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels([LABELS[0], LABELS[1]])
ax.set_yticklabels([LABELS[0], LABELS[1]])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")
left, _ = st.columns([1, 2])
with left:
    st.pyplot(fig)

# ---------- Section 3: misclassified examples ----------
st.header("3. Misclassified tweets (BERT)")
wrong = preds[preds["actual"] != preds["predicted"]].copy()
wrong["Actual"] = wrong["actual"].map(LABELS)
wrong["Predicted"] = wrong["predicted"].map(LABELS)

st.write(f"BERT got {len(wrong):,} of {len(preds):,} test tweets wrong.")

choice = st.selectbox(
    "Show errors of type:",
    ["All", "Actually Positive, predicted Negative", "Actually Negative, predicted Positive"],
)
if choice == "Actually Positive, predicted Negative":
    wrong = wrong[(wrong["actual"] == 1) & (wrong["predicted"] == 0)]
elif choice == "Actually Negative, predicted Positive":
    wrong = wrong[(wrong["actual"] == 0) & (wrong["predicted"] == 1)]

st.dataframe(
    wrong[["text", "Actual", "Predicted"]].head(50),
    width="stretch",
)


# ---------- Section 4: live prediction ----------
st.header("4. Try it yourself (live BERT prediction)")

MODEL_ID = "MalshaVK/bert-sentiment140"


@st.cache_resource
def load_model():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval()
    return tokenizer, model


def light_clean(text):
    # Same cleaning that was used when training BERT
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    return text.strip()


user_text = st.text_area(
    "Type a tweet or sentence (English):",
    "I love this phone, the battery is amazing!",
)

if st.button("Predict sentiment"):
    import torch

    with st.spinner("Loading BERT (the first time takes about a minute)..."):
        tokenizer, model = load_model()

    enc = tokenizer(
        light_clean(user_text),
        truncation=True,
        padding="max_length",
        max_length=64,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
        ).logits
    probs = torch.softmax(logits, dim=1)[0]

    confidence = float(probs.max())
    label = LABELS[int(probs.argmax())]
    if confidence < 0.65:
        label = "Uncertain (could be neutral)"

    st.subheader(f"Prediction: {label}")
    st.write(f"Confidence: {confidence:.1%}")
    st.progress(float(probs[1]), text=f"Positive probability: {float(probs[1]):.1%}")