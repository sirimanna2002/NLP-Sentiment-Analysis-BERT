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