# NLP Sentiment Analysis: Traditional ML vs BERT

**Live dashboard:** [Open the dashboard](https://nlp-sentiment-analysis-bert-fahxl47vwczqc44iyxgizg.streamlit.app/)

Binary sentiment classification of social media posts (tweets), comparing traditional machine learning models with a fine-tuned BERT transformer.

## Results

Evaluated on a held-out test set of 10,000 tweets (20% of 50,000 sampled tweets).

| Model               | Accuracy | Precision | Recall | F1-Score |
|---------------------|---------:|----------:|-------:|---------:|
| Naive Bayes         | 73.97%   | 74.78%    | 72.69% | 73.72%   |
| Logistic Regression | 75.59%   | 74.87%    | 77.38% | 76.10%   |
| SVM (LinearSVC)     | 74.27%   | 73.65%    | 75.95% | 74.78%   |
| BERT                | 83.71%   | 82.65%    | 85.53% | 84.06%   |

Precision, recall and F1 are calculated for the positive class.

**Main finding:** fine-tuned BERT outperforms the best traditional model (Logistic Regression) by about 8 points of accuracy and F1.

## Dataset

- Sentiment140 (tweets labelled negative or positive)
- 50,000 randomly sampled tweets, 80/20 train/test split (`random_state=42`)
- Downloaded automatically with `kagglehub`

## Approach

**Traditional models** (Naive Bayes, Logistic Regression, SVM)
- Text cleaning: remove URLs, mentions, hashtags, special characters, numbers; lowercase; remove stopwords; Porter stemming
- TF-IDF features (5,000 features)

**BERT**
- Pre-trained `bert-base-uncased` fine-tuned for 3 epochs
- Light cleaning only (remove URLs and mentions); no stemming or stopword removal, because BERT works best with natural sentences
- Max sequence length 64, batch size 32, learning rate 2e-5, AdamW optimizer

All models use the same train/test split, so the comparison is fair.

## Project Structure

```
NLP-Sentiment-Analysis-BERT/
├── src/
│   ├── preprocessing.py
│   ├── train_models.py
│   └── train_bert.py
├── results/
│   ├── results_traditional.csv
│   ├── results_bert.csv
│   └── bert_predictions.csv
├── requirements.txt
└── README.md
```

## How to Run

```
pip install -r requirements.txt
python src/train_models.py
python src/train_bert.py
```

BERT training is slow on CPU. A GPU (for example a free Google Colab T4) is recommended.

## Work in Progress

- Interactive dashboard (Streamlit) with confusion matrix, sentiment trends and misclassified examples
- Neutral sentiment class
- Model explainability

## Author

**Malsha Nethmini**
[LinkedIn](https://www.linkedin.com/in/malsha-nethmini-vk/)