import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from preprocessing import clean_text


def load_and_preprocess_data():
    """
    Dataset Load and Preprocess
    """
    # Kaggle Dataset Download
    import kagglehub
    path = kagglehub.dataset_download("kazanova/sentiment140")

    # Dataset Load
    df = pd.read_csv(path + '/training.1600000.processed.noemoticon.csv',
                     encoding='latin-1',
                     header=None,
                     names=['target', 'id', 'date', 'flag', 'user', 'text'])

    df = df[['target', 'text']]
    df['target'] = df['target'].replace(4, 1)  # 4 = Positive, 0 = Negative

    df = df.sample(n=50000, random_state=42)

    # Preprocess
    df['cleaned_text'] = df['text'].apply(clean_text)

    return df


def evaluate_model(name, y_true, y_pred):
    """
    Calculate all metrics for one model and print the full report
    """
    print(f"\n{name}")
    print(classification_report(y_true, y_pred, target_names=['Negative', 'Positive']))

    return {
        'Model': name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred),
    }


def train_models():
    """
    Traditional ML Models Train
    """
    # Data Load
    df = load_and_preprocess_data()

    # Train/Test Split
    X = df['cleaned_text']
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    results = []

    # 1. Naive Bayes
    print("Naive Bayes Train...")
    nb = MultinomialNB()
    nb.fit(X_train_tfidf, y_train)
    nb_pred = nb.predict(X_test_tfidf)
    results.append(evaluate_model('Naive Bayes', y_test, nb_pred))

    # 2. Logistic Regression
    print("Logistic Regression Train...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_tfidf, y_train)
    lr_pred = lr.predict(X_test_tfidf)
    results.append(evaluate_model('Logistic Regression', y_test, lr_pred))

    # 3. SVM
    print("SVM Train...")
    svm = LinearSVC(random_state=42)
    svm.fit(X_train_tfidf, y_train)
    svm_pred = svm.predict(X_test_tfidf)
    results.append(evaluate_model('SVM', y_test, svm_pred))

    return results, y_test, nb_pred, lr_pred, svm_pred


if __name__ == "__main__":
    results, y_test, nb_pred, lr_pred, svm_pred = train_models()

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 60)
    print("Traditional Models Results")
    print("=" * 60)
    print(results_df.round(4).to_string(index=False))

    # Save to a new file (keeps your old results.csv safe)
    results_df.to_csv('results_traditional.csv', index=False)