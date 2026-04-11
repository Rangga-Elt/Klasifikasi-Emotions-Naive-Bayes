# Import libraries
import io
import string
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from google.colab import files
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import ComplementNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, learning_curve, validation_curve, GridSearchCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

# Upload file dataset
def upload_file():
    uploaded = files.upload()
    for filename in uploaded.keys():
        print(f"File '{filename}' berhasil diunggah.")
        return filename, uploaded

file_name, uploaded = upload_file()
if not file_name:
    raise ValueError("File tidak dipilih. Program berhenti.")

# Load dataset
if file_name.endswith('.csv'):
    df = pd.read_csv(io.BytesIO(uploaded[file_name]))
elif file_name.endswith('.xlsx'):
    df = pd.read_excel(io.BytesIO(uploaded[file_name]))
else:
    raise ValueError("Format file tidak didukung. Harap unggah file .csv atau .xlsx.")

# Preprocessing
# Handle missing values
df['text'] = df['text'].fillna('')

# lowercasing
df['text'] = df['text'].str.lower()

# Remove Punctuation
df['text'] = df['text'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))

# Remove Double Space
df['text'] = df['text'].apply(lambda x: ' '.join(x.split()))

# Hapus Data Kosong Jika Hasil Preprocessing Menghasilkan Data Kosong
df = df[df['text'].apply(lambda x: len(x) > 0)]

# Pisahkan Label
X_text = df['text']
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_text, y, test_size=0.3, random_state=42, stratify=y
)

# Pipeline F-IDF dan Classifier
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', ComplementNB())
])

# Parameter
param_grid = {
    'tfidf__stop_words':['english'],
    'tfidf__token_pattern': ['(?u)\\b\\w\\w+\\b'],
    'tfidf__ngram_range': [(1, 3)],
    'tfidf__max_df': [1.0],
    'tfidf__min_df': [1],
    'tfidf__max_features': [10000],
    'tfidf__analyzer': ['word'],
    'tfidf__norm': ['l1', 'l2'],
    'tfidf__use_idf': [True, False],
    'tfidf__smooth_idf': [True, False],
    'tfidf__sublinear_tf': [True, False],
    'clf__alpha': [0.8],
    'clf__fit_prior': [True],
    'clf__norm': [False]
}

print(f"Jumlah dokumen: {len(X_train)}")

# Grid search
grid_search = GridSearchCV(pipeline, param_grid, cv=10, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# TF-IDF awal
tfidf_initial = TfidfVectorizer()
X_tfidf_initial = tfidf_initial.fit_transform(X_train)
initial_features = tfidf_initial.get_feature_names_out()
print("Jumlah Fitur Awal Hasil Vektorisasi TF-IDF:", len(initial_features))

# Setelah GridSearch
best_vectorizer = grid_search.best_estimator_.named_steps['tfidf']
tuned_features = best_vectorizer.get_feature_names_out()
print("Jumlah Fitur Setelah Tuning dengan Parameter tfidf__max_features :", len(tuned_features))

# Best result
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

# Evaluasi Final
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Metrik Evaluasi
acc = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_test, y_pred, average='weighted', zero_division=0
)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Heatmap confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=True, yticklabels=True)
plt.title('Heatmap Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Gunakan pipeline yang belum fit, tapi sudah di-set param terbaik
model_for_curve = pipeline.set_params(**grid_search.best_params_)

# Validation Curve
param_name = 'clf__alpha'
param_range = [0.1, 0.5, 1.0, 2.0, 5.0]

train_scores, test_scores = validation_curve(
    model_for_curve, X_train, y_train,
    param_name=param_name, param_range=param_range,
    cv=5, scoring="accuracy", n_jobs=-1
)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure(figsize=(8, 6))
plt.plot(param_range, train_scores_mean, label="Training score", color="blue")
plt.plot(param_range, test_scores_mean, label="Validation score", color="red")
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2, color="blue")
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2, color="red")
plt.title("Validation Curve untuk Alpha")
plt.xlabel("Alpha (clf__alpha)")
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.grid()
plt.show()

# Learning Curve
train_sizes = np.linspace(0.1, 1.0, 10)

train_sizes, train_scores, test_scores = learning_curve(
    model_for_curve, X_train, y_train,
    train_sizes=train_sizes, cv=5, scoring="accuracy", n_jobs=-1
)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_scores_mean, label="Training score", color="blue")
plt.plot(train_sizes, test_scores_mean, label="Cross-validation score", color="red")
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2, color="blue")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2, color="red")
plt.title("Learning Curve")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.grid()
plt.show()
