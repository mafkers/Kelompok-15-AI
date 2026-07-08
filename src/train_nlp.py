import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Load dataset
base_dir = os.path.dirname(os.path.dirname(__file__))
dataset_path = os.path.join(base_dir, 'data', 'dataset_keluhan.csv')
df = pd.read_csv(dataset_path)

# Preprocessing
factory = StemmerFactory()
stemmer = factory.create_stemmer()

print("Melakukan stemming teks...")
df['keluhan_bersih'] = df['keluhan'].apply(stemmer.stem)

X = df['keluhan_bersih']
y = df['kegawatan']

# Train/test split
# Split 70-30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Evaluasi
y_pred = model.predict(X_test_vec)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print("=== Hasil Evaluasi Model NLP ===")
print(f"Accuracy : {acc*100:.2f}%")
print(f"Precision: {prec*100:.2f}%")
print(f"Recall   : {rec*100:.2f}%")
print(f"F1-Score : {f1*100:.2f}%")

# Simpan model
base_dir = os.path.dirname(os.path.dirname(__file__))
models_dir = os.path.join(base_dir, 'models')
os.makedirs(models_dir, exist_ok=True)

with open(os.path.join(models_dir, 'vectorizer.pkl'), 'wb') as f:
    pickle.dump(vectorizer, f)

with open(os.path.join(models_dir, 'kegawatan_model.pkl'), 'wb') as f:
    pickle.dump(model, f)

print("Model dan Vectorizer berhasil disimpan di folder models/")
