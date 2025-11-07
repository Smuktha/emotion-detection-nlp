# -----------------------------
# 🧠 Emotion Detection
# Author: Muktha 
# -----------------------------

import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import resample

# -----------------------------------
# STEP 1️⃣ - Load Dataset
# -----------------------------------
df = pd.read_csv("tweet_emotions.csv")  
df = df[['content', 'sentiment']].rename(columns={'content': 'text', 'sentiment': 'label'})

print("✅ Dataset loaded successfully!")
print(df.head())

# -----------------------------------
# STEP 2️⃣ - Simplify Labels
# Keep top 7 frequent emotions only
# -----------------------------------
top_labels = df['label'].value_counts().nlargest(7).index
df = df[df['label'].isin(top_labels)]
print("\n🧩 Using Top Emotions:\n", df['label'].value_counts())

# -----------------------------------
# STEP 3️⃣ - Balance the Dataset
# -----------------------------------
balanced_df = []
min_size = df['label'].value_counts().min()

for emotion in df['label'].unique():
    subset = df[df['label'] == emotion]
    balanced_subset = resample(subset, 
                               replace=True, 
                               n_samples=min_size, 
                               random_state=42)
    balanced_df.append(balanced_subset)

df_balanced = pd.concat(balanced_df)
df = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
print("\n✅ Balanced Label Distribution:\n", df['label'].value_counts())

# -----------------------------------
# STEP 4️⃣ - Text Cleaning Function
# -----------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # remove links
    text = re.sub(r"@\S+", "", text)     # remove mentions
    text = re.sub(r"#", "", text)        # remove hashtags
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r"\d+", "", text)      # remove numbers
    text = text.strip()
    return text

df['text'] = df['text'].apply(clean_text)

# -----------------------------------
# STEP 5️⃣ - Split Dataset
# -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label'])

# -----------------------------------
# STEP 6️⃣ - TF-IDF Vectorization
# -----------------------------------
vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1,3))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -----------------------------------
# STEP 7️⃣ - Train Model (Logistic Regression)
# -----------------------------------
model = LogisticRegression(max_iter=1000, C=2)
model.fit(X_train_vec, y_train)

# -----------------------------------
# STEP 8️⃣ - Evaluate Model
# -----------------------------------
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ Accuracy: {accuracy * 100:.2f} %")
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# -----------------------------------
# STEP 9️⃣ - Save Model and Vectorizer
# -----------------------------------
import pickle

with open("emotion_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\n🎉 Model and Vectorizer saved successfully!")
