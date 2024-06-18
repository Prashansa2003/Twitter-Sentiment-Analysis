import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Loading the dataset
url = 'training.1600000.processed.noemoticon.csv'  
columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
df = pd.read_csv(url, encoding='latin1', names=columns)

df = df[['target', 'date', 'text']]

df['target'] = df['target'].apply(lambda x: 1 if x == 4 else 0)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Preprocess text
def preprocess_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\W', ' ', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word.lower() not in stop_words]
    return ' '.join(tokens)

# Apply text preprocessing
df['cleaned_text'] = df['text'].apply(preprocess_text)

# Feature extraction
vectorizer = CountVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_text']).toarray()
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'F1 Score: {f1}')

# Get feature importance
feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_[0]
feature_importance = pd.DataFrame({'feature': feature_names, 'importance': coefficients})
feature_importance['absolute_importance'] = feature_importance['importance'].abs()

# Top positive and negative features
top_positive_features = feature_importance.sort_values(by='importance', ascending=False).head(10)
top_negative_features = feature_importance.sort_values(by='importance', ascending=True).head(10)

# Plotting bar charts
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.barh(top_positive_features['feature'], top_positive_features['importance'], color='green')
plt.title('Top Positive Features')
plt.xlabel('Coefficient Value')

plt.subplot(1, 2, 2)
plt.barh(top_negative_features['feature'], top_negative_features['importance'], color='red')
plt.title('Top Negative Features')
plt.xlabel('Coefficient Value')

plt.tight_layout()
plt.show()

# Generate word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(zip(feature_names, coefficients)))

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Feature Importance Word Cloud')
plt.show()
