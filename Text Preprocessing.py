import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Path to the CSV file
url = 'training.1600000.processed.noemoticon.csv' 

# Load the data into a DataFrame
df = pd.read_csv(url, encoding='latin-1', header=None)

# Assuming the tweets are in the 6th column (index 5)
df.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']

# Initialize lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove special characters and numbers
    text = re.sub(r'\@\w+|\#\w+|\d+|[^A-Za-z\s]', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize text
    word_tokens = word_tokenize(text)
    
    # Remove stop words and lemmatize
    filtered_text = [lemmatizer.lemmatize(w) for w in word_tokens if w not in stop_words]
    
    return ' '.join(filtered_text)

# Apply preprocessing to the 'text' column
df['processed_text'] = df['text'].apply(preprocess_text)

# Print the first few rows of the DataFrame to verify
print(df[['text', 'processed_text']].head())
