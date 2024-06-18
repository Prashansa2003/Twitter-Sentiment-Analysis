import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the cleaned dataset
url = 'training.1600000.processed.noemoticon.csv'  
columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
df = pd.read_csv(url, encoding='latin1', names=columns)

# Data Cleaning
df['date'] = df['date'].str.replace(r' \w+$', '', regex=True)
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['date'] = df['date'].dt.tz_localize('UTC')  

df = df.dropna(subset=['date'])
df = df.drop_duplicates()
df = df.drop(columns=['ids', 'flag', 'user'])
df['target'] = df['target'].apply(lambda x: 0 if x == 0 else 1)

# Sentiment Distribution

# 1. Visualize the distribution of sentiment labels
plt.figure(figsize=(8, 6))
sns.countplot(x='target', data=df, palette='viridis')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], labels=['Negative', 'Positive'])
plt.show()

# 2. Analyze the balance of sentiment classes
sentiment_counts = df['target'].value_counts()
print("\nSentiment Counts:")
print(sentiment_counts)

# Calculate the percentage of each sentiment class
sentiment_percentages = (sentiment_counts / sentiment_counts.sum()) * 100
print("\nSentiment Percentages:")
print(sentiment_percentages)
