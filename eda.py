import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Loading the cleaned dataset
url = 'training.1600000.processed.noemoticon.csv'  
columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
df = pd.read_csv(url, encoding='latin1', names=columns)

# Data Cleaning
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date'])
df = df.drop_duplicates()
df = df.drop(columns=['ids', 'flag', 'user'])
df['target'] = df['target'].apply(lambda x: 0 if x == 0 else 1)

# Exploratory Data Analysis (EDA)

# 1. Sentiment Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='target', data=df, palette='viridis')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], labels=['Negative', 'Positive'])
plt.show()

# 2. Temporal Trends
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month

plt.figure(figsize=(14, 7))
sns.countplot(x='year', hue='target', data=df, palette='viridis')
plt.title('Number of Tweets per Year by Sentiment')
plt.xlabel('Year')
plt.ylabel('Count')
plt.legend(title='Sentiment', loc='upper right', labels=['Negative', 'Positive'])
plt.show()

plt.figure(figsize=(14, 7))
sns.countplot(x='month', hue='target', data=df, palette='viridis')
plt.title('Number of Tweets per Month by Sentiment')
plt.xlabel('Month')
plt.ylabel('Count')
plt.legend(title='Sentiment', loc='upper right', labels=['Negative', 'Positive'])
plt.show()

# 3. Word Cloud for Positive Tweets
positive_tweets = ' '.join(df[df['target'] == 1]['text'])
wordcloud_positive = WordCloud(width=800, height=400, max_font_size=100, background_color='white').generate(positive_tweets)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud_positive, interpolation='bilinear')
plt.title('Word Cloud for Positive Tweets')
plt.axis('off')
plt.show()

# 4. Word Cloud for Negative Tweets
negative_tweets = ' '.join(df[df['target'] == 0]['text'])
wordcloud_negative = WordCloud(width=800, height=400, max_font_size=100, background_color='white').generate(negative_tweets)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud_negative, interpolation='bilinear')
plt.title('Word Cloud for Negative Tweets')
plt.axis('off')
plt.show()

# 5. Length of Tweets
df['tweet_length'] = df['text'].apply(len)

plt.figure(figsize=(14, 7))
sns.histplot(df[df['target'] == 1]['tweet_length'], bins=50, color='green', label='Positive', kde=True)
sns.histplot(df[df['target'] == 0]['tweet_length'], bins=50, color='red', label='Negative', kde=True)
plt.title('Distribution of Tweet Lengths by Sentiment')
plt.xlabel('Tweet Length')
plt.ylabel('Frequency')
plt.legend()
plt.show()
