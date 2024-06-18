import pandas as pd
import matplotlib.pyplot as plt

# Loading the cleaned dataset
url = 'training.1600000.processed.noemoticon.csv'  
columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
df = pd.read_csv(url, encoding='latin1', names=columns)

# Data Cleaning
df['date'] = df['date'].str.replace(r' \w+$', '', regex=True)
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['date'] = df['date'].dt.tz_localize('UTC')  # Localize to UTC

df = df.dropna(subset=['date'])
df = df.drop_duplicates()
df = df.drop(columns=['ids', 'flag', 'user'])
df['target'] = df['target'].apply(lambda x: 0 if x == 0 else 1)

# Aggregate sentiment data by day
df['day'] = df['date'].dt.date
daily_sentiment = df.groupby(['day', 'target']).size().unstack(fill_value=0)

# Calculate the proportion of each sentiment
daily_sentiment['total'] = daily_sentiment.sum(axis=1)
daily_sentiment['negative'] = daily_sentiment[0] / daily_sentiment['total']
daily_sentiment['positive'] = daily_sentiment[1] / daily_sentiment['total']

# Visualize sentiment trends over time
plt.figure(figsize=(15, 8))
plt.plot(daily_sentiment.index, daily_sentiment['negative'], label='Negative Sentiment', color='red')
plt.plot(daily_sentiment.index, daily_sentiment['positive'], label='Positive Sentiment', color='green')
plt.title('Sentiment Proportions Over Time')
plt.xlabel('Date')
plt.ylabel('Proportion of Tweets')
plt.legend()
plt.grid(True)
plt.show()
