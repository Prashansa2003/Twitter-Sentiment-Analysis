import pandas as pd

url = 'training.1600000.processed.noemoticon.csv'  
columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
df = pd.read_csv(url, encoding='latin1', names=columns)

print("First few rows of the dataset:")
print(df.head())

print("\nSummary of the dataset:")
print(df.info())

print("\nStatistical summary of numeric columns:")
print(df.describe())

print("\nDistribution of sentiment labels:")
print(df['target'].value_counts())

print("\nMissing values in the dataset:")
print(df.isnull().sum())

df['date'] = pd.to_datetime(df['date'], errors='coerce')

print("\nFirst few rows of the dataset after converting the date column:")
print(df.head())
