import pandas as pd

# Load the dataset
url = 'training.1600000.processed.noemoticon.csv'

columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
df = pd.read_csv(url, encoding='latin1', names=columns)

print("First few rows of the dataset:")
print(df.head())
