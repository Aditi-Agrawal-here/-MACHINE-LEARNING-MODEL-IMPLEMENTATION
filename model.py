import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# Load dataset
try:
    df = pd.read_csv('spam.csv', encoding='latin-1')
except FileNotFoundError:
    print("Error: 'spam.csv' not found. Make sure the file is in the correct directory.")
    exit()

# Drop unnecessary columns and rename important ones
df = df.iloc[:, :2]  # Selecting only the first two columns
df.columns = ['label', 'message']

# Handle missing values
df.dropna(inplace=True)

# Convert labels to numerical values (spam = 1, ham = 0)
df['label'] = df['label'].map({'spam': 1, 'ham': 0})

# Text vectorization
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['message'].astype(str)).toarray()  # Ensure all messages are strings

# Display shape of transformed data
print("Shape of transformed data:", X.shape)
