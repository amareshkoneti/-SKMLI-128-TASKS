import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

# Sample Data
data = {
    'text': [
        'I love this movie', 'I hate this movie', 'This movie is great', 'This movie is terrible',
        'Awesome film', 'Awful film', 'I enjoyed the film', 'I disliked the film',
        'Best movie ever', 'Worst movie ever', 'Fantastic movie', 'Dreadful movie',
        'Excellent plot and acting', 'Poor direction', 'Marvelous story', 'Horrible experience',
        'I would watch it again', 'I will never watch it again', 'It was wonderful', 'It was a disaster',
        'Great acting and story', 'Terrible script and acting', 'Enjoyable and fun', 'Boring and dull',
        'Highly recommended', 'Not recommended', 'Loved the characters', 'Disliked the characters'
    ],
    'label': [
        'positive', 'negative', 'positive', 'negative',
        'positive', 'negative', 'positive', 'negative',
        'positive', 'negative', 'positive', 'negative',
        'positive', 'negative', 'positive', 'negative',
        'positive', 'negative', 'positive', 'negative',
        'positive', 'negative', 'positive', 'negative',
        'positive', 'negative', 'positive', 'negative'
    ]
}

df = pd.DataFrame(data)

# Preprocess the data
# Here, we assume the text is already clean and tokenized. Normally, you'd include steps to clean the text.

# Feature extraction
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['text'])
y = df['label']

# Split the data into training and testing sets using stratified split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train the Naïve Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='positive', average='binary', zero_division=0)
recall = recall_score(y_test, y_pred, pos_label='positive', average='binary', zero_division=0)
classification_rep = classification_report(y_test, y_pred, target_names=['negative', 'positive'], zero_division=0)

# Print the evaluation metrics
print(f'Accuracy: {accuracy*100:.2f}%')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print('Classification Report:')
print(classification_rep)
