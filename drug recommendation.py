# Import necessary libraries
import pandas as pd  # pandas is a library for data manipulation
from sklearn.model_selection import train_test_split  # scikit-learn helps with machine learning
from sklearn.feature_extraction.text import CountVectorizer  # helps convert symptoms to numbers
from sklearn.naive_bayes import MultinomialNB  # a simple machine learning algorithm
from sklearn.metrics import accuracy_score, classification_report  # measures how well our model is doing

# Sample dataset (replace with your own dataset)
data = {'Symptoms': ['headache fever', 'cough fever', 'nausea', 'headache', 'cough', 'fever'],
        'Drug': ['Aspirin', 'Cough Syrup', 'Anti-Nausea Medication', 'Aspirin', 'Cough Syrup', 'Antipyretic']}

df = pd.DataFrame(data)  # create a table-like structure for our data

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['Symptoms'], df['Drug'], test_size=0.2, random_state=42)

# Vectorize the symptoms using CountVectorizer
vectorizer = CountVectorizer()  # turn words into numbers
X_train_vectorized = vectorizer.fit_transform(X_train)  # convert symptoms in training set to numbers
X_test_vectorized = vectorizer.transform(X_test)  # convert symptoms in testing set to numbers

# Train a Naive Bayes classifier
classifier = MultinomialNB()  # create a simple learning model
classifier.fit(X_train_vectorized, y_train)  # train the model with our data

# Make predictions on the test set
predictions = classifier.predict(X_test_vectorized)  # use the trained model to predict on new data

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)  # see how well our model is performing
print(f'Accuracy: {accuracy:.2f}\n')

print('Classification Report:')
print(classification_report(y_test, predictions))  # more detailed performance metrics

# Function to get drug recommendation based on symptoms
def recommend_drug(symptoms):
    symptoms_vectorized = vectorizer.transform([symptoms])  # convert user symptoms to numbers
    prediction = classifier.predict(symptoms_vectorized)  # use the model to suggest a drug
    return prediction[0]

# Example usage
user_symptoms = 'headache'
recommended_drug = recommend_drug(user_symptoms)
print(f'Recommended Drug for Symptoms "{user_symptoms}": {recommended_drug}')
