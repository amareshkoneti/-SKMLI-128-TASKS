import pandas as pd

# Loading the dataset
data = pd.read_csv('dataset.csv')  
print(data.head())

# Checking for missing values
print(data.isnull().sum())

# Droping rows with missing values 
data = data.dropna()
from sklearn.preprocessing import LabelEncoder

# Add 'None' to the symptoms before encoding
symptom_columns = [f'Symptom_{i}' for i in range(1, 18)]

# Creating a unique set of symptoms and add 'None'
unique_symptoms = set(data[symptom_columns].values.ravel())
unique_symptoms.add('None')

symptom_encoder = LabelEncoder()
symptom_encoder.fit(list(unique_symptoms))

# Fill missing symptom columns with 'None'
data[symptom_columns] = data[symptom_columns].fillna('None')

# Encode symptoms
for col in symptom_columns:
    data[col] = symptom_encoder.transform(data[col])

# Encode the target variable (Disease)
disease_encoder = LabelEncoder()
data['Disease'] = disease_encoder.fit_transform(data['Disease'])

# Split the data into features and target
X = data[symptom_columns]
y = data['Disease']

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

def predict_disease(symptoms):
    # Encode the input symptoms
    symptoms_encoded = [symptom_encoder.transform([symptom])[0] if symptom in symptom_encoder.classes_ else symptom_encoder.transform(['None'])[0] for symptom in symptoms]
    symptoms_encoded += [symptom_encoder.transform(['None'])[0]] * (17 - len(symptoms_encoded))  # Fill the rest with 'None'
    
    # Create a DataFrame for the input
    input_data = pd.DataFrame([symptoms_encoded], columns=symptom_columns)
    
    # Predict the disease
    prediction = model.predict(input_data)
    
    # Decode the prediction
    return disease_encoder.inverse_transform(prediction)[0]

# Example usage
new_symptoms = ['cough', 'fever', 'headache']  # Replace with actual symptoms
predicted_disease = predict_disease(new_symptoms)
print(f'Predicted Disease: {predicted_disease}')

