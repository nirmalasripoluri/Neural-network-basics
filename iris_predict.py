import joblib
import numpy as np

# Load model and scaler
model = joblib.load("iris_nn_model.pkl")
scaler = joblib.load("iris_scaler.pkl")

# Get input from user
print("Enter flower measurements (in cm):")
sepal_length = float(input("Sepal length: "))
sepal_width = float(input("Sepal width: "))
petal_length = float(input("Petal length: "))
petal_width = float(input("Petal width: "))

# Create feature array and scale it
input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
input_scaled = scaler.transform(input_features)

# Make prediction
prediction = model.predict(input_scaled)[0]
target_names = ['setosa', 'versicolor', 'virginica']

# Output result
print(f"\n🌸 Predicted Iris Species: {target_names[prediction].capitalize()}")
