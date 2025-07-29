import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# Preprocess
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

# Build Neural Network
model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("\n🔹 Accuracy:", accuracy_score(y_test, y_pred))
print("\n🔹 Classification Report:\n", classification_report(y_test, y_pred, target_names=target_names))

# Save model and scaler
joblib.dump(model, "iris_nn_model.pkl")
print("\n✅ Model saved as iris_nn_model.pkl")

joblib.dump(scaler, "iris_scaler.pkl")
print("✅ Scaler saved as iris_scaler.pkl")
