from sklearn.datasets import load_iris
import pandas as pd

# Load the Iris dataset
iris = load_iris()

# Create DataFrame for better readability
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['target_name'] = df['target'].apply(lambda x: iris.target_names[x])

# Print dataset summary
print("🌸 Iris Dataset Summary 🌸")
print("\nFeature names:", iris.feature_names)
print("Target names:", iris.target_names)
print("\nShape of data:", iris.data.shape)

# Show sample data
print("\n🔹 First 5 rows of the dataset:")
print(df.head())

# Show target distribution
print("\n🔹 Target value counts:")
print(df['target_name'].value_counts())
