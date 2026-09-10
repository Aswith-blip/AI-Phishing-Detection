from ucimlrepo import fetch_ucirepo
import pandas as pd

print("Downloading PhiUSIIL Phishing URL Dataset...")

# Fetch dataset from UCI
dataset = fetch_ucirepo(id=967)

# Get features and targets
X = dataset.data.features
y = dataset.data.targets

print("Dataset downloaded successfully!")
print("Rows:", len(X))
print("Feature columns:", len(X.columns))

print("\nAvailable columns:")
print(X.columns.tolist())

print("\nTarget columns:")
print(y.columns.tolist())

# Combine URL and target data
data = pd.concat([X, y], axis=1)

print("\nFirst 5 rows:")
print(data.head())

# Save the original downloaded dataset
data.to_csv(
    "dataset/phiusiil_original.csv",
    index=False
)

print("\nOriginal dataset saved to:")
print("dataset/phiusiil_original.csv")