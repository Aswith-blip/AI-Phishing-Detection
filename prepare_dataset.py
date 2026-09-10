import pandas as pd

INPUT_FILE = "dataset/phiusiil_original.csv"
OUTPUT_FILE = "dataset/phishing_dataset.csv"

print("Loading original dataset...")

data = pd.read_csv(INPUT_FILE)

print("Original rows:", len(data))

# Keep only the URL and label columns
clean_data = data[["URL", "label"]].copy()

# Convert labels:
# UCI label 0 = Phishing
# UCI label 1 = Legitimate
clean_data["Label"] = clean_data["label"].map({
    0: "Phishing",
    1: "Legitimate"
})

# Remove old numeric label column
clean_data = clean_data[["URL", "Label"]]

# Remove missing URLs
clean_data = clean_data.dropna(subset=["URL"])

# Remove duplicate URLs
clean_data = clean_data.drop_duplicates(subset=["URL"])

# Reset index
clean_data = clean_data.reset_index(drop=True)

# Save final dataset
clean_data.to_csv(OUTPUT_FILE, index=False)

print("\nDataset prepared successfully!")

print("Total URLs:", len(clean_data))

print("\nClass distribution:")
print(clean_data["Label"].value_counts())

print("\nSaved to:")
print(OUTPUT_FILE)

print("\nFirst 5 rows:")
print(clean_data.head())
