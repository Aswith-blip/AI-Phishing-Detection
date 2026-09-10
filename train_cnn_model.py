import os
import time
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Embedding,
    Conv1D,
    GlobalMaxPooling1D,
    Dense,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)


print("=" * 75)
print("CHARACTER-LEVEL CNN PHISHING DETECTION MODEL")
print("=" * 75)

# =========================================================
# SETTINGS
# =========================================================

MAX_VOCAB_SIZE = 150
MAX_URL_LENGTH = 200

EMBEDDING_DIM = 32

EPOCHS = 15
BATCH_SIZE = 512

RANDOM_STATE = 42


# =========================================================
# 1. LOAD DATASET
# =========================================================

print("\n[1/7] Loading dataset...")

data = pd.read_csv(
    "dataset/phishing_dataset.csv"
)

data = data.dropna(
    subset=["URL", "Label"]
)

urls = data["URL"].astype(str).values

labels = np.array([
    1 if label == "Phishing" else 0
    for label in data["Label"]
])

print("Total URLs:", len(urls))

print("\nClass distribution:")
print(data["Label"].value_counts())


# =========================================================
# 2. SPLIT DATASET
# =========================================================

print("\n[2/7] Creating train/validation/test split...")

indices = np.arange(len(urls))

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=labels
)

train_idx, validation_idx = train_test_split(
    train_idx,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=labels[train_idx]
)

train_urls = urls[train_idx]
validation_urls = urls[validation_idx]
test_urls = urls[test_idx]

y_train = labels[train_idx]
y_validation = labels[validation_idx]
y_test = labels[test_idx]

print("Training   :", len(train_urls))
print("Validation :", len(validation_urls))
print("Testing    :", len(test_urls))


# =========================================================
# 3. CHARACTER TOKENIZATION
# =========================================================

print("\n[3/7] Creating character tokenizer...")

tokenizer = Tokenizer(
    num_words=MAX_VOCAB_SIZE,
    char_level=True,
    lower=False,
    filters=""
)

tokenizer.fit_on_texts(
    train_urls
)

print(
    "Character vocabulary:",
    len(tokenizer.word_index)
)

print(
    "Maximum URL length:",
    MAX_URL_LENGTH
)


# =========================================================
# CONVERT URLs TO SEQUENCES
# =========================================================

print("\nConverting URLs to character sequences...")

X_train = tokenizer.texts_to_sequences(
    train_urls
)

X_validation = tokenizer.texts_to_sequences(
    validation_urls
)

X_test = tokenizer.texts_to_sequences(
    test_urls
)


X_train = pad_sequences(
    X_train,
    maxlen=MAX_URL_LENGTH,
    padding="post",
    truncating="post"
)

X_validation = pad_sequences(
    X_validation,
    maxlen=MAX_URL_LENGTH,
    padding="post",
    truncating="post"
)

X_test = pad_sequences(
    X_test,
    maxlen=MAX_URL_LENGTH,
    padding="post",
    truncating="post"
)


print(
    "Training shape:",
    X_train.shape
)

print(
    "Validation shape:",
    X_validation.shape
)

print(
    "Testing shape:",
    X_test.shape
)


# =========================================================
# 4. BUILD CNN MODEL
# =========================================================

print("\n[4/7] Building CNN model...")

model = Sequential([
    
    Embedding(
        input_dim=MAX_VOCAB_SIZE,
        output_dim=EMBEDDING_DIM,
        input_length=MAX_URL_LENGTH
    ),

    Conv1D(
        filters=128,
        kernel_size=3,
        activation="relu",
        padding="same"
    ),

    BatchNormalization(),

    Conv1D(
        filters=128,
        kernel_size=5,
        activation="relu",
        padding="same"
    ),

    BatchNormalization(),

    GlobalMaxPooling1D(),

    Dense(
        128,
        activation="relu"
    ),

    Dropout(
        0.35
    ),

    Dense(
        64,
        activation="relu"
    ),

    Dropout(
        0.25
    ),

    Dense(
        1,
        activation="sigmoid"
    )
])


model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(
            name="auc"
        )
    ]
)


model.summary()


# =========================================================
# 5. TRAIN CNN
# =========================================================

print("\n[5/7] Training CNN...")

early_stopping = EarlyStopping(
    monitor="val_auc",
    mode="max",
    patience=3,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=0.00001
)


start = time.time()

history = model.fit(
    X_train,
    y_train,

    validation_data=(
        X_validation,
        y_validation
    ),

    epochs=EPOCHS,
    batch_size=BATCH_SIZE,

    callbacks=[
        early_stopping,
        reduce_lr
    ],

    verbose=1
)

training_time = time.time() - start

print(
    f"\nTraining time: "
    f"{training_time:.2f} seconds"
)


# =========================================================
# 6. EVALUATE MODEL
# =========================================================

print("\n[6/7] Evaluating CNN...")

probabilities = model.predict(
    X_test,
    batch_size=BATCH_SIZE,
    verbose=1
).ravel()

predictions = (
    probabilities >= 0.5
).astype(int)


accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    probabilities
)

pr_auc = average_precision_score(
    y_test,
    probabilities
)


print("\n" + "=" * 65)
print("CNN MODEL RESULTS")
print("=" * 65)

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1       : {f1:.4f}"
)

print(
    f"ROC-AUC  : {roc_auc:.4f}"
)

print(
    f"PR-AUC   : {pr_auc:.4f}"
)

print(
    f"Training Time: {training_time:.2f} seconds"
)


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        predictions
    )
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "Legitimate",
            "Phishing"
        ],
        zero_division=0
    )
)


# =========================================================
# 7. SAVE MODEL
# =========================================================

print("\n[7/7] Saving CNN model...")

os.makedirs(
    "model",
    exist_ok=True
)


model.save(
    "model/phishing_cnn.keras"
)


with open(
    "model/cnn_tokenizer.pkl",
    "wb"
) as file:

    pickle.dump(
        tokenizer,
        file
    )


cnn_info = {
    "model_name": "Character-Level CNN",
    "dataset_size": len(data),
    "vocabulary_size": len(
        tokenizer.word_index
    ),
    "max_url_length": MAX_URL_LENGTH,
    "embedding_dimension": EMBEDDING_DIM,
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "roc_auc": roc_auc,
    "pr_auc": pr_auc,
    "training_samples": len(train_urls),
    "validation_samples": len(validation_urls),
    "testing_samples": len(test_urls),
    "training_time": training_time
}


with open(
    "model/cnn_info.pkl",
    "wb"
) as file:

    pickle.dump(
        cnn_info,
        file
    )


print("\nSaved:")
print("model/phishing_cnn.keras")
print("model/cnn_tokenizer.pkl")
print("model/cnn_info.pkl")


print("\n" + "=" * 75)
print("CHARACTER-LEVEL CNN TRAINING COMPLETED")
print("=" * 75)