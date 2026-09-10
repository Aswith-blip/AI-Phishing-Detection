import os
import time
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Embedding,
    Dense,
    Dropout,
    LayerNormalization,
    GlobalAveragePooling1D,
    MultiHeadAttention
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)

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
print("TRANSFORMER URL PHISHING DETECTION MODEL")
print("=" * 75)


# =========================================================
# SETTINGS
# =========================================================

MAX_VOCAB_SIZE = 150
MAX_URL_LENGTH = 200

EMBEDDING_DIM = 64

NUM_HEADS = 4
TRANSFORMER_DIM = 128

EPOCHS = 12
BATCH_SIZE = 512

RANDOM_STATE = 42


# =========================================================
# 1. LOAD DATASET
# =========================================================

print("\n[1/8] Loading dataset...")

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
# 2. SPLIT DATA
# =========================================================

print("\n[2/8] Creating train/validation/test split...")

indices = np.arange(
    len(urls)
)

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

print("\n[3/8] Creating character tokenizer...")

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

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
    "Vocabulary size:",
    len(tokenizer.word_index)
)


# =========================================================
# CONVERT URLS TO SEQUENCES
# =========================================================

print("\nConverting URLs to sequences...")

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
# 4. POSITIONAL EMBEDDING
# =========================================================

class PositionalEmbedding(
    tf.keras.layers.Layer
):

    def __init__(
        self,
        max_length,
        vocab_size,
        embedding_dim
    ):

        super().__init__()

        self.token_embedding = Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim
        )

        self.position_embedding = Embedding(
            input_dim=max_length,
            output_dim=embedding_dim
        )

        self.max_length = max_length

    def call(self, inputs):

        length = tf.shape(inputs)[-1]

        positions = tf.range(
            start=0,
            limit=length,
            delta=1
        )

        embedded_tokens = (
            self.token_embedding(inputs)
        )

        embedded_positions = (
            self.position_embedding(
                positions
            )
        )

        return (
            embedded_tokens +
            embedded_positions
        )


# =========================================================
# 5. TRANSFORMER ENCODER
# =========================================================

class TransformerEncoder(
    tf.keras.layers.Layer
):

    def __init__(
        self,
        embedding_dim,
        num_heads,
        transformer_dim,
        dropout=0.2
    ):

        super().__init__()

        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embedding_dim
        )

        self.dense_projection = tf.keras.Sequential([
            Dense(
                transformer_dim,
                activation="relu"
            ),
            Dense(
                embedding_dim
            )
        ])

        self.layer_norm_1 = (
            LayerNormalization(
                epsilon=1e-6
            )
        )

        self.layer_norm_2 = (
            LayerNormalization(
                epsilon=1e-6
            )
        )

        self.dropout_1 = Dropout(
            dropout
        )

        self.dropout_2 = Dropout(
            dropout
        )

    def call(
        self,
        inputs,
        training=False
    ):

        attention_output = (
            self.attention(
                inputs,
                inputs,
                training=training
            )
        )

        attention_output = (
            self.dropout_1(
                attention_output,
                training=training
            )
        )

        out_1 = self.layer_norm_1(
            inputs +
            attention_output
        )

        projection = (
            self.dense_projection(
                out_1
            )
        )

        projection = (
            self.dropout_2(
                projection,
                training=training
            )
        )

        return self.layer_norm_2(
            out_1 +
            projection
        )


# =========================================================
# BUILD MODEL
# =========================================================

print("\n[4/8] Building Transformer model...")

inputs = Input(
    shape=(MAX_URL_LENGTH,),
    dtype=tf.int32,
    name="url_characters"
)

x = PositionalEmbedding(
    MAX_URL_LENGTH,
    MAX_VOCAB_SIZE,
    EMBEDDING_DIM
)(inputs)

x = TransformerEncoder(
    EMBEDDING_DIM,
    NUM_HEADS,
    TRANSFORMER_DIM,
    dropout=0.2
)(x)

x = GlobalAveragePooling1D()(x)

x = Dense(
    128,
    activation="relu"
)(x)

x = Dropout(
    0.30
)(x)

x = Dense(
    64,
    activation="relu"
)(x)

x = Dropout(
    0.20
)(x)

outputs = Dense(
    1,
    activation="sigmoid",
    name="phishing_probability"
)(x)

model = Model(
    inputs=inputs,
    outputs=outputs
)


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
# 6. TRAIN
# =========================================================

print("\n[5/8] Training Transformer...")

early_stopping = EarlyStopping(
    monitor="val_auc",
    mode="max",
    patience=2,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=1,
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

training_time = (
    time.time() - start
)

print(
    f"\nTraining time: "
    f"{training_time:.2f} seconds"
)


# =========================================================
# 7. EVALUATION
# =========================================================

print("\n[6/8] Evaluating Transformer...")

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
print("TRANSFORMER MODEL RESULTS")
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
    f"Training Time: "
    f"{training_time:.2f} seconds"
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
# 8. SAVE
# =========================================================

print("\n[7/8] Saving Transformer model...")

os.makedirs(
    "model",
    exist_ok=True
)

model.save(
    "model/phishing_transformer.keras"
)


with open(
    "model/transformer_tokenizer.pkl",
    "wb"
) as file:

    pickle.dump(
        tokenizer,
        file
    )


transformer_info = {
    "model_name": "Character-Level Transformer",
    "dataset_size": len(data),
    "vocabulary_size": len(
        tokenizer.word_index
    ),
    "max_url_length": MAX_URL_LENGTH,
    "embedding_dimension": EMBEDDING_DIM,
    "attention_heads": NUM_HEADS,
    "transformer_dimension": TRANSFORMER_DIM,
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
    "model/transformer_info.pkl",
    "wb"
) as file:

    pickle.dump(
        transformer_info,
        file
    )


print("\nSaved:")
print(
    "model/phishing_transformer.keras"
)

print(
    "model/transformer_tokenizer.pkl"
)

print(
    "model/transformer_info.pkl"
)


print("\n" + "=" * 75)
print("TRANSFORMER TRAINING COMPLETED")
print("=" * 75)