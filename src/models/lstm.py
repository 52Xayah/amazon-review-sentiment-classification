
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, LSTM, Dropout, Dense
)


def create_lstm_model(
    vocab_size=20000,
    embedding_dim=128,
    max_len=100,
    num_classes=5
):
    """构建用于五分类情感分析的 LSTM 模型。"""
    model = Sequential([
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=max_len,
            mask_zero=True
        ),
        LSTM(64, dropout=0.2, recurrent_dropout=0.0),
        Dropout(0.3),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
