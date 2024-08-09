import keras

from data_loading import load_csv
import numpy as np
import tensorflow as tf
from transformer.transformer_encoder_layer import build_transformer_model


def data_loading():
    X_train, y_train = load_csv("data_files/supervised_ttbar_train.csv")
    # reshape X_train into the appropriate shape for transformer input, by adding new axis
    X_train = X_train[:, :, np.newaxis]
    print("Done loading training data.")

    # Load testing data batches
    X_test, y_test = load_csv("data_files/50_track_ttbar_pt_12.csv")
    # Reshape testing data for transformer
    X_test = X_test[:, :, np.newaxis]
    print("Done loading testing data.")

    return X_train, y_train, X_test, y_test


def evaluate_results(predictions, event_flag):
    partition_idxs = np.where(event_flag == 1)[0]
    event_subs = np.split(predictions, partition_idxs)[1:]  # remember to ignore first subarr since its empty
    N_events = len(event_subs)
    correct_cnt = 0
    for arr in event_subs:
        predicted_hs_idx = np.argmax(arr)
        correct_cnt += int(predicted_hs_idx == 0)
    print(f"Model Accuracy: {correct_cnt/N_events}")

    return correct_cnt / N_events


if __name__ == "__main__":
    input_shape = (50, 1)  # sequence length of 50 and each token represented by a float (track pt)
    head_size = input_shape[1]
    num_heads = 4
    ff_dim = 3
    num_layers = 1


    # Load all data
    X_train, y_train, X_test, y_test = data_loading()

    # Create model
    print("*****\nBuilding and Testing Transformer model\n*****")
    model = build_transformer_model(input_shape, head_size, num_heads, ff_dim, num_layers,
                                    regularizer=None)
    model.compile(
        loss="binary_crossentropy",  # Change this based on your task
        optimizer=tf.keras.optimizers.Adam(learning_rate=2.5e-3),
        metrics=["accuracy"]  # Change this based on your task
    )

    model.summary()

    # Training
    model.fit(X_train, y_train, epochs=10, batch_size=256)

    # Inference
    predictions = model.predict(X_test)
    predictions = predictions.flatten()

    # Evaluate model
    evaluate_results(predictions, y_test)