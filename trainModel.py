import tensorflow as tf
from modelWrapper import MLWrapper  # Assumes your MLWrapper and ModelStateManager are defined here

if __name__ == "__main__":
    tf.keras.backend.clear_session()

    # Initialize the ML system
    ml_system = MLWrapper()

    # Load MNIST data and pre-process it
    (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype('float32') / 255

    # 1. Initial Training: Train on 10 epochs on the 'current' branch.
    print("=== Initial Training on 'current' branch (10 epochs) ===")
    ml_system.train((x_train, y_train), epochs=10, save_interval=2)
    # This will save states (e.g., at epochs 2, 4, 6, 8, and 10) in model_state/current/

    # 2. Modify the model at a chosen epoch (e.g., epoch 4).
    print("\n=== Modifying from epoch 4 ===")
    layer_idx = 1  # For example, modify weights of the second layer.
    original_weights = ml_system.model.layers[layer_idx].get_weights()
    # Example modification: scale weights by 1.1.
    modified_weights = [w * 1.1 for w in original_weights]
    ml_system.modify_epoch(layer_idx, modified_weights, source_epoch=4, train_data=(x_train, y_train))

    # 3. Rollback the modifications to revert to the state from epoch 4 (unmodified).
    print("\n=== Rolling back modifications ===")
    ml_system.rollback()
