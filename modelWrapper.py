import os
import numpy as np
import tensorflow as tf
from getModelInfo import ModelStateManager

class MLWrapper:
    def __init__(self, model=None, input_shape=(4,), num_classes=10):
        if model is None:
            self.model = self._build_model(input_shape, num_classes)
        else:
            self.model = model
            self._build_model(input_shape, num_classes)
        self.state_manager = ModelStateManager(self.model)
        self.current_epoch = 0
        self.backup_epoch = None


    def _build_model(self, input_shape, num_classes):
        # self.model = tf.keras.Sequential([
        #     tf.keras.layers.Input(shape=input_shape),
        #     tf.keras.layers.Dense(128, activation='relu'),
        #     tf.keras.layers.Dense(64, activation='relu'),
        #     tf.keras.layers.Dense(num_classes, activation='softmax')
        # ])
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return self.model

    def train(self, train_data, epochs=10, save_interval=2):
        # A simple training callback that saves state every save_interval epochs.
        class TrainingCallback(tf.keras.callbacks.Callback):
            def __init__(self, wrapper):
                self.wrapper = wrapper
                self.epoch_count = 0

            def on_epoch_end(self, epoch, logs=None):
                self.wrapper.current_epoch += 1
                self.epoch_count += 1
                if self.epoch_count % save_interval == 0:
                    self.wrapper.state_manager.save_state(self.wrapper.current_epoch, logs['loss'])
        self.model.fit(
            train_data[0], train_data[1],
            initial_epoch=self.current_epoch,
            epochs=self.current_epoch + epochs,
            callbacks=[TrainingCallback(self)]
        )

    def modify_epoch(self, layer_idx, new_weights, source_epoch, train_data=None, replay_training=True):
        """
        Modify the model from a specific epoch (e.g., epoch 4):
         - Load the state from the given epoch.
         - Save it as a backup (for rollback).
         - Apply the weight modifications.
         - Reset the current epoch to the modified epoch.
         - Optionally, resume training from that point.
        """
        # Load and back up state from the chosen epoch.
        self.state_manager.load_state(source_epoch)
        self.backup_epoch = source_epoch
        # Apply modification to the specified layer.
        self.model.layers[layer_idx].set_weights(new_weights)
        # Save the modified state at that epoch.
        self.state_manager.save_state(source_epoch, loss=0)
        # Reset the training counter so training resumes from the modified state.
        self.current_epoch = source_epoch
        print(f"Weights modified at epoch {source_epoch}")
        # Optionally, resume training until epoch 10.
        if replay_training and train_data is not None:
            additional_epochs = 10 - self.current_epoch
            if additional_epochs > 0:
                print(f"Resuming training for {additional_epochs} epochs from modified epoch {source_epoch}")
                self.train(train_data, epochs=additional_epochs)
        return source_epoch

    def rollback(self):
        """
        Rollback the modifications by reloading the backup state.
        """
        if self.backup_epoch is None:
            print("No backup state available to rollback.")
            return
        self.current_epoch = self.state_manager.load_state(self.backup_epoch)
        print(f"Rolled back to epoch {self.backup_epoch}")
        self.backup_epoch = None
