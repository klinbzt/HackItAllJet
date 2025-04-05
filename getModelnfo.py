import os
import json
import numpy as np
import tensorflow as tf
from pathlib import Path

# Stronger warning suppression
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

class ModelStateManager:
    def __init__(self, model=None, save_dir="model_states"):
        self.model = model
        self.save_dir = Path(save_dir).absolute()  # Use absolute path
        self.save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving states to: {self.save_dir}")

    def save_state(self, epoch=None, loss=None):
        """Save model state with better file validation"""
        if not isinstance(self.model, tf.keras.Sequential):
            raise ValueError("Only Sequential models are supported")

        # Generate safe filename
        if epoch is None:
            filename = "state_init.json"
        else:
            filename = f"state_{int(epoch)}.json"
        
        file_path = self.save_dir / filename

        try:
            state = {
                'model_type': 'Sequential',
                'model_config': self.model.get_config(),
                'weights': [layer.get_weights() for layer in self.model.layers],
                'training_info': {
                    'epoch': epoch,
                    'loss': float(loss) if loss else None
                }
            }

            with open(file_path, 'w') as f:
                json.dump(state, f, default=self._numpy_converter, indent=2)
            
            print(f"Saved state to: {file_path}")
            return file_path

        except Exception as e:
            print(f"Error saving state: {str(e)}")
            raise

    def load_state(self, file_path):
        """Load model state with better error handling"""
        try:
            path = Path(file_path).absolute()
            if not path.exists():
                available = [f.name for f in self.save_dir.glob("state_*.json")]
                raise FileNotFoundError(
                    f"State file {path} not found. Available states: {available}"
                )

            with open(path, 'r') as f:
                state = json.load(f)

            # Model reconstruction
            self.model = tf.keras.Sequential.from_config(state['model_config'])
            
            # Weight loading
            for idx, layer_weights in enumerate(state['weights']):
                if layer_weights:
                    self.model.layers[idx].set_weights(
                        [np.array(w) for w in layer_weights]
                    )
            
            print(f"Successfully loaded state from: {path}")
            return self

        except Exception as e:
            print(f"Error loading state: {str(e)}")
            raise

    def _numpy_converter(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        return obj

# Usage Example
if __name__ == "__main__":
    # Clean initialization
    tf.keras.backend.clear_session()
    
    # Model creation
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(784,), name='input_layer'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    # State manager with verification
    state_mgr = ModelStateManager(model)
    
    # Save initial state
    try:
        saved_path = state_mgr.save_state(epoch=0, loss=0.5)
    except Exception as e:
        print(f"Initial save failed: {e}")
        exit(1)
    
    # Verify loading
    try:
        state_mgr.load_state(saved_path)
        print("Load verification successful!")
        model.summary()
    except Exception as e:
        print(f"Load verification failed: {e}")
        exit(1)