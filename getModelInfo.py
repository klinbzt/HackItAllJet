import os
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
import shutil

# Suppression settings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

class ModelStateManager:
    def __init__(self, model, base_dir="model_state"):
        self.model = model
        self.base_dir = Path(base_dir)
        self.current_version = "current"
        # We keep a simple list of state files saved in the current branch.
        self.history = []  
        (self.base_dir / self.current_version).mkdir(parents=True, exist_ok=True)
        
    def _init_structure(self):
        """Initialize directory structure and version tree"""
        main_dir = self.base_dir / "current"
        main_dir.mkdir(parents=True, exist_ok=True)
        self.version_tree = {
            "current": {
                "path": main_dir,
                "parent": None,
                "children": [],
                "history": []
            }
        }

    def _create_version_metadata(self, version_name, parent_name):
        """Create metadata for new version"""
        version_dir = self.base_dir / parent_name / version_name
        version_dir.mkdir(parents=True, exist_ok=True)
        
        self.version_tree[version_name] = {
            "path": version_dir,
            "parent": parent_name,
            "children": [],
            "history": []
        }
        self.version_tree[parent_name]["children"].append(version_name)

    def create_branch(self, branch_name, source_epoch=None):
        """
        Create new branch from the current version.
        Copies all state files from the parent's history that have an epoch <= source_epoch.
        """
        if branch_name in self.version_tree:
            raise ValueError("Branch name already exists")
            
        parent = self.current_version
        self._create_version_metadata(branch_name, parent)
        
        parent_history = self.version_tree[parent]["history"]
        if source_epoch is not None:
            copied_files = []
            # Sort parent's history based on epoch number
            for state_file in sorted(parent_history, key=lambda f: int(f.stem.split('_')[1])):
                epoch_num = int(state_file.stem.split('_')[1])
                if epoch_num <= source_epoch:
                    new_file = self.version_tree[branch_name]["path"] / state_file.name
                    shutil.copy(state_file, new_file)
                    copied_files.append(new_file)
            if not copied_files:
                raise FileNotFoundError(f"No state for epoch {source_epoch} in '{parent}'")
            self.version_tree[branch_name]["history"] = copied_files
        else:
            if not parent_history:
                raise FileNotFoundError(f"No states in parent '{parent}'")
            source_file = parent_history[-1]
            new_file = self.version_tree[branch_name]["path"] / source_file.name
            shutil.copy(source_file, new_file)
            self.version_tree[branch_name]["history"].append(new_file)

        self.current_version = branch_name
        print(f"Created branch '{branch_name}' under '{parent}' from epoch {source_epoch}")
        return branch_name

    def save_state(self, epoch, loss):
        version_dir = self.base_dir / self.current_version
        filename = version_dir / f"epoch_{epoch}.json"
        state = {
            'epoch': epoch,
            'weights': [layer.get_weights() for layer in self.model.layers],
            'optimizer': tf.keras.optimizers.serialize(self.model.optimizer),
            'loss': float(loss)
        }
        with open(filename, 'w') as f:
            json.dump(state, f, default=self._numpy_converter)
        self.history.append(filename)
        print(f"Saved state to {filename}")
        return filename

    def load_state(self, epoch):
        version_dir = self.base_dir / self.current_version
        state_file = version_dir / f"epoch_{epoch}.json"
        if not state_file.exists():
            raise FileNotFoundError(f"No state for epoch {epoch} in '{self.current_version}'")
        with open(state_file, 'r') as f:
            state = json.load(f)
        # Restore model weights
        for layer, weights in zip(self.model.layers, state['weights']):
            layer.set_weights([np.array(w) for w in weights])
        # Restore optimizer
        self.model.optimizer = tf.keras.optimizers.deserialize(state['optimizer'])
        print(f"Loaded state from {state_file}")
        return state['epoch']

    def rollback(self):
        """
        Delete the current branch (if not the root) and revert to the parent's state.
        """
        current_info = self.version_tree[self.current_version]
        if not current_info["parent"]:
            raise ValueError("Cannot rollback root version")
            
        parent_version = current_info["parent"]
        print(f"Rolling back from '{self.current_version}' to '{parent_version}'")
        
        # Delete current version directory and remove from version tree
        shutil.rmtree(current_info["path"])
        del self.version_tree[self.current_version]
        self.version_tree[parent_version]["children"].remove(self.current_version)
        
        self.current_version = parent_version
        return self.load_state()

    def _numpy_converter(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        return obj
