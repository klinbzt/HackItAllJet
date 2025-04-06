import os
import json
import shutil
from pathlib import Path
import numpy as np
import tensorflow as tf
from dash import html, dcc, Output, Input, State, ctx, dash_table
import dash
import dash_cytoscape as cyto
from dash.exceptions import PreventUpdate
import copy

# ------------------ ModelStateManager ------------------
# This class saves and loads model states to/from disk.
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

class ModelStateManager:
    def __init__(self, model, base_dir="model_state"):
        self.model = model
        self.base_dir = Path(base_dir)
        self.current_version = "current"
        self.history = []  # List of state file paths
        (self.base_dir / self.current_version).mkdir(parents=True, exist_ok=True)
        self._init_structure()
        
    def _init_structure(self):
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
        if branch_name in self.version_tree:
            raise ValueError("Branch name already exists")
        parent = self.current_version
        self._create_version_metadata(branch_name, parent)
        parent_history = self.version_tree[parent]["history"]
        if source_epoch is not None:
            copied_files = []
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
        for layer, weights in zip(self.model.layers, state['weights']):
            layer.set_weights([np.array(w) for w in weights])
        self.model.optimizer = tf.keras.optimizers.deserialize(state['optimizer'])
        print(f"Loaded state from {state_file}")
        return state['epoch']

    def rollback(self):
        current_info = self.version_tree[self.current_version]
        if not current_info["parent"]:
            raise ValueError("Cannot rollback root version")
        parent_version = current_info["parent"]
        print(f"Rolling back from '{self.current_version}' to '{parent_version}'")
        shutil.rmtree(current_info["path"])
        del self.version_tree[self.current_version]
        self.version_tree[parent_version]["children"].remove(self.current_version)
        self.current_version = parent_version
        # A proper rollback would load a particular state; adjust as needed.
        return

    def _numpy_converter(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        return obj

# ------------------ FullNNVisualizer ------------------
class FullNNVisualizer:
    def __init__(self, model):
        self.model = model
        self.elements = []
        self.forward_steps = []
        self.backward_steps = []
        self.layer_neurons = []  # Track neuron IDs by layer
        self.layer_info = []     # Track layer names and units
        self._build_elements_and_steps()
        self._extract_weights()

    def _extract_weights(self):
        """Extract weights from the model for visualization."""
        self.weights = {}
        for layer in self.model.layers:
            if hasattr(layer, 'kernel'):
                weights = layer.get_weights()[0]  # Weight matrix
                biases = layer.get_weights()[1]   # Bias vector
                self.weights[layer.name] = {
                    'weights': weights,
                    'biases': biases
                }

    def _build_elements_and_steps(self):
        spacing_x = 250
        spacing_y = 70
        x = 100

        all_layers_units = []
        input_shape = self.model.input_shape
        input_units = input_shape[1] if input_shape else 1
        all_layers_units.append(("input_layer", input_units, None))

        for layer in self.model.layers:
            try:
                units = layer.units
                all_layers_units.append((layer.name, units, layer))
            except AttributeError:
                continue

        prev_layer_neurons = []
        for layer_idx, (layer_id, units, layer) in enumerate(all_layers_units):
            y_offset = 300 - (units * spacing_y) // 2
            current_layer_neurons = []

            for i in range(units):
                node_id = f"{layer_id}_{i}"
                current_layer_neurons.append(node_id)
                self.elements.append({
                    'data': {
                        'id': node_id,
                        'label': f"{i}",
                        'layer': layer_idx, 
                        'layer_name': layer_id,
                        'neuron_idx': i
                    },
                    'position': {'x': x, 'y': y_offset + i * spacing_y},
                    'classes': ''
                })

            for from_node in prev_layer_neurons:
                for to_node in current_layer_neurons:
                    self.elements.append({
                        'data': {
                            'source': from_node,
                            'target': to_node,
                            'id': f"{from_node}->{to_node}"
                        },
                        'classes': ''
                    })

            if prev_layer_neurons:
                self.forward_steps.extend([
                    (prev_layer_neurons, [current_neuron])
                    for current_neuron in current_layer_neurons
                ])
                self.backward_steps.extend([
                    ([prev_neuron], current_layer_neurons)
                    for prev_neuron in prev_layer_neurons
                ])

            self.layer_neurons.append(current_layer_neurons)
            self.layer_info.append((layer_id, units, layer))
            prev_layer_neurons = current_layer_neurons
            x += spacing_x

    def get_elements(self):
        return self.elements

    def get_stylesheet(self):
        return [
            {'selector': 'node', 'style': {
                'content': 'data(label)',
                'text-valign': 'center',
                'text-halign': 'center',
                'width': 40,
                'height': 40,
                'background-color': '#0074D9',
                'color': 'black',
                'font-size': '12px',
                'shape': 'ellipse'
            }},
            {'selector': 'edge', 'style': {
                'width': 1,
                'line-color': '#ccc',
            }},
            {'selector': '.active-node', 'style': {
                'background-color': '#FF4136',
                'transition-property': 'background-color',
                'transition-duration': '0.3s',
            }},
            {'selector': '.active-edge', 'style': {
                'line-color': '#FF851B',
                'width': 3,
                'transition-property': 'line-color, width',
                'transition-duration': '0.3s',
            }},
            {'selector': '.previous-node', 'style': {
                'background-color': '#2ECC40',
                'transition-property': 'background-color',
                'transition-duration': '0.3s',
            }},
            {'selector': '.receiving-node', 'style': {
                'background-color': '#FFDC00',
                'transition-property': 'background-color',
                'transition-duration': '0.3s',
            }},
            {'selector': '.selected-node', 'style': {
                'background-color': '#B10DC9',
                'border-width': '2px',
                'border-color': 'white'
            }},
            {'selector': '.weight-node', 'style': {
                'width': 40,
                'height': 12,
                'background-color': '#FFDC00',
                'color': 'black',
                'font-size': '10px',
                'shape': 'rectangle',
                'text-halign': 'center',
                'text-valign': 'center',
                'border-width': '1px',
                'border-color': '#888',
            }},
        ]

    def get_weights_for_neuron(self, layer_name, neuron_idx):
        if layer_name == 'input_layer':
            return None
        layer_weights = self.weights.get(layer_name)
        if not layer_weights:
            return None
        neuron_weights = layer_weights['weights'][:, neuron_idx]
        neuron_bias = layer_weights['biases'][neuron_idx]
        return {
            'weights': neuron_weights,
            'bias': neuron_bias
        }

# ------------------ DashFullNNApp ------------------
class DashFullNNApp:
    def __init__(self, model):
        self.current_epoch = 0
        self.model = model
        self.visualizer = FullNNVisualizer(model)
        self.elements = self.visualizer.get_elements()
        self.forward_steps = self.visualizer.forward_steps
        self.backward_steps = self.visualizer.backward_steps
        self.current_step = 0
        self.mode = None  # 'forward' or 'backward'
        self.is_paused = False
        self.selected_neuron = None
        self.backup_weights = None

        # Instantiate the state manager.
        self.state_manager = ModelStateManager(self.model)

        self.app = dash.Dash(__name__)
        self.app.title = "NN Propagation Visualizer"
        self._build_layout()
        self._register_callbacks()

    def _get_old_weight(self, layer_name, neuron_idx, weight_idx, is_bias=False):
        """
        Retrieve the previous weight value for the given neuron from the last saved state.
        """
        if not self.state_manager.history:
            return None
        last_file = self.state_manager.history[-1]
        try:
            with open(last_file, 'r') as f:
                state = json.load(f)
        except Exception:
            return None
        # Determine the corresponding layer index by matching layer names in visualizer.layer_info.
        layer_index = None
        for idx, (lname, units, layer) in enumerate(self.visualizer.layer_info):
            if lname == layer_name:
                layer_index = idx
                break
        if layer_index is None:
            return None
        try:
            if is_bias:
                return state['weights'][layer_index][1][neuron_idx]
            else:
                return state['weights'][layer_index][0][weight_idx][neuron_idx]
        except (IndexError, KeyError):
            return None

    def _update_weight_nodes(self, elements):
        base_elements = [el for el in elements if not el.get('data', {}).get('is_weight_node', False)]
        new_elements = base_elements.copy()
        for el in base_elements:
            if el.get('data', {}).get('layer_name') and el['data']['layer_name'] != 'input_layer':
                neuron_id = el['data']['id']
                pos = el['position']
                weight_info = self.visualizer.get_weights_for_neuron(el['data']['layer_name'],
                                                                      el['data']['neuron_idx'])
                if weight_info is not None:
                    num_weights = len(weight_info['weights'])
                    total_items = num_weights + 1  # include bias
                    start_y = pos['y'] - (total_items * 12) / 2
                    offset_y = 12
                    offset_x = 50
                    # Bias node: get current and old bias.
                    current_bias = weight_info['bias']
                    old_bias = self._get_old_weight(el['data']['layer_name'],
                                                    el['data']['neuron_idx'],
                                                    weight_idx=0,
                                                    is_bias=True)
                    bias_label = f"B: {current_bias:.2f}"
                    if old_bias is not None:
                        bias_label += f"\n(old: {old_bias:.2f})"
                    bias_node = {
                        'data': {
                            'id': f"w_{neuron_id}_bias",
                            'label': bias_label,
                            'is_weight_node': True
                        },
                        'position': {'x': pos['x'] + offset_x, 'y': start_y},
                        'classes': 'weight-node'
                    }
                    new_elements.append(bias_node)
                    # Weight nodes.
                    for i, w in enumerate(weight_info['weights']):
                        old_w = self._get_old_weight(el['data']['layer_name'],
                                                     el['data']['neuron_idx'],
                                                     weight_idx=i,
                                                     is_bias=False)
                        weight_label = f"{w:.2f}"
                        if old_w is not None:
                            weight_label += f"\n(old: {old_w:.2f})"
                        weight_node = {
                            'data': {
                                'id': f"w_{neuron_id}_{i}",
                                'label': weight_label,
                                'is_weight_node': True
                            },
                            'position': {'x': pos['x'] + offset_x, 'y': start_y + (i+1)*offset_y},
                            'classes': 'weight-node'
                        }
                        new_elements.append(weight_node)
        return new_elements

    def _build_layout(self):
        self.app.layout = html.Div([
            html.H2("Neural Network Propagation Visualizer"),
            html.Div([
                html.Button("\u25B6 Forward Propagation", id="forward-btn", n_clicks=0),
                html.Button("\u21BB Backpropagation", id="backward-btn", n_clicks=0),
                html.Button("\u23F8 Pause", id="pause-btn", n_clicks=0),
                dcc.Interval(id='animation-interval', interval=1000, disabled=True),
                html.Div(id='step-info', style={'marginLeft': '20px', 'fontWeight': 'bold'}),
            ], style={'marginBottom': '20px', 'display': 'flex', 'alignItems': 'center'}),
            
            html.Div([
                html.H3("Training Controls", style={'marginTop': '30px'}),
                html.Div([
                    dcc.Input(id='epochs-input', type='number', value=1, min=1,
                              style={'width': '60px', 'marginRight': '10px'}),
                    html.Button('Train', id='train-btn', n_clicks=0),
                    html.Button('Rollback', id='rollback-btn', n_clicks=0,
                                style={'marginLeft': '20px'}),
                ], style={'marginBottom': '10px'}),
                html.Div(id='training-status'),
                html.H4("Weight Modification", style={'marginTop': '20px'}),
                html.Div([
                    dcc.Input(id='weight-input', type='text',
                              placeholder='e.g., 0.1, -0.2, 0.3',
                              style={'width': '200px', 'marginRight': '10px'}),
                    dcc.Input(id='bias-input', type='number', step=0.01,
                              placeholder='Bias',
                              style={'width': '80px', 'marginRight': '10px'}),
                    html.Button('Modify Weights', id='modify-weights-btn', n_clicks=0),
                ]),
                html.Div(id='modification-status', style={'marginTop': '10px'}),
                html.Div(id='editable-weights-table')
            ], style={
                'marginTop': '30px',
                'padding': '20px',
                'border': '1px solid #ddd',
                'borderRadius': '5px'
            }),
            
            cyto.Cytoscape(
                id='nn-cytoscape',
                elements=self.elements,
                layout={'name': 'preset'},
                style={'width': '100%', 'height': '700px'},
                stylesheet=self.visualizer.get_stylesheet(),
            ),
            html.Div(id='neuron-info'),
            dcc.Store(id='selected-neuron-store'),
            dcc.Store(id='training-progress', data={'training': False, 'epochs_remaining': 0}),
            dcc.Interval(id='training-interval', interval=1000, disabled=True)
        ])

    def _register_callbacks(self):
        @self.app.callback(
            Output('animation-interval', 'disabled'),
            Output('animation-interval', 'n_intervals'),
            Output('step-info', 'children'),
            Output('pause-btn', 'children'),
            Input('forward-btn', 'n_clicks'),
            Input('backward-btn', 'n_clicks'),
            Input('pause-btn', 'n_clicks'),
            Input('train-btn', 'n_clicks'),
            State('animation-interval', 'disabled'),
            prevent_initial_call=True
        )
        def control_animation(fwd_clicks, bwd_clicks, pause_clicks, train_clicks, is_interval_disabled):
            triggered_id = ctx.triggered_id
            if triggered_id == 'pause-btn':
                self.is_paused = not self.is_paused
                if self.is_paused:
                    return True, dash.no_update, "Animation paused", "\u25B6 Play"
                else:
                    return False, dash.no_update, "Animation resumed", "\u23F8 Pause"
            elif triggered_id == 'forward-btn':
                self.mode = 'forward'
                self.current_step = 0
                self.is_paused = False
                return False, 0, "Forward propagation started", "\u23F8 Pause"
            elif triggered_id == 'backward-btn':
                self.mode = 'backward'
                self.current_step = 0
                self.is_paused = False
                return False, 0, "Backpropagation started", "\u23F8 Pause"
            elif triggered_id == 'train-btn':
                self.mode = 'forward'
                self.current_step = 0
                self.is_paused = False
                return False, 0, "Training initiated; propagation will start after epoch", "\u23F8 Pause"
            raise PreventUpdate

        @self.app.callback(
            Output('nn-cytoscape', 'elements'),
            Output('step-info', 'children', allow_duplicate=True),
            Input('animation-interval', 'n_intervals'),
            State('nn-cytoscape', 'elements'),
            prevent_initial_call=True
        )
        def animate_step(n, current_elements):
            if self.is_paused:
                raise PreventUpdate
            steps = self.forward_steps if self.mode == 'forward' else self.backward_steps
            if self.current_step >= len(steps):
                updated = self._update_weight_nodes(current_elements)
                return self._clear_classes(updated), "Propagation complete!"
            elements = [copy.deepcopy(el) for el in current_elements]
            from_nodes, to_nodes = steps[self.current_step]
            for el in elements:
                el['classes'] = el.get('classes', '')
                if 'selected-node' not in el['classes']:
                    el['classes'] = ''
            for el in elements:
                el_id = el['data'].get('id')
                if el_id in from_nodes:
                    el['classes'] = 'previous-node'
                elif el_id in to_nodes:
                    el['classes'] = 'receiving-node'
                elif 'source' in el['data'] and el['data']['source'] in from_nodes and el['data']['target'] in to_nodes:
                    el['classes'] = 'active-edge'
            if self.mode == 'forward':
                cumulative = []
                count = 0
                for i in range(1, len(self.visualizer.layer_neurons)):
                    count += len(self.visualizer.layer_neurons[i])
                    cumulative.append(count)
                if (self.current_step + 1) in cumulative:
                    finished_layer = cumulative.index(self.current_step + 1) + 1
                    for el in elements:
                        if el['data'].get('layer') == finished_layer:
                            if 'blink' not in el['classes']:
                                el['classes'] += ' blink'
            self.current_step += 1
            updated_elements = self._update_weight_nodes(elements)
            return updated_elements, "Propagation in progress..."

        @self.app.callback(
            Output('nn-cytoscape', 'elements', allow_duplicate=True),
            Output('neuron-info', 'children'),
            Output('selected-neuron-store', 'data'),
            Output('editable-weights-table', 'children'),
            Input('nn-cytoscape', 'tapNode'),
            State('nn-cytoscape', 'elements'),
            State('selected-neuron-store', 'data'),
            prevent_initial_call=True
        )
        def select_neuron(node_data, current_elements, stored_data):
            if not node_data or (not self.is_paused and self.current_step != 0):
                raise PreventUpdate
            elements = [copy.deepcopy(el) for el in current_elements]
            for el in elements:
                el['classes'] = el.get('classes', '')
                if 'selected-node' in el['classes']:
                    el['classes'] = el['classes'].replace('selected-node', '').strip()
            node_id = node_data['data']['id']
            for el in elements:
                if el['data']['id'] == node_id:
                    el['classes'] = 'selected-node'
                    break
            layer_name = node_data['data']['layer_name']
            neuron_idx = node_data['data']['neuron_idx']
            selected = {'layer': layer_name, 'neuron': neuron_idx}
            weights_data = self.visualizer.get_weights_for_neuron(layer_name, neuron_idx)
            weights_table = self._create_weights_table(weights_data)
            info_content = html.Div([
                html.H3(f"Neuron Info - Layer: {layer_name}, Neuron: {neuron_idx}"),
                weights_table
            ])
            updated_elements = self._update_weight_nodes(elements)
            return updated_elements, info_content, selected, weights_table

        @self.app.callback(
            Output('nn-cytoscape', 'elements', allow_duplicate=True),
            Output('training-status', 'children', allow_duplicate=True),
            Input('rollback-btn', 'n_clicks'),
            State('nn-cytoscape', 'elements'),
            prevent_initial_call=True
        )
        def rollback_model(n_clicks, current_elements):
            if self.backup_weights is not None:
                self.model.set_weights(self.backup_weights)
                self.visualizer._extract_weights()
                elements = self._clear_classes(current_elements)
                return elements, "Rolled back to the backup weights."
            else:
                return dash.no_update, "No backup available for rollback."

        @self.app.callback(
            Output('training-progress', 'data', allow_duplicate=True),
            Output('training-status', 'children'),
            Input('train-btn', 'n_clicks'),
            State('epochs-input', 'value'),
            prevent_initial_call=True
        )
        def initiate_training(n_clicks, epochs):
            if not epochs or epochs < 1:
                return dash.no_update, "Please enter a valid number of epochs"
            return {'training': True, 'epochs_remaining': epochs}, f"Training initiated for {epochs} epochs..."    
        
        @self.app.callback(
            Output('training-interval', 'disabled'),
            Input('training-progress', 'data'),
            prevent_initial_call=True
        )
        def toggle_training_interval(progress_data):
            if progress_data and progress_data.get('training', False):
                return False
            return True

        @self.app.callback(
            Output('nn-cytoscape', 'elements', allow_duplicate=True),
            Output('training-progress', 'data'),
            Output('training-status', 'children', allow_duplicate=True),
            Input('training-interval', 'n_intervals'),
            State('training-progress', 'data'),
            State('nn-cytoscape', 'elements'),
            prevent_initial_call=True
        )
        def training_epoch(n_intervals, progress_data, current_elements):
            if not progress_data or not progress_data.get('training', False):
                raise PreventUpdate
            if progress_data['epochs_remaining'] <= 0:
                return current_elements, {'training': False, 'epochs_remaining': 0}, "Training complete."
            # Run one epoch of training.
            history = self.model.fit(
                self.x_train, self.y_train,
                initial_epoch=self.current_epoch,
                epochs=self.current_epoch + 1,
                verbose=0
            )
            # Save state (you might want to use the loss from history.history if needed).
            self.state_manager.save_state(self.current_epoch, loss=history.history.get('loss', [0])[0])
            self.current_epoch += 1
            self.backup_weights = self.model.get_weights()
            self.visualizer._extract_weights()
            updated_elements = self._update_weight_nodes(current_elements)
            progress_data['epochs_remaining'] -= 1
            status_msg = f"Completed epoch {self.current_epoch}. {progress_data['epochs_remaining']} epochs remaining."
            return updated_elements, progress_data, status_msg

    def _create_weights_table(self, weights_data):
        if not weights_data:
            return html.P("No weights available for input layer neurons")
        weights = weights_data['weights']
        bias = weights_data['bias']
        table_data = [{
            'Connection': f'Input {i}',
            'Weight': f'{weight:.4f}'
        } for i, weight in enumerate(weights)]
        table_data.append({'Connection': 'Bias', 'Weight': f'{bias:.4f}'})
        return dash_table.DataTable(
            id='weights-table',
            columns=[
                {'name': 'Connection', 'id': 'Connection', 'editable': False},
                {'name': 'Weight', 'id': 'Weight', 'editable': True}
            ],
            data=table_data,
            editable=True,
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '8px'},
            style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
            style_data_conditional=[
                {
                    'if': {'row_index': len(table_data)-1},
                    'fontWeight': 'bold',
                    'backgroundColor': 'rgb(240, 240, 240)'
                }
            ]
        )

    def _clear_classes(self, elements):
        for el in elements:
            el['classes'] = el.get('classes', '')
            if 'selected-node' not in el['classes']:
                el['classes'] = ''
        return elements

    def run(self):
        self.app.run(debug=True)

if __name__ == "__main__":
    tf.keras.backend.clear_session()
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(4,), name='input_layer'),
        tf.keras.layers.Dense(3, activation='relu', name='layer_1'),
        tf.keras.layers.Dense(4, activation='relu', name='layer_2'),
        tf.keras.layers.Dense(4, activation='relu', name='layer_3'),
        tf.keras.layers.Dense(4, activation='softmax', name='output_layer')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    x_train = np.random.rand(100, 4).astype('float32')
    y_train = np.random.randint(0, 4, size=(100,))
    app = DashFullNNApp(model)
    app.x_train = x_train
    app.y_train = y_train
    app.backup_weights = model.get_weights()
    app.run()
