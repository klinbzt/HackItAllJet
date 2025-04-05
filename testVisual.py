import dash
from dash import html, dcc, Output, Input, State, ctx, dash_table
import dash_cytoscape as cyto
import tensorflow as tf
from dash.exceptions import PreventUpdate
import copy
import numpy as np

# Visualization helper class
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
        """Extract weights from the model for visualization"""
        self.weights = {}
        for layer in self.model.layers:
            if hasattr(layer, 'kernel'):
                weights = layer.get_weights()[0]  # Weights matrix
                biases = layer.get_weights()[1]   # Biases vector
                self.weights[layer.name] = {
                    'weights': weights,
                    'biases': biases
                }

    def _build_elements_and_steps(self):
        spacing_x = 250
        spacing_y = 70
        x = 100

        all_layers_units = []

        # Extract units from InputLayer manually
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
                    'classes': ''  # Ensure default classes exist
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
                # For forward steps, process neuron-by-neuron in the current layer
                self.forward_steps.extend([
                    (prev_layer_neurons, [current_neuron])
                    for current_neuron in current_layer_neurons
                ])
                # For backward steps, process neuron-by-neuron in the previous layer
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
                'width': 30,
                'height': 30,
                'background-color': '#0074D9',
                'color': 'white',
                'font-size': '10px',
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
        ]

    def get_weights_for_neuron(self, layer_name, neuron_idx):
        """Get weights for a specific neuron"""
        if layer_name == 'input_layer':
            return None  # Input layer has no weights
        
        layer_weights = self.weights.get(layer_name)
        if not layer_weights:
            return None
            
        # For Dense layers, weights are stored as (input_dim, units)
        neuron_weights = layer_weights['weights'][:, neuron_idx]
        neuron_bias = layer_weights['biases'][neuron_idx]
        
        return {
            'weights': neuron_weights,
            'bias': neuron_bias
        }

# Dash application class
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

        # For rollback, store backup weights
        self.backup_weights = None

        self.app = dash.Dash(__name__)
        self.app.title = "NN Propagation Visualizer"
        self._build_layout()
        self._register_callbacks()

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
            
            # Training Controls Section
            html.Div([
                html.H3("Training Controls", style={'marginTop': '30px'}),
                html.Div([
                    dcc.Input(
                        id='epochs-input',
                        type='number',
                        value=1,
                        min=1,
                        style={'width': '60px', 'marginRight': '10px'}
                    ),
                    html.Button('Train', id='train-btn', n_clicks=0),
                    html.Button('Rollback', id='rollback-btn', n_clicks=0, 
                            style={'marginLeft': '20px'}),
                ], style={'marginBottom': '10px'}),
                html.Div(id='training-status'),
                
                # Weight Modification Section
                html.H4("Weight Modification", style={'marginTop': '20px'}),
                html.Div([
                    # A manual entry option (kept for reference)
                    dcc.Input(
                        id='weight-input',
                        type='text',
                        placeholder='e.g., 0.1, -0.2, 0.3',
                        style={'width': '200px', 'marginRight': '10px'}
                    ),
                    dcc.Input(
                        id='bias-input',
                        type='number',
                        step=0.01,
                        placeholder='Bias',
                        style={'width': '80px', 'marginRight': '10px'}
                    ),
                    html.Button('Modify Weights', id='modify-weights-btn', n_clicks=0),
                ]),
                html.Div(id='modification-status', style={'marginTop': '10px'}),
                
                # Editable weights table (generated when a node is selected)
                html.Div(id='editable-weights-table')
            ], style={
                'marginTop': '30px',
                'padding': '20px',
                'border': '1px solid #ddd',
                'borderRadius': '5px'
            }),
            
            # Visualization Components
            cyto.Cytoscape(
                id='nn-cytoscape',
                elements=self.elements,
                layout={'name': 'preset'},
                style={'width': '100%', 'height': '700px'},
                stylesheet=self.visualizer.get_stylesheet(),
            ),
            html.Div(id='neuron-info'),
            dcc.Store(id='selected-neuron-store')
        ])

    def _create_weights_table(self, weights_data):
        """Create an editable table showing weights for a neuron"""
        if not weights_data:
            return html.P("No weights available for input layer neurons")
        
        weights = weights_data['weights']
        bias = weights_data['bias']
        
        table_data = [{
            'Connection': f'Input {i}',
            'Weight': f'{weight:.4f}'
        } for i, weight in enumerate(weights)]
        
        table_data.append({
            'Connection': 'Bias',
            'Weight': f'{bias:.4f}'
        })
        
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

    def _register_callbacks(self):
        # Animation control callback
        @self.app.callback(
            Output('animation-interval', 'disabled'),
            Output('animation-interval', 'n_intervals'),
            Output('step-info', 'children'),
            Output('pause-btn', 'children'),
            Input('forward-btn', 'n_clicks'),
            Input('backward-btn', 'n_clicks'),
            Input('pause-btn', 'n_clicks'),
            State('animation-interval', 'disabled'),
            prevent_initial_call=True
        )
        def control_animation(fwd_clicks, bwd_clicks, pause_clicks, is_interval_disabled):
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
                message = "Forward propagation started"
                return False, 0, message, "\u23F8 Pause"
            
            elif triggered_id == 'backward-btn':
                self.mode = 'backward'
                self.current_step = 0
                self.is_paused = False
                message = "Backpropagation started"
                return False, 0, message, "\u23F8 Pause"
            
            raise PreventUpdate

        # Animation step callback
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
                
            if self.mode == 'forward':
                steps = self.forward_steps
            elif self.mode == 'backward':
                steps = self.backward_steps
            else:
                raise PreventUpdate

            if self.current_step >= len(steps):
                return self._clear_classes(current_elements), "Propagation complete!"

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
                layer_idx = next(el['data']['layer'] for el in elements if el['data'].get('id') == to_nodes[0])
                message = f"Layer {layer_idx}: Neuron {to_nodes[0].split('_')[-1]} receiving inputs"
            else:
                layer_idx = next(el['data']['layer'] for el in elements if el['data'].get('id') == from_nodes[0])
                message = f"Layer {layer_idx}: Neuron {from_nodes[0].split('_')[-1]} sending gradients"

            self.current_step += 1
            return elements, message

        # Animation stop callback
        @self.app.callback(
            Output('animation-interval', 'disabled', allow_duplicate=True),
            Input('animation-interval', 'n_intervals'),
            prevent_initial_call=True
        )
        def stop_animation(n):
            if self.is_paused:
                return True
                
            if self.mode == 'forward':
                max_steps = len(self.forward_steps)
            else:
                max_steps = len(self.backward_steps)
            
            if self.current_step >= max_steps:
                return True
            return False

        # Neuron selection callback: when a node is clicked, show its weights and create an editable table.
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
            
            # Create an editable weights table
            weights_table = self._create_weights_table(weights_data)
            
            info_content = html.Div([
                html.H3(f"Neuron Info - Layer: {layer_name}, Neuron: {neuron_idx}"),
                weights_table
            ])
            
            return elements, info_content, selected, weights_table

        # Training callback using direct model.fit with a custom callback to backup weights.
        @self.app.callback(
            Output('training-status', 'children'),
            Output('nn-cytoscape', 'elements', allow_duplicate=True),
            Input('train-btn', 'n_clicks'),
            State('epochs-input', 'value'),
            State('nn-cytoscape', 'elements'),
            prevent_initial_call=True
        )
        def train_model(n_clicks, epochs, current_elements):
            if not epochs or epochs < 1:
                return "Please enter a valid number of epochs", dash.no_update
            try:
                save_interval = 1  # Save state every epoch (adjust as needed)
                # Define a training callback similar to MLWrapper's
                class TrainingCallback(tf.keras.callbacks.Callback):
                    def __init__(self, app):
                        self.app = app
                        self.epoch_count = 0

                    def on_epoch_end(self, epoch, logs=None):
                        self.app.current_epoch += 1
                        self.epoch_count += 1
                        if self.epoch_count % save_interval == 0:
                            # Backup weights for rollback
                            self.app.backup_weights = self.app.model.get_weights()
                
                history = self.model.fit(
                    self.x_train, self.y_train,
                    initial_epoch=self.current_epoch,
                    epochs=self.current_epoch + epochs,
                    callbacks=[TrainingCallback(self)],
                    verbose=0
                )
                # Update visualization weights after training
                self.visualizer._extract_weights()
                elements = self._clear_classes(current_elements)
                return f"Training completed for {epochs} epochs. Current epoch: {self.current_epoch}", elements
            except Exception as e:
                return f"Training failed: {str(e)}", dash.no_update

        # Weight modification callback using the manual inputs (if used)
        @self.app.callback(
            Output('weight-modification-status', 'children'),
            Output('nn-cytoscape', 'elements', allow_duplicate=True),
            Input('modify-weights-btn', 'n_clicks'),
            State('selected-neuron-store', 'data'),
            State('weight-input', 'value'),
            State('bias-input', 'value'),
            State('nn-cytoscape', 'elements'),
            prevent_initial_call=True
        )
        def manual_modify_weights(n_clicks, neuron_data, weights_str, bias, current_elements):
            if not neuron_data:
                return "Please select a neuron first", dash.no_update
            if not weights_str or bias is None:
                return "Please enter both weights and bias values", dash.no_update
            try:
                new_weights = [float(w.strip()) for w in weights_str.split(',')]
                layer_idx = next(
                    i for i, layer in enumerate(self.model.layers)
                    if layer.name == neuron_data['layer']
                )
                current_weights = self.model.layers[layer_idx].get_weights()
                current_weights[0][:, neuron_data['neuron']] = new_weights
                current_weights[1][neuron_data['neuron']] = bias
                self.model.layers[layer_idx].set_weights(current_weights)
                self.visualizer._extract_weights()
                elements = self._clear_classes(current_elements)
                success_msg = html.Div([
                    html.P("Weights modified successfully!", style={'color': 'green'}),
                    html.P(f"Layer: {neuron_data['layer']}, Neuron: {neuron_data['neuron']}"),
                    html.P(f"New weights: {', '.join(f'{w:.4f}' for w in new_weights)}"),
                    html.P(f"New bias: {bias:.4f}")
                ])
                return success_msg, elements
            except Exception as e:
                return html.Div(f"Error: {str(e)}", style={'color': 'red'}), dash.no_update

        # Callback to update weights from the editable table when a cell is edited.
        @self.app.callback(
            Output('weight-modification-status', 'children', allow_duplicate=True),
            Output('nn-cytoscape', 'elements', allow_duplicate=True),
            Input('weights-table', 'data_timestamp'),
            State('weights-table', 'data'),
            State('selected-neuron-store', 'data'),
            State('nn-cytoscape', 'elements'),
            prevent_initial_call=True
        )
        def update_weight_from_table(ts, table_data, neuron_data, current_elements):
            if not neuron_data:
                raise PreventUpdate
            layer_name = neuron_data['layer']
            neuron_idx = neuron_data['neuron']
            # Find the corresponding layer index.
            layer_idx = next(i for i, layer in enumerate(self.model.layers) if layer.name == layer_name)
            try:
                # Assume table_data rows: first len-1 rows for weights, last row for bias.
                new_weights = [float(row['Weight']) for row in table_data[:-1]]
                new_bias = float(table_data[-1]['Weight'])
            except Exception as e:
                return html.Div(f"Error parsing table data: {str(e)}", style={'color': 'red'}), current_elements

            current_weights = self.model.layers[layer_idx].get_weights()
            if len(new_weights) != current_weights[0].shape[0]:
                return html.Div("Dimension mismatch in weight update.", style={'color': 'red'}), current_elements
            current_weights[0][:, neuron_idx] = new_weights
            current_weights[1][neuron_idx] = new_bias
            self.model.layers[layer_idx].set_weights(current_weights)
            self.visualizer._extract_weights()
            new_elements = self._clear_classes(current_elements)
            return html.Div("Weights updated from table.", style={'color': 'green'}), new_elements

        # Rollback callback
        @self.app.callback(
            Output('training-status', 'children', allow_duplicate=True),
            Output('nn-cytoscape', 'elements', allow_duplicate=True),
            Input('rollback-btn', 'n_clicks'),
            State('nn-cytoscape', 'elements'),
            prevent_initial_call=True
        )
        def rollback_model(n_clicks, current_elements):
            if self.backup_weights is not None:
                self.model.set_weights(self.backup_weights)
                self.visualizer._extract_weights()
                elements = self._clear_classes(current_elements)
                return "Rolled back to the backup weights.", elements
            else:
                return "No backup available for rollback.", dash.no_update

    def _clear_classes(self, elements):
        for el in elements:
            el['classes'] = el.get('classes', '')
            if 'selected-node' not in el['classes']:
                el['classes'] = ''
        return elements

    def run(self):
        self.app.run(debug=True)

# Main block: build a NN with 4 layers (each with 4 neurons), create random data, and run the app.
if __name__ == "__main__":
    tf.keras.backend.clear_session()
    
    # Build a neural network with 4 layers (each with 4 neurons)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(4,), name='input_layer'),
        tf.keras.layers.Dense(4, activation='relu', name='layer_1'),
        tf.keras.layers.Dense(4, activation='relu', name='layer_2'),
        tf.keras.layers.Dense(4, activation='relu', name='layer_3'),
        tf.keras.layers.Dense(4, activation='softmax', name='output_layer')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Generate a random dataset: 100 samples, 4 features, 4 classes.
    x_train = np.random.rand(100, 4).astype('float32')
    y_train = np.random.randint(0, 4, size=(100,))
    
    # Instantiate the Dash application with the model.
    app = DashFullNNApp(model)
    # Attach training data to the app instance so callbacks can use them.
    app.x_train = x_train
    app.y_train = y_train
    # Backup initial weights for rollback
    app.backup_weights = model.get_weights()
    
    app.run()
