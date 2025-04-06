import dash
from dash import html, dcc, Output, Input, State, ctx, dash_table
import dash_cytoscape as cyto
import tensorflow as tf
from dash.exceptions import PreventUpdate
import copy
import numpy as np
from modelWrapper import MLWrapper
import datetime
import os
import json
from chatbot import ChatAdvisor
import dash_bootstrap_components as dbc

openai_api_key = "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"

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
    def __init__(self, ml_wrapper):  # Change to take MLWrapper instance
        self.ml_wrapper = ml_wrapper
        self.model = ml_wrapper.model
        self.current_epoch = 0
        self.visualizer = FullNNVisualizer(self.model)
        self.elements = self.visualizer.get_elements()
        self.forward_steps = self.visualizer.forward_steps
        self.backward_steps = self.visualizer.backward_steps
        self.current_step = 0
        self.mode = None
        self.is_paused = False
        self.selected_neuron = None
        # Create TensorBoard log directory
        self.log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(self.log_dir, exist_ok=True)
        
        # TensorBoard callback
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=self.log_dir,
            histogram_freq=1,  # Log weight histograms every epoch
            write_graph=True,
            write_images=True,
            update_freq='epoch',
            profile_batch=0  # Disable profiling for simplicity
        )
        self.app = dash.Dash(__name__, external_stylesheets=[
    dbc.themes.BOOTSTRAP,
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css"
])
        self.app.title = "NN Propagation Visualizer"
        self.chat_advisor = ChatAdvisor(self.app, openai_api_key)
        self._build_layout()
        self._register_callbacks()
        

    def _build_layout(self):
        # Custom CSS styles
        custom_css = {
            'button': {
                'transition': 'all 0.3s ease',
                'transform': 'scale(1)',
                'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
                'borderRadius': '8px',
                'fontWeight': '600',
                'letterSpacing': '0.5px'
            },
            'button:hover': {
                'transform': 'scale(1.05)',
                'boxShadow': '0 6px 8px rgba(0, 0, 0, 0.15)'
            },
            'button:active': {
                'transform': 'scale(0.98)'
            },
            'container': {
                'background': 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)',
                'padding': '2rem',
                'minHeight': '100vh'
            },
            'card': {
                'borderRadius': '12px',
                'boxShadow': '0 6px 10px rgba(0, 0, 0, 0.08)'
            }
        }

        self.app.layout = dbc.Container([
            # Custom CSS
            
            dbc.Row(
                dbc.Col(
                    html.H2("Model Training Visualizer", 
                            className="text-center text-primary my-4",
                            style={
                                'textShadow': '1px 1px 2px rgba(0,0,0,0.1)',
                                'fontSize': '42px'  # Adjust this value as needed
                            })
                )
            ),
            dbc.Row([
                dbc.Col(
                    dbc.Button(
                        html.Span([
                            html.I(className="fas fa-play me-2"),  # Font Awesome icon
                            "Forward Propagation"
                        ]),
                        id="forward-btn", n_clicks=0,
                        className="btn-custom-success btn-hover-animation me-2 mb-2",
                        style=custom_css['button']
                    )
                ),
                dbc.Col(
                    dbc.Button(
                        html.Span([
                            html.I(className="fas fa-undo me-2"),  # Font Awesome icon
                            "Backpropagation"
                        ]),
                        id="backward-btn", n_clicks=0,
                        className="btn-custom-secondary btn-hover-animation me-2 mb-2",
                        style=custom_css['button']
                    )
                ),
                dbc.Col(
                    dbc.Button(
                        html.Span([
                            html.I(className="fas fa-pause me-2"),  # Font Awesome icon
                            "Pause"
                        ]),
                        id="pause-btn", n_clicks=0,
                        className="btn-custom-warning btn-hover-animation me-2 mb-2",
                        style=custom_css['button']
                    )
                ),
                dbc.Col(
                    dbc.Button(
                        html.Span([
                            html.I(className="fas fa-chart-line me-2"),  # Font Awesome icon
                            "Open TensorBoard"
                        ]),
                        id="tensorboard-btn", n_clicks=0,
                        className="btn-custom-info btn-hover-animation ms-2 mb-2",
                        style=custom_css['button']
                    )
                ),
                dcc.Interval(id='animation-interval', interval=1000, disabled=True)
            ], align="center", className="mb-4"),
            
            dbc.Row(
                dbc.Col(
                    html.Div(id='step-info', 
                            className="fw-bold text-dark ms-2 my-3",
                            style={'fontSize': '1.1rem'}),
                    width=12
                )
            ),
            
            dbc.Row(
                dbc.Col(
                    html.Div(self.chat_advisor.get_component(), 
                            className="my-4",
                            style={'borderRadius': '10px', 'overflow': 'hidden'})
                )
            ),
            
            dbc.Card([
                dbc.CardHeader(
                    html.H3("Training Controls", className="text-white"), 
                    className="bg-dark",
                    style={'borderTopLeftRadius': '12px', 'borderTopRightRadius': '12px'}
                ),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(
                            dbc.Input(
                                id='epochs-input', type='number', value=5, min=1,
                                style={'width': '80px', 'borderRadius': '6px'}
                            ),
                            width="auto"
                        ),
                        dbc.Col(
                            dbc.Button(
                                html.Span([
                                    html.I(className="fas fa-brain me-2"),
                                    "Train"
                                ]),
                                id='train-btn', n_clicks=0,
                                className="btn-custom-primary btn-hover-animation ms-2",
                                style=custom_css['button']
                            ),
                            width="auto"
                        ),
                        dbc.Col(
                            dbc.Button(
                                html.Span([
                                    html.I(className="fas fa-history me-2"),
                                    "Rollback"
                                ]),
                                id='rollback-btn', n_clicks=0,
                                className="btn-custom-danger btn-hover-animation ms-2",
                                style=custom_css['button']
                            ),
                            width="auto"
                        )
                    ], align="center", className="mb-3"),
                    
                    html.Div(id='training-status', className="mb-3"),
                    
                    dbc.Row([
                        dbc.Col(
                            html.Iframe(
                                id='tensorboard-iframe',
                                src=f"http://localhost:6006/",
                                style={
                                    'width': '100%', 
                                    'height': '600px', 
                                    'border': 'none',
                                    'borderRadius': '8px'
                                }
                            )
                        )
                    ], className="mt-3", id="tensorboard-container", style={'display': 'none'}),
                    
                    dbc.Row(
                        dbc.Col(
                            html.H4("Weight Modification", 
                                className="mt-3",
                                style={'color': '#4a4a4a'})
                        )
                    ),
                    dbc.Row([
                        dbc.Col(
                            dbc.Input(
                                id='source-epoch-input', 
                                type='number', 
                                placeholder='Epoch to modify',
                                min=0, 
                                style={'width': '150px', 'borderRadius': '6px'}
                            ),
                            width="auto"
                        ),
                        dbc.Col(
                            dbc.Button(
                                html.Span([
                                    html.I(className="fas fa-edit me-2"),
                                    "Modify Weights"
                                ]),
                                id='modify-weights-btn', n_clicks=0,
                                className="btn-custom-primary ms-2",
                                style=custom_css['button']
                            ),
                            width="auto"
                        )
                    ], align="center", className="mb-3"),
                    
                    html.Div(id='modification-status', className="mt-2"),
                    html.Div(id='editable-weights-table'),
                    
                    dbc.Row(
                        dbc.Col(
                            html.H5("Compare Epoch States", 
                                className="mt-3",
                                style={'color': '#4a4a4a'})
                        )
                    ),
                    dbc.Row([
                        dbc.Col(
                            dbc.Input(
                                id='compare-epoch-1', 
                                type='number', 
                                placeholder='Epoch 1', 
                                min=0, 
                                style={'width': '120px', 'borderRadius': '6px'}
                            ),
                            width="auto"
                        ),
                        dbc.Col(
                            dbc.Input(
                                id='compare-epoch-2', 
                                type='number', 
                                placeholder='Epoch 2', 
                                min=0,
                                style={
                                    'width': '120px', 
                                    'marginLeft': '10px',
                                    'borderRadius': '6px'
                                }
                            ),
                            width="auto"
                        ),
                        dbc.Col(
                            dbc.Button(
                                html.Span([
                                    html.I(className="fas fa-balance-scale me-2"),
                                    "Compare"
                                ]),
                                id='compare-btn', n_clicks=0,
                                className="btn-custom-info ms-2",
                                style=custom_css['button']
                            ),
                            width="auto"
                        )
                    ], align="center"),
                    
                    dbc.Row(
                        dbc.Col(
                            html.Pre(id='compare-output', style={
                                'whiteSpace': 'pre-wrap',
                                'backgroundColor': '#f8f8f8',
                                'padding': '15px',
                                'border': '1px solid #ddd',
                                'borderRadius': '8px',
                                'fontFamily': 'monospace',
                                'fontSize': '0.9rem'
                            }),
                            className="mt-3"
                        )
                    )
                ])
            ], className="mb-4"),
            
            dbc.Row(
                dbc.Col(
                    cyto.Cytoscape(
                        id='nn-cytoscape',
                        elements=self.elements,
                        layout={'name': 'preset'},
                        style={
                            'width': '100%', 
                            'height': '700px',
                            'borderRadius': '12px',
                            'boxShadow': '0 4px 8px rgba(0,0,0,0.1)'
                        },
                        stylesheet=self.visualizer.get_stylesheet(),
                    ),
                    width=12
                )
            ),
            
            dbc.Row(
                dbc.Col(
                    html.Div(
                        id='neuron-info', 
                        style={
                            'backgroundColor': 'white',
                            'padding': '15px',
                            'borderRadius': '8px',
                            'boxShadow': '0 2px 4px rgba(0,0,0,0.05)'
                        }
                    ), 
                    width=12
                )
            ),
            
            dcc.Store(id='selected-neuron-store')
        ], fluid=True, style=custom_css['container'])
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
    
    def _compare_model_states(self, state1, state2):
        import numpy as np

        epoch1 = state1.get("epoch", "?")
        epoch2 = state2.get("epoch", "?")
        loss1 = state1.get("loss", None)
        loss2 = state2.get("loss", None)

        weights1 = state1["weights"]
        weights2 = state2["weights"]

        lines = []
        lines.append(f"📊 Epoch {epoch1} vs Epoch {epoch2}")
        lines.append("-" * 25)

        # Loss comparison
        if loss1 is not None and loss2 is not None:
            delta = loss2 - loss1
            sign = "✅" if delta < 0 else "⚠️"
            lines.append(f"- Loss changed from {loss1:.4f} → {loss2:.4f} {sign} ({delta:+.4f})")

        # Per-layer weight/bias diff
        max_bias_delta = 0
        most_changed_neuron = (None, None, 0.0)
        for i, (layer1, layer2) in enumerate(zip(weights1, weights2)):
            W1, b1 = np.array(layer1[0]), np.array(layer1[1])
            W2, b2 = np.array(layer2[0]), np.array(layer2[1])

            weight_diff = np.linalg.norm(W2 - W1)
            bias_diff = np.abs(b2 - b1)
            max_bias_layer = np.max(bias_diff)

            lines.append(f"- Layer {i+1}: weights changed by L2 norm {weight_diff:.4f}")

            if max_bias_layer > max_bias_delta:
                max_bias_delta = max_bias_layer
                most_neuron = np.argmax(bias_diff)
                most_changed_neuron = (i+1, most_neuron, max_bias_layer)

        lines.append(f"- Layer {most_changed_neuron[0]}: biases changed most: max delta {most_changed_neuron[2]:.4f}")
        lines.append(f"- Most affected neuron: layer_{most_changed_neuron[0]}, neuron {most_changed_neuron[1]} (bias changed by {most_changed_neuron[2]:.4f})")

        return "\n".join(lines)

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
        
        @self.app.callback(
            Output('compare-output', 'children'),
            Input('compare-btn', 'n_clicks'),
            State('compare-epoch-1', 'value'),
            State('compare-epoch-2', 'value'),
            prevent_initial_call=True
        )
        def compare_epochs(n_clicks, epoch1, epoch2):
            if epoch1 is None or epoch2 is None:
                return "Please enter two valid epoch numbers."

            try:
                with open(f"model_state/current/epoch_{epoch1}.json") as f1, open(f"model_state/current/epoch_{epoch2}.json") as f2:
                    state1 = json.load(f1)
                    state2 = json.load(f2)
            except Exception as e:
                return f"Error loading files: {e}"

            return self._compare_model_states(state1, state2)

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
            Output('tensorboard-container', 'style'),
            Input('train-btn', 'n_clicks'),
            State('epochs-input', 'value'),
            State('nn-cytoscape', 'elements'),
            prevent_initial_call=True
        )
        def train_model(n_clicks, epochs, current_elements):
            if not epochs or epochs < 1:
                return "Please enter a valid number of epochs", dash.no_update, dash.no_update
            
            try:
                save_interval = 1
                
                class TrainingCallback(tf.keras.callbacks.Callback):
                    def __init__(self, app):
                        self.app = app
                        self.epoch_count = 0

                    def on_epoch_end(self, epoch, logs=None):
                        self.app.current_epoch += 1
                        self.epoch_count += 1
                        if self.epoch_count % save_interval == 0:
                            self.app.backup_weights = self.app.model.get_weights()
                
                callbacks = [
                    TrainingCallback(self),
                    self.tensorboard_callback  # Add TensorBoard callback
                ]
                
                # history = self.model.fit(
                #     self.x_train, self.y_train,
                #     initial_epoch=self.current_epoch,
                #     epochs=self.current_epoch + epochs,
                #     callbacks=callbacks,
                #     verbose=0
                # )

                # Use MLWrapper's train method
                self.ml_wrapper.train(
                    (self.x_train, self.y_train),
                    epochs=epochs,
                    save_interval=1
                )

                self.visualizer._extract_weights()
                elements = self._clear_classes(current_elements)
                
                # Show TensorBoard after training
                tensorboard_style = {'display': 'block', 'marginTop': '20px'}
                return (f"Training completed for {epochs} epochs. Current epoch: {self.ml_wrapper.current_epoch}", 
                        elements, 
                        tensorboard_style)
            except Exception as e:
                return f"Training failed: {str(e)}", dash.no_update, dash.no_update

        # Callback to toggle TensorBoard display
        @self.app.callback(
            Output('tensorboard-container', 'style', allow_duplicate=True),
            Input('tensorboard-btn', 'n_clicks'),
            State('tensorboard-container', 'style'),
            prevent_initial_call=True
        )
        def toggle_tensorboard(n_clicks, current_style):
            if not n_clicks:
                raise PreventUpdate
                
            if current_style.get('display') == 'none':
                return {'display': 'block', 'marginTop': '20px'}
            else:
                return {'display': 'none'}

        # Weight modification callback using the manual inputs (if used)
        @self.app.callback(
            Output('modification-status', 'children'),
            Output('nn-cytoscape', 'elements', allow_duplicate=True),
            Input('modify-weights-btn', 'n_clicks'),
            State('selected-neuron-store', 'data'),
            State('source-epoch-input', 'value'),
            State('weights-table', 'data'),
            State('nn-cytoscape', 'elements'),
            prevent_initial_call=True
        )
        def modify_weights(n_clicks, neuron_data, source_epoch, table_data, current_elements):
            if not neuron_data or not source_epoch or not table_data:
                return "Complete all fields: select neuron, enter epoch, and modify weights", dash.no_update
            
            try:
                layer_name = neuron_data['layer']
                layer_idx = next(i for i, layer in enumerate(self.model.layers) if layer.name == layer_name)
                
                # Parse new weights and bias
                new_weights = [float(row['Weight']) for row in table_data[:-1]]
                new_bias = float(table_data[-1]['Weight'])
                
                # Get current weights and modify them
                layer_weights = self.model.layers[layer_idx].get_weights()
                layer_weights[0][:, neuron_data['neuron']] = new_weights
                layer_weights[1][neuron_data['neuron']] = new_bias
                
                # Apply modifications through MLWrapper
                self.ml_wrapper.modify_epoch(
                    layer_idx=layer_idx,
                    new_weights=layer_weights,
                    source_epoch=source_epoch,
                    train_data=None,
                    replay_training=False
                )
                
                # Log the manual modifications to TensorBoard
                log_dir = os.path.join("logs", "manual", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
                writer = tf.summary.create_file_writer(log_dir)
                
                with writer.as_default():
                    weights, biases = self.model.layers[layer_idx].get_weights()
                    tf.summary.histogram(f"{layer_name}/weights_modified", weights, step=source_epoch)
                    tf.summary.histogram(f"{layer_name}/biases_modified", biases, step=source_epoch)
                    writer.flush()
                
                self.visualizer._extract_weights()
                elements = self._clear_classes(current_elements)
                return f"Weights modified at epoch {source_epoch}. Changes logged to TensorBoard.", elements
            except Exception as e:
                return f"Error: {str(e)}", dash.no_update
            
        # Callback to update weights from the editable table when a cell is edited.
        # @self.app.callback(
        #     Output('weight-modification-status', 'children', allow_duplicate=True),
        #     Output('nn-cytoscape', 'elements', allow_duplicate=True),
        #     Input('weights-table', 'data_timestamp'),
        #     State('weights-table', 'data'),
        #     State('selected-neuron-store', 'data'),
        #     State('nn-cytoscape', 'elements'),
        #     prevent_initial_call=True
        # )
        # def update_weight_from_table(ts, table_data, neuron_data, current_elements):
        #     if not neuron_data:
        #         raise PreventUpdate
        #     layer_name = neuron_data['layer']
        #     neuron_idx = neuron_data['neuron']
        #     # Find the corresponding layer index.
        #     layer_idx = next(i for i, layer in enumerate(self.model.layers) if layer.name == layer_name)
        #     try:
        #         # Assume table_data rows: first len-1 rows for weights, last row for bias.
        #         new_weights = [float(row['Weight']) for row in table_data[:-1]]
        #         new_bias = float(table_data[-1]['Weight'])
        #     except Exception as e:
        #         return html.Div(f"Error parsing table data: {str(e)}", style={'color': 'red'}), current_elements

        #     current_weights = self.model.layers[layer_idx].get_weights()
        #     if len(new_weights) != current_weights[0].shape[0]:
        #         return html.Div("Dimension mismatch in weight update.", style={'color': 'red'}), current_elements
        #     current_weights[0][:, neuron_idx] = new_weights
        #     current_weights[1][neuron_idx] = new_bias
        #     self.model.layers[layer_idx].set_weights(current_weights)
        #     self.visualizer._extract_weights()
        #     new_elements = self._clear_classes(current_elements)
        #     return html.Div("Weights updated from table.", style={'color': 'green'}), new_elements

        # Rollback callback
        @self.app.callback(
            Output('training-status', 'children', allow_duplicate=True),
            Output('nn-cytoscape', 'elements', allow_duplicate=True),
            Input('rollback-btn', 'n_clicks'),
            State('nn-cytoscape', 'elements'),
            prevent_initial_call=True
        )
        def rollback_model(n_clicks, current_elements):
            try:
                self.ml_wrapper.rollback()
                self.visualizer._extract_weights()
                elements = self._clear_classes(current_elements)
                return "Rolled back to previous state.", elements
            except Exception as e:
                return f"Rollback failed: {str(e)}", dash.no_update

    def _clear_classes(self, elements):
        for el in elements:
            el['classes'] = el.get('classes', '')
            if 'selected-node' not in el['classes']:
                el['classes'] = ''
        return elements

    def run(self):
        self.app.run()

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
    ml_wrapper = MLWrapper(model)
    
    # Generate random data
    x_train = np.random.rand(100, 4)
    y_train = np.random.randint(0, 4, 100)

    import threading
    def run_tensorboard():
        import subprocess
        subprocess.run(["tensorboard", "--logdir", "logs", "--port", "6006", "--reload_multifile=true"])
        
    tensorboard_thread = threading.Thread(target=run_tensorboard, daemon=True)
    tensorboard_thread.start()
    
    
    # Initialize Dash app with MLWrapper
    app = DashFullNNApp(ml_wrapper)
    app.x_train = x_train
    app.y_train = y_train
    app.backup_weights = model.get_weights()
    app.run()