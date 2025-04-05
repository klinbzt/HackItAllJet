import dash
from dash import html, dcc, Output, Input, State, ctx
import dash_cytoscape as cyto
import tensorflow as tf
from dash.exceptions import PreventUpdate
import copy
import numpy as np
from dash import dash_table

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
        for i, layer in enumerate(self.model.layers):
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
                    'data': {'id': node_id, 'label': f"{i}", 'layer': layer_idx, 
                            'layer_name': layer_id, 'neuron_idx': i},
                    'position': {'x': x, 'y': y_offset + i * spacing_y},
                    'classes': ''
                })

            for from_node in prev_layer_neurons:
                for to_node in current_layer_neurons:
                    self.elements.append({
                        'data': {'source': from_node, 'target': to_node, 
                                'id': f"{from_node}->{to_node}"},
                        'classes': ''
                    })

            if prev_layer_neurons:
                # For forward steps, we'll process neuron-by-neuron in the current layer
                self.forward_steps.extend([
                    (prev_layer_neurons, [current_neuron])
                    for current_neuron in current_layer_neurons
                ])
                
                # For backward steps, we'll process neuron-by-neuron in the previous layer
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
        # So each column represents a neuron's weights
        neuron_weights = layer_weights['weights'][:, neuron_idx]
        neuron_bias = layer_weights['biases'][neuron_idx]
        
        return {
            'weights': neuron_weights,
            'bias': neuron_bias
        }

class DashFullNNApp:
    def __init__(self, model):
        self.model = model
        self.visualizer = FullNNVisualizer(model)
        self.elements = self.visualizer.get_elements()
        self.forward_steps = self.visualizer.forward_steps
        self.backward_steps = self.visualizer.backward_steps
        self.current_step = 0
        self.mode = None  # 'forward' or 'backward'
        self.animation_interval = None
        self.is_paused = False
        self.selected_neuron = None

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
            cyto.Cytoscape(
                id='nn-cytoscape',
                elements=self.elements,
                layout={'name': 'preset'},
                style={'width': '100%', 'height': '700px'},
                stylesheet=self.visualizer.get_stylesheet(),
            ),
            html.Div(id='neuron-info', style={
                'marginTop': '20px',
                'padding': '10px',
                'border': '1px solid #ddd',
                'borderRadius': '5px',
                'backgroundColor': '#f9f9f9'
            }),
            dcc.Store(id='selected-neuron-store')
        ])

    def _create_weights_table(self, weights_data):
        """Create a table showing weights for a neuron"""
        if not weights_data:
            return html.P("No weights available for input layer neurons")
        
        weights = weights_data['weights']
        bias = weights_data['bias']
        
        # Create table data
        table_data = [{
            'Connection': f'Input {i}',
            'Weight': f'{weight:.4f}'
        } for i, weight in enumerate(weights)]
        
        table_data.append({
            'Connection': 'Bias',
            'Weight': f'{bias:.4f}'
        })
        
        return html.Div([
            html.H4("Neuron Weights"),
            dash_table.DataTable(
                id='weights-table',
                columns=[
                    {'name': 'Connection', 'id': 'Connection'},
                    {'name': 'Weight', 'id': 'Weight'}
                ],
                data=table_data,
                style_table={'overflowX': 'auto'},
                style_cell={
                    'textAlign': 'left',
                    'padding': '8px'
                },
                style_header={
                    'backgroundColor': 'rgb(230, 230, 230)',
                    'fontWeight': 'bold'
                },
                style_data_conditional=[
                    {
                        'if': {'row_index': len(table_data)-1},
                        'fontWeight': 'bold',
                        'backgroundColor': 'rgb(240, 240, 240)'
                    }
                ]
            )
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

            # Make a deep copy of elements to modify
            elements = [copy.deepcopy(el) for el in current_elements]

            from_nodes, to_nodes = steps[self.current_step]

            # Reset all classes first (except selected nodes)
            for el in elements:
                if 'selected-node' not in el['classes']:
                    el['classes'] = ''

            # Highlight previous layer nodes (green)
            for el in elements:
                el_id = el['data'].get('id')
                if el_id in from_nodes:
                    el['classes'] = 'previous-node'
            
            # Highlight current receiving node (yellow)
            for el in elements:
                el_id = el['data'].get('id')
                if el_id in to_nodes:
                    el['classes'] = 'receiving-node'
            
            # Highlight active edges (orange)
            for el in elements:
                if 'source' in el['data']:
                    if el['data']['source'] in from_nodes and el['data']['target'] in to_nodes:
                        el['classes'] = 'active-edge'
            
            # Create informative message
            if self.mode == 'forward':
                layer_idx = next(el['data']['layer'] for el in elements 
                             if el['data'].get('id') == to_nodes[0])
                message = f"Layer {layer_idx}: Neuron {to_nodes[0].split('_')[-1]} receiving inputs"
            else:
                layer_idx = next(el['data']['layer'] for el in elements 
                             if el['data'].get('id') == from_nodes[0])
                message = f"Layer {layer_idx}: Neuron {from_nodes[0].split('_')[-1]} sending gradients"

            self.current_step += 1
            return elements, message

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
                return True  # Disable interval
            return False  # Keep interval enabled

        @self.app.callback(
            Output('nn-cytoscape', 'elements', allow_duplicate=True),
            Output('neuron-info', 'children'),
            Output('selected-neuron-store', 'data'),
            Input('nn-cytoscape', 'tapNode'),
            State('nn-cytoscape', 'elements'),
            State('selected-neuron-store', 'data'),
            prevent_initial_call=True
        )
        def select_neuron(node_data, current_elements, stored_data):
            if not node_data or not self.is_paused and not self.current_step == 0:
                raise PreventUpdate
                
            # Make a deep copy of elements to modify
            elements = [copy.deepcopy(el) for el in current_elements]
            
            # Clear previous selection
            for el in elements:
                if 'selected-node' in el['classes']:
                    el['classes'] = el['classes'].replace('selected-node', '').strip()
            
            # Get the clicked node ID
            node_id = node_data['data']['id']
            
            # Find and highlight the clicked node
            for el in elements:
                if el['data']['id'] == node_id:
                    el['classes'] = 'selected-node'
                    break
            
            # Get neuron info
            layer_name = node_data['data']['layer_name']
            neuron_idx = node_data['data']['neuron_idx']
            self.selected_neuron = {'layer': layer_name, 'neuron': neuron_idx}
            
            # Get weights for this neuron
            weights_data = self.visualizer.get_weights_for_neuron(layer_name, neuron_idx)
            
            # Create info display
            info_content = html.Div([
                html.H3(f"Neuron Info - Layer: {layer_name}, Neuron: {neuron_idx}"),
                self._create_weights_table(weights_data)
            ])
            
            return elements, info_content, {'layer': layer_name, 'neuron': neuron_idx}

    def _clear_classes(self, elements):
        # Reset classes for all elements (except selected nodes)
        for el in elements:
            if 'selected-node' not in el['classes']:
                el['classes'] = ''
        return elements

    def run(self):
        self.app.run(debug=True)

# Example usage
if __name__ == "__main__":
    tf.keras.backend.clear_session()
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(3,), name='input_layer'),
        tf.keras.layers.Dense(4, activation='relu', name='hidden_1'),
        tf.keras.layers.Dense(2, activation='softmax', name='output_layer')
    ])
    app = DashFullNNApp(model)
    app.run()