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

