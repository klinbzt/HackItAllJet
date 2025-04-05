import dash
from dash import html
import dash_cytoscape as cyto

app = dash.Dash(__name__)

# Define neurons (nodes)
nodes = []
input_layer_size = 3
hidden_layer_size = 5
output_layer_size = 2

# Helper to create neuron nodes
def create_nodes(layer_name, layer_index, size):
    y_spacing = 100
    return [
        {
            'data': {'id': f'{layer_name}_{i}', 'label': f'{layer_name.capitalize()} {i+1}'},
            'position': {'x': layer_index * 200, 'y': i * y_spacing},
            'grabbable': True
        }
        for i in range(size)
    ]

# Create nodes and edges
nodes += create_nodes('input', 0, input_layer_size)
nodes += create_nodes('hidden', 1, hidden_layer_size)
nodes += create_nodes('output', 2, output_layer_size)

edges = []

# Connect input -> hidden
for i in range(input_layer_size):
    for j in range(hidden_layer_size):
        edges.append({'data': {'source': f'input_{i}', 'target': f'hidden_{j}'}})

# Connect hidden -> output
for i in range(hidden_layer_size):
    for j in range(output_layer_size):
        edges.append({'data': {'source': f'hidden_{i}', 'target': f'output_{j}'}})

# Layout
app.layout = html.Div([
    html.H3("Drag-and-Drop Neural Network Visualizer"),
    cyto.Cytoscape(
        id='nn-graph',
        layout={'name': 'preset'},  # Positions are custom, not auto-generated
        style={'width': '100%', 'height': '600px'},
        elements=nodes + edges,
        stylesheet=[
            {'selector': 'node', 'style': {
                'label': 'data(label)',
                'width': 30,
                'height': 30,
                'background-color': '#87CEFA',
                'text-valign': 'center',
                'text-halign': 'center',
                'border-width': 2,
                'border-color': '#2F4F4F'
            }},
            {'selector': 'edge', 'style': {
                'line-color': '#ccc',
                'width': 2
            }}
        ],
        userPanningEnabled=True,
        userZoomingEnabled=True,
        boxSelectionEnabled=False,
        autoungrabify=False
    )
])

if __name__ == '__main__':
    app.run(debug=True)