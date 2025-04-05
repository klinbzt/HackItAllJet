import os
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from sklearn.datasets import load_digits
import getModelnfo
from getModelnfo import ModelStateManager

# Importing the ModelStateManager class from the previous script (adjust import as needed)
# from model_state_manager import ModelStateManager  # This is assuming you have saved the previous script as 'model_state_manager.py'

# Stronger warning suppression
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TensorFlow logs
tf.get_logger().setLevel('ERROR')

# Initialize Dash app
app = Dash(__name__)

# Load or create the model (it assumes ModelStateManager is already implemented)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(64,), name='input_layer'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Initialize ModelStateManager with the model
state_mgr = ModelStateManager(model)

# Load a saved state (assuming there's a saved state file)
# Replace 'path_to_saved_state.json' with the actual path where you saved the state
saved_path = "model_states/state_init.json"  # Modify path accordingly
state_mgr.load_state(saved_path)

# Prepare sample data (digits dataset)
digits = load_digits()
X = digits.data
y = digits.target

# Dash layout
app.layout = html.Div([
    html.H1("Model State Visualization"),
    
    # Dropdown to select image
    dcc.Dropdown(
        id='digit-dropdown',
        options=[{'label': f"Digit {i}", 'value': i} for i in range(10)],
        value=0
    ),
    
    # Display the image
    dcc.Graph(id='image-display'),
    
    # Display model prediction
    html.Div(id='prediction-display')
])

# Callback to update the graph and prediction based on selected digit
@app.callback(
    [Output('image-display', 'figure'),
     Output('prediction-display', 'children')],
    [Input('digit-dropdown', 'value')]
)
def update_graph(digit_idx):
    # Extract the selected image
    digit_image = X[digit_idx].reshape(8, 8)
    
    # Make a prediction
    prediction = model.predict(np.expand_dims(X[digit_idx], axis=0))
    predicted_class = np.argmax(prediction, axis=1)[0]
    
    # Create a Plotly plot for the digit image
    image_fig = go.Figure(go.Image(z=digit_image))
    image_fig.update_layout(title=f"Digit {digit_idx} Image")
    
    # Prepare the prediction display
    prediction_message = f"Predicted Class: {predicted_class}"
    
    return image_fig, prediction_message

if __name__ == '__main__':
    app.run(debug=True)
