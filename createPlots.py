import numpy as np
import tensorflow as tf
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output

# Dummy model setup for demonstration (can be replaced with state_mgr.model)
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(2,)),
        tf.keras.layers.Dense(1, use_bias=True, name="dense")
    ])
    model.compile(optimizer='sgd', loss='mse')
    return model

# Sample dataset (XOR problem as example)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)

model = create_model()

# Perform one training step and collect descent path
def simulate_gradient_descent(model, epochs=20):
    descent_path = []
    for epoch in range(epochs):
        weights = model.layers[0].get_weights()[0]
        biases = model.layers[0].get_weights()[1]
        descent_path.append((weights.copy(), biases.copy()))
        model.fit(X, y, epochs=1, verbose=0)
    return descent_path

# Generate a 3D loss surface for two weights (w00 and w01)
def generate_loss_surface(model, weight_idx1=(0, 0), weight_idx2=(0, 1), span=2.0, steps=30):
    w1_range = np.linspace(-span, span, steps)
    w2_range = np.linspace(-span, span, steps)
    Z = np.zeros((steps, steps))

    original_weights = model.layers[0].get_weights()
    W_orig, b_orig = original_weights

    for i, w1 in enumerate(w1_range):
        for j, w2 in enumerate(w2_range):
            W_new = W_orig.copy()
            W_new[weight_idx1] = w1
            W_new[weight_idx2] = w2
            model.layers[0].set_weights([W_new, b_orig])
            loss = model.evaluate(X, y, verbose=0)
            Z[i, j] = loss

    # Restore original weights
    model.layers[0].set_weights(original_weights)
    return w1_range, w2_range, Z

# Build the figure
def plot_3d_surface(w1_range, w2_range, Z, descent_path):
    fig = go.Figure(data=[
        go.Surface(z=Z, x=w1_range, y=w2_range, colorscale='Viridis', opacity=0.8),
    ])

    # Add gradient descent path
    w00s = [w[0][0] for w, _ in descent_path]
    w01s = [w[0][1] for w, _ in descent_path]
    losses = []
    for weights, biases in descent_path:
        model.layers[0].set_weights([weights, biases])
        loss = model.evaluate(X, y, verbose=0)
        losses.append(loss)

    fig.add_trace(go.Scatter3d(
        x=w00s, y=w01s, z=losses,
        mode='lines+markers',
        marker=dict(size=4, color='red'),
        line=dict(color='red', width=3),
        name='Descent Path'
    ))

    fig.update_layout(
        title='3D Loss Surface + Descent Path',
        scene=dict(
            xaxis_title='Weight[0][0]',
            yaxis_title='Weight[0][1]',
            zaxis_title='Loss'
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    return fig

# Dash App
app = Dash(__name__)

descent_path = simulate_gradient_descent(model)
w1_range, w2_range, Z = generate_loss_surface(model)
fig = plot_3d_surface(w1_range, w2_range, Z, descent_path)

app.layout = html.Div([
    html.H2("Loss Surface and Gradient Descent Visualization"),
    dcc.Graph(id='loss-surface-plot', figure=fig),
])

if __name__ == '__main__':
    app.run_server(debug=True)
