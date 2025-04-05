import numpy as np
import plotly.graph_objs as go

def simulate_gradient_descent(model, X, y, epochs=20):
    from tensorflow.keras.models import clone_model

    descent_path = []

    # Clone the model to avoid modifying the original
    sim_model = clone_model(model)
    sim_model.set_weights(model.get_weights())
    sim_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    for _ in range(epochs):
        weights = sim_model.layers[0].get_weights()[0].copy()
        biases = sim_model.layers[0].get_weights()[1].copy()
        descent_path.append((weights, biases))
        sim_model.fit(X, y, epochs=1, verbose=0)

    return descent_path

def generate_loss_surface(model, X, y, weight_idx1=(0, 0), weight_idx2=(0, 1), span=2.0, steps=30):
    print("Generating loss surface...")
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
            loss = model.evaluate(X, y, verbose=0)[0]  # only the scalar loss
            Z[i, j] = loss

    model.layers[0].set_weights(original_weights)
    return w1_range, w2_range, Z


def plot_3d_surface(model, X, y, w1_range, w2_range, Z, descent_path):
    fig = go.Figure(data=[
        go.Surface(z=Z.T, x=w1_range, y=w2_range, colorscale='Viridis', opacity=0.8)
    ])

    w00s = [w[0][0] for w, _ in descent_path]
    w01s = [w[0][1] for w, _ in descent_path]
    losses = []
    for weights, biases in descent_path:
        model.layers[0].set_weights([weights, biases])
        loss = model.evaluate(X, y, verbose=0)[0]  # only the scalar loss
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
