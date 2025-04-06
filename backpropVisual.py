# genericChainBackprop.py
import copy
import numpy as np
import dash_cytoscape as cyto
import tensorflow as tf

class GenericChainBackprop:
    """
    Builds a simple chain of intermediate layer outputs for any Keras model
    and uses finite differences to approximate partial derivatives w.r.t each node.

    The chain's "nodes" are:
      a0 = x  (the input)
      a1 = model.layers[0](a0)
      a2 = model.layers[1](a1)
      ...
      aL = model.layers[L-1](a_{L-1})
      final_loss = loss(aL, y)

    We store forward_steps = [(a0)->(a1), (a1)->(a2), ..., (a_{L-1})->(aL)]
    and backward_steps = [(aL)->(a_{L-1}), ..., (a1)->(a0)].

    For the backward pass, we do finite difference w.r.t the *mean* of a_i for demonstration.
    """

    def __init__(self, model, x, y, loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(), epsilon=1e-4):
        """
        :param model: A compiled Keras model.
        :param x: A single input sample, shape = (1, input_dim). (Or can be batched, for demonstration.)
        :param y: The corresponding label for that input, shape = (1,) or (1, output_dim).
        :param loss_fn: A Keras/tf loss function object, e.g. SparseCategoricalCrossentropy.
        :param epsilon: Step size for finite difference.
        """
        self.model = model
        self.x = x
        self.y = y
        self.loss_fn = loss_fn
        self.epsilon = epsilon

        # We'll store each "node" in the chain as a dictionary with:
        # {"id": "a_i", "label": "...", "activation": ...}
        # a0 is input, aL is the final layer output
        self.nodes = []
        self.forward_steps = []
        self.backward_steps = []
        self.backward_derivatives = []

        # Build the chain
        self._build_chain()

    def _build_chain(self):
        """
        1) Perform a forward pass through each layer, storing each layer's mean activation as "a_i".
        2) Build forward_steps for i -> i+1.
        3) Build backward_steps for L -> L-1, etc.
        4) Approximate partial derivatives with finite difference on each a_i's mean value.
        """
        # a0 = x
        a_list = [self.x]  # store actual tf Tensors for each layer's output
        for layer in self.model.layers:
            a_list.append(layer(a_list[-1]))

        # We'll keep them in self.nodes as Cytoscape elements
        # Node ID: "a0", "a1", ...
        self.nodes = []
        for i, a_i_tensor in enumerate(a_list):
            mean_val = float(tf.reduce_mean(a_i_tensor))  # single scalar
            self.nodes.append({
                "id": f"a_{i}",
                "label": f"a{i} ~ {mean_val:.3f}",
                "activation": a_i_tensor  # store the actual tensor
            })

        # Build forward steps
        self.forward_steps = []
        for i in range(len(self.nodes) - 1):
            from_id = self.nodes[i]["id"]
            to_id = self.nodes[i+1]["id"]
            self.forward_steps.append(([from_id], [to_id]))

        # Build backward steps
        # We'll just do a_{L} -> a_{L-1} -> ... -> a_0
        self.backward_steps = []
        for i in range(len(self.nodes)-1, 0, -1):
            from_id = self.nodes[i]["id"]
            to_id = self.nodes[i-1]["id"]
            self.backward_steps.append(([from_id], [to_id]))

        # Compute final loss
        final_out = a_list[-1]
        loss_val = self.loss_fn(self.y, final_out)
        loss_val = float(loss_val.numpy())

        # We'll do finite differences for each "activation node" from a_{L} down to a_1.
        # (We skip a0 because it's "x"; you *could* approximate partial d(loss)/d(x).)
        self.backward_derivatives = []
        for i in range(len(self.nodes)-1, 0, -1):
            # We'll define a small function that sets the mean of a_i to a_i_mean + epsilon,
            # then forward the rest to get a new loss. For demonstration, we "hack" the entire a_i.
            # This is obviously a big simplification for multi-dimensional a_i. 
            def fn_perturb_a_i(eps):
                # Re-run the forward pass, but override "a_i" with "a_i + eps shift in the mean"
                # We'll keep everything up to i-1 the same, then from i onward we re-run.
                # 1) Get original activations up to i-1
                sub_list = [self.x]
                for layer_idx, layer in enumerate(self.model.layers[:i]):
                    sub_list.append(layer(sub_list[-1]))

                original_tensor = sub_list[-1]  # this is a_i
                # Shift its mean by eps:
                mean_orig = tf.reduce_mean(original_tensor)
                shift = eps
                # scale factor to shift the entire tensor by 'shift / shape'
                shape = tf.shape(original_tensor)
                # We'll do: new_tensor = original_tensor + shift * ones
                # so the new mean is mean_orig + shift.
                new_tensor = original_tensor + shift

                # Then pass new_tensor through the remaining layers
                next_val = new_tensor
                for layer_idx, layer in enumerate(self.model.layers[i:]):
                    next_val = layer(next_val)

                # Compute the new loss
                new_loss = self.loss_fn(self.y, next_val)
                return float(new_loss.numpy())

            base_loss = loss_val
            # Evaluate loss with +epsilon
            plus_loss = fn_perturb_a_i(self.epsilon)
            # Approx derivative
            dloss_dai = (plus_loss - base_loss) / self.epsilon
            self.backward_derivatives.append(f"dLoss/d(a{i}) ≈ {dloss_dai:.4f}")

    def get_elements(self):
        """
        Return Cytoscape elements: one node per a_i, plus edges for forward pass.
        We'll place them horizontally for convenience.
        """
        elements = []
        x_start = 100
        y_mid = 300
        spacing = 200

        # Make nodes
        for i, node_info in enumerate(self.nodes):
            elements.append({
                "data": {
                    "id": node_info["id"],
                    "label": node_info["label"],
                },
                "position": {"x": x_start + i*spacing, "y": y_mid},
            })

        # Make edges from forward_steps
        for (from_list, to_list) in self.forward_steps:
            for f_id in from_list:
                for t_id in to_list:
                    elements.append({
                        "data": {
                            "source": f_id,
                            "target": t_id
                        }
                    })

        return elements

    def get_forward_steps(self):
        # e.g. [(["a_0"], ["a_1"]), (["a_1"], ["a_2"]), ...]
        return self.forward_steps

    def get_backward_steps(self):
        # e.g. [(["a_2"], ["a_1"]), (["a_1"], ["a_0"])]
        return self.backward_steps

    def get_backward_derivs(self):
        # e.g. ["dLoss/d(a2) ~ 3.14", "dLoss/d(a1) ~ 0.12", ...]
        return self.backward_derivatives

    def get_stylesheet(self):
        """
        A minimal Cytoscape stylesheet that highlights nodes/edges on forward or backward steps.
        """
        return [
            {
                "selector": "node",
                "style": {
                    "label": "data(label)",
                    "text-halign": "center",
                    "text-valign": "center",
                    "border-width": 2,
                    "border-color": "#999",
                    "background-color": "#eee",
                },
            },
            {
                "selector": "edge",
                "style": {
                    "width": 3,
                    "line-color": "#ccc",
                    "target-arrow-color": "#ccc",
                    "target-arrow-shape": "triangle",
                },
            },
            {
                "selector": ".previous-node",
                "style": {
                    "background-color": "#f5a",
                    "border-color": "#f5a",
                },
            },
            {
                "selector": ".receiving-node",
                "style": {
                    "background-color": "#6f6",
                    "border-color": "#6f6",
                },
            },
            {
                "selector": ".active-edge",
                "style": {
                    "line-color": "#fa0",
                    "target-arrow-color": "#fa0",
                    "width": 4,
                },
            },
            {
                "selector": ".selected-node",
                "style": {
                    "background-color": "#00BFFF",
                    "border-color": "#00BFFF",
                    "border-width": 4,
                },
            },
        ]
