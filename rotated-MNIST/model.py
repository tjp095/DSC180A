import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn


ACTIVATION_MAP = {
	"relu": nn.ReLU,
	"leaky_relu": nn.LeakyReLU,
	"sigmoid": nn.Sigmoid,
	"tanh": nn.Tanh,
	"elu": nn.ELU,
	"softmax": nn.Softmax,
	"softplus": nn.Softplus
}


class MultiTaskDNN(nn.Module):
	def __init__(self, model_cfg: dict, train_cfg: dict):
		''' Initializes model using parameters from config file '''
		super().__init__()

		# Activation function selected from activation map using config
		act_str = train_cfg.get('activation').lower()
		self.ActClass = ACTIVATION_MAP[act_str]

		self.flatten = nn.Flatten()

		# Helper function to build the task branches based on config parameters
		def build_branch_list(input_size, num_layers, nodes_per_layer, output_size):
			layers = nn.ModuleList()
			current_in = input_size

			# Create the specified number of layers:
			for _ in range(num_layers):
				layers.append(nn.Linear(current_in, nodes_per_layer))
				layers.append(nn.BatchNorm1d(nodes_per_layer)) # ADDING BATCH NORMALIZATION
				layers.append(self.ActClass()) # Dynamically instantiate the activation function
				current_in = nodes_per_layer

			# Add the final classification/regression layer (no activations after this)
			layers.append(nn.Linear(current_in, output_size))
			return layers

		# ---- Step 1. Shared Backbone ----
		# Starting with input size (784) --> loop for the shared layers --> output shared nodes
		self.shared_backbone = build_branch_list(
			input_size = model_cfg['input_shape'],
			num_layers = model_cfg['shared_layers'],
			nodes_per_layer = model_cfg['shared_nodes'],
			output_size = model_cfg['shared_nodes']
		)

		shared_out_size = model_cfg['shared_nodes']

		# ---- Step 2. Three Individual Task Branches ----
		# Takes in the shared output and applies to its task layers
		self.digit_head = build_branch_list(
			input_size = shared_out_size,
			num_layers = model_cfg['task_layers'],
			nodes_per_layer = model_cfg['task_nodes'],
			output_size = model_cfg['digit_out']
		)

		self.parity_head = build_branch_list(
			input_size = shared_out_size,
			num_layers = model_cfg['task_layers'],
			nodes_per_layer = model_cfg['task_nodes'],
			output_size = model_cfg['parity_out']
		)

		self.rotation_head = build_branch_list(
			input_size = shared_out_size,
			num_layers = model_cfg['task_layers'],
			nodes_per_layer = model_cfg['task_nodes'],
			output_size = model_cfg['rotation_out']
		)

	def forward(self, x):
		activations = {}
		x = self.flatten(x)

		# Process Shared Backbone:
		out = x
		layer_depth = 0
		for layer in self.shared_backbone:
			out = layer(out)
			# Dynamically check against the activation class used in this run
			if isinstance(layer, self.ActClass):
				activations[f'Shared_Layer_{layer_depth}'] = out
				layer_depth += 1

		shared_features = out

		# Process Digit Branch:
		out_d = shared_features
		layer_depth = 0
		for layer in self.digit_head:
			out_d = layer(out_d)
			if isinstance(layer, self.ActClass):
				activations[f'Digit_Layer_{layer_depth}'] = out_d
				layer_depth += 1
		digit_out = out_d

		# Process Parity Branch:
		out_p = shared_features
		layer_depth = 0
		for layer in self.parity_head:
			out_p = layer(out_p)
			if isinstance(layer, self.ActClass):
				activations[f'Parity_Layer_{layer_depth}'] = out_p
				layer_depth += 1
		parity_out = out_p

		# Process Rotation Branch:
		out_r = shared_features
		layer_depth = 0
		for layer in self.rotation_head:
			out_r = layer(out_r)
			if isinstance(layer, self.ActClass):
				activations[f'Rotation_Layer_{layer_depth}'] = out_r
				layer_depth += 1
		rotation_out = out_r

		# Return predictions and the dictionary of tracked variables
		return digit_out, parity_out, rotation_out, activations


# ========================================================
#					LOCAL SANITY CHECK
# ========================================================
if __name__ == "__main__":
	print("Running Model Architecture Sanity Check...")

	mock_model_cfg = {
		'input_shape': 784,
		'shared_layers': 2,
		'task_layers': 2,
		'shared_nodes': 16,
		'task_nodes': 16,
		'digit_out': 10,
		'parity_out': 2,
		'rotation_out': 1
	}

	mock_train_cfg = {
		'activation': 'relu'
	}

	# Initialize model
	model = MultiTaskDNN(mock_model_cfg, mock_train_cfg)

	# Create a dummy batch of 5 images (28x28)
	dummy_input = torch.randn(5, 1, 28, 28)

	# Pass through the model
	digit_preds, oddeven_preds, rot_preds, acts = model(dummy_input)

	print(f"Model successfully processed dummy batch of shape {dummy_input.shape}")
	print(f"Digit Output Shape: {digit_preds.shape} (Expected: 5, 10)")
	print(f"Odd/Even Output Shape: {oddeven_preds.shape} (Expected: 5, 2)")
	print(f"Rotation Output Shape: {rot_preds.shape} (Expected: 5, 2)")

	print("\nTracked Activations:")
	for key, tensor in acts.items():
		print(f"  {key}: shape {tensor.shape}")
