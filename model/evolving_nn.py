import torch
import torch.nn as nn
from datetime import datetime


class EvolvingNet(nn.Module):
    def __init__(self, input_size=10, hidden_dims=None):
        super().__init__()
        self.input_size = input_size
        self.hidden_dims = hidden_dims[:] if hidden_dims is not None else [16]
        self.layers = nn.ModuleList()
        self.classifier = None
        self.mutation_log = []
        self._rebuild_network()

    def _rebuild_network(self):
        self.layers = nn.ModuleList()
        last_dim = self.input_size

        for hidden_dim in self.hidden_dims:
            self.layers.append(nn.Linear(last_dim, hidden_dim))
            last_dim = hidden_dim

        self.classifier = nn.Linear(last_dim, 1)

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.classifier(x)

    def architecture_text(self):
        return " -> ".join([str(self.input_size)] + [str(dim) for dim in self.hidden_dims] + ["1"])

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def mutate_width(self, step_size=8, layer_index=0):
        if layer_index < 0 or layer_index >= len(self.hidden_dims):
            return False

        old_hidden_dims = self.hidden_dims[:]
        new_hidden_dims = self.hidden_dims[:]
        new_hidden_dims[layer_index] += step_size

        old_layers = self.layers
        old_classifier = self.classifier

        self.hidden_dims = new_hidden_dims
        self._rebuild_network()

        with torch.no_grad():
            for idx in range(len(old_layers)):
                old_out = old_layers[idx].out_features
                old_in = old_layers[idx].in_features

                self.layers[idx].weight[:old_out, :old_in] = old_layers[idx].weight
                self.layers[idx].bias[:old_out] = old_layers[idx].bias

            old_classifier_in = old_classifier.in_features
            self.classifier.weight[:, :old_classifier_in] = old_classifier.weight
            self.classifier.bias.copy_(old_classifier.bias)

        timestamp = datetime.now().strftime("%H:%M:%S")
        self.mutation_log.append(
            f"[{timestamp}] Width mutation on layer {layer_index + 1}: {old_hidden_dims[layer_index]} -> {new_hidden_dims[layer_index]}"
        )
        return True

    def mutate_depth(self):
        new_layer_dim = self.hidden_dims[-1] if self.hidden_dims else 16

        old_layers = self.layers
        old_classifier = self.classifier

        self.hidden_dims.append(new_layer_dim)
        self._rebuild_network()

        with torch.no_grad():
            for idx in range(len(old_layers)):
                self.layers[idx].weight.copy_(old_layers[idx].weight)
                self.layers[idx].bias.copy_(old_layers[idx].bias)

            inserted_index = len(self.layers) - 1
            self.layers[inserted_index].weight.zero_()
            rows = self.layers[inserted_index].weight.shape[0]
            cols = self.layers[inserted_index].weight.shape[1]
            diag = min(rows, cols)
            self.layers[inserted_index].weight[:diag, :diag] = torch.eye(diag)
            self.layers[inserted_index].bias.zero_()

            old_classifier_in = old_classifier.in_features
            self.classifier.weight[:, :old_classifier_in] = old_classifier.weight
            self.classifier.bias.copy_(old_classifier.bias)

        timestamp = datetime.now().strftime("%H:%M:%S")
        self.mutation_log.append(
            f"[{timestamp}] Depth mutation: added hidden layer with {new_layer_dim} neurons"
        )
        return True
