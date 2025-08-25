import math

import torch
import torch.nn as nn
from addict import Dict


class FMWrapper:

    def __init__(self, model, guidance):
        self.model = model
        self.guidance = guidance

    def forward(self, x, c, t):
        c = c.unsqueeze(1)
        # Our model expects a batch, time and feature dimension, so we add it here
        t = t.clone().detach().unsqueeze(0).unsqueeze(0).expand(x.size(0), -1)
        if x.squeeze().dim() == 1:
            inp1 = torch.cat([x, c, t], dim=1)
            inp2 = torch.cat([x, torch.zeros_like(c), t], dim=1)
        else:
            inp1 = torch.cat([x.squeeze(), c, t], dim=1)
            inp2 = torch.cat([x.squeeze(), torch.zeros_like(c), t], dim=1)
        with torch.no_grad():
            out1 = self.model(inp1)  # conditional model
            out2 = self.model(inp2)  # unconditional model
        return (1 - self.guidance) * out2 + self.guidance * out1


class CustomEmbeddingOutModule(nn.Module):
    """Custom output module for conditional models with learnable embeddings."""

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        e_size,
        activation,
        get_normalization_layer,
        num_hidden_layers=5,
        spectral_norm=False,
        dropout=0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # Input layer
        linear = nn.Linear(input_size, hidden_size)
        self.layers.append(self._apply_spectral_norm(linear, spectral_norm))
        if get_normalization_layer:
            self.norms.append(get_normalization_layer(hidden_size))
        if dropout > 0.0:
            self.dropouts.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(num_hidden_layers):
            linear = nn.Linear(hidden_size + e_size, hidden_size)
            self.layers.append(self._apply_spectral_norm(linear, spectral_norm))
            if get_normalization_layer:
                self.norms.append(get_normalization_layer(hidden_size))
            if dropout > 0.0:
                self.dropouts.append(nn.Dropout(dropout))

        # Output layer
        linear = nn.Linear(hidden_size + e_size, output_size)
        self.layers.append(self._apply_spectral_norm(linear, spectral_norm))

        self.activation = activation
        self.use_norm = bool(get_normalization_layer)
        self.use_dropout = dropout > 0.0

    def _apply_spectral_norm(self, layer, spectral_norm):
        return nn.utils.spectral_norm(layer) if spectral_norm else layer

    def forward(self, x, e):
        for i, layer in enumerate(self.layers[:-1]):
            # Note that the embedding is already attached to x in the first round,
            # hence we concatenate e to x in the end of the forward pass
            x = layer(x)
            if self.use_norm:
                x = self.norms[i](x)
            x = self.activation(x)
            if self.use_dropout:
                x = self.dropouts[i](x)
            x = torch.cat([x, e], dim=1)

        x = self.layers[-1](x)
        return x


class VectorFieldModel(nn.Module):
    """Vector field model for ODEs."""

    def __init__(
        self,
        data_dim: int = 2,
        x_latent_dim: int = 128,
        time_embed_dim: int = 128,
        cond_embed_dim: int = 128,
        conditional_model: bool = False,
        embedding_type: str = None,
        n_classes: int = 1,
        label_list: list = None,
        step_scale: int = 1000,
        normalization: str = None,
        activation: str = "SELU",
        affine_transform: bool = False,
        sum_time_embed: bool = False,
        sum_cond_embed: bool = False,
        max_norm_embedding: bool = True,
        num_out_layers: int = 3,
        spectral_norm: bool = False,
        dropout: float = 0.0,
        conditional_bias: bool = False,
        keep_constants: bool = False,
        **kwargs,
    ):
        super().__init__()

        if kwargs:
            print(f"Unknown arguments: {kwargs.keys()}")

        self.x_latent_dim = x_latent_dim
        self.time_embed_dim = time_embed_dim
        self.cond_emebd_dim = cond_embed_dim
        self.step_scale = step_scale
        self.conditional_model = conditional_model
        self.n_classes = n_classes
        self.normalization = normalization
        self.affine_transform = affine_transform
        self.sum_time_embed = sum_time_embed
        self.sum_cond_embed = sum_cond_embed
        self.max_norm_embedding = max_norm_embedding
        self.embedding_type = embedding_type
        self.label_list = label_list
        self.num_out_layers = num_out_layers
        self.spectral_norm = spectral_norm
        self.dropout = dropout

        if self.conditional_model:
            self.lookup = False
            if self.embedding_type not in ["free", "ohe", "value", "value-all"]:
                raise ValueError(f"Unsupported embedding type: {embedding_type}")

            if self.embedding_type == "free":
                if not len(label_list) == n_classes:
                    raise ValueError(
                        "Number of classes must match the length of label_list"
                    )
                # Ensure that label list is a list with integers ranging from 1 to n_classes
                # If label_list are not integers from 0 to n_classes, we need to map them
                if set(range(1, len(label_list) + 1)) != set(label_list):
                    self.lookup = True
                    self.mapping = {
                        label: (i + 1) for i, label in enumerate(sorted(label_list))
                    }
                # +1 for the class 0
                # We could set padding_idx=0, but we learn the embedding instead
                self.embed_cond = nn.Embedding(
                    n_classes + 1, self.cond_emebd_dim, max_norm=self.max_norm_embedding
                )
            elif self.embedding_type == "ohe":
                if self.cond_emebd_dim != n_classes:
                    raise ValueError(
                        "One-hot encoding requires cond_embed_dim == n_classes"
                    )
                self.cond_emebd_dim = n_classes

        # Define non-linear activation
        if activation == "SELU":
            self.activation = torch.nn.SELU()
        elif activation == "ReLU":
            self.activation = torch.nn.ReLU()
        elif activation == "LeakyReLU":
            self.activation = torch.nn.LeakyReLU(negative_slope=0.02)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # X-module
        self.x_module = nn.Sequential(
            nn.Linear(data_dim, x_latent_dim),
            self._get_normalization_layer(x_latent_dim),
            self.activation,
            nn.Linear(x_latent_dim, x_latent_dim),
            self._get_normalization_layer(x_latent_dim),
            self.activation,
            nn.Linear(x_latent_dim, x_latent_dim),
        )

        # Time-module
        if self.sum_time_embed:
            if self.x_latent_dim != self.time_embed_dim:
                self.t_module = nn.Sequential(
                    nn.Linear(self.time_embed_dim, x_latent_dim),
                    self._get_normalization_layer(x_latent_dim),
                    self.activation,
                    nn.Linear(x_latent_dim, x_latent_dim),
                )
            else:
                self.t_module = nn.Identity()

        # Conditional-module
        if self.conditional_model:
            if self.sum_cond_embed:
                if self.embedding_type == "ohe":
                    self.e_module = nn.Identity()
                elif self.embedding_type == "free":
                    if self.x_latent_dim != self.cond_emebd_dim:
                        self.e_module = nn.Sequential(
                            nn.Linear(self.cond_emebd_dim, x_latent_dim),
                            self._get_normalization_layer(x_latent_dim),
                            self.activation,
                            nn.Linear(x_latent_dim, x_latent_dim),
                        )
                    else:
                        self.e_module = nn.Identity()
                else:
                    self.e_module = nn.Identity()
            else:
                if self.embedding_type in ["value", "value-all"]:
                    self.e_module = nn.Linear(
                        1, self.cond_emebd_dim, bias=conditional_bias
                    )
                    if keep_constants:
                        # Initialize the weights to ones and the bias to zeros
                        # And freeze the weights
                        self.e_module.weight.data.fill_(1.0)
                        self.e_module.weight.requires_grad = False
                        if conditional_bias:
                            self.e_module.bias.data.fill_(0.0)
                            self.e_module.bias.requires_grad = False
                    # Use Spectral Norm if specified
                    if self.spectral_norm:
                        self.e_module = nn.utils.spectral_norm(self.e_module)

        # Output-module
        if not self.conditional_model:
            # We only combine time and x domain
            input_to_output = (
                x_latent_dim if self.sum_time_embed else x_latent_dim + time_embed_dim
            )
        else:
            # We combine time, x domain and conditional domain
            if self.sum_cond_embed & self.sum_time_embed:
                input_to_output = x_latent_dim
            elif self.sum_cond_embed & ~self.sum_time_embed:
                input_to_output = x_latent_dim + time_embed_dim
            elif ~self.sum_cond_embed & self.sum_time_embed:
                input_to_output = x_latent_dim + cond_embed_dim
            else:
                input_to_output = x_latent_dim + time_embed_dim + cond_embed_dim

        if (self.embedding_type == "value-all") & self.conditional_model:
            self.out_module = CustomEmbeddingOutModule(
                input_to_output,
                x_latent_dim,
                data_dim,
                self.cond_emebd_dim,
                self.activation,
                self._get_normalization_layer,
                self.num_out_layers,
                self.spectral_norm,
                self.dropout,
            )
        else:
            self.out_module = self._build_out_module(
                input_to_output, x_latent_dim, data_dim, spectral_norm, dropout
            )

    def _build_out_module(
        self, input_size, hidden_size, output_size, spectral_norm, dropout
    ):
        layers = []
        for i in range(self.num_out_layers):
            if i == 0:
                linear = nn.Linear(input_size, hidden_size)
            elif i == self.num_out_layers - 1:
                linear = nn.Linear(hidden_size, output_size)
            else:
                linear = nn.Linear(hidden_size, hidden_size)

            if spectral_norm:
                linear = nn.utils.spectral_norm(linear)

            layers.append(linear)

            if (
                i < self.num_out_layers - 1
            ):  # Don't add normalization, activation, and dropout after the last layer
                if self.normalization is not None:
                    layers.append(self._get_normalization_layer(hidden_size))
                layers.append(self.activation)
                if dropout > 0.0:  # Only add dropout if the rate is greater than 0
                    layers.append(nn.Dropout(dropout))

        return nn.Sequential(*layers)

    def _get_normalization_layer(self, dim):
        if self.normalization is None:
            return nn.Identity()
        elif self.normalization.lower() == "batchnorm":
            return nn.BatchNorm1d(dim, affine=self.affine_transform)
        elif self.normalization.lower() == "layernorm":
            return nn.LayerNorm(dim, elementwise_affine=self.affine_transform)
        else:
            raise ValueError(f"Unsupported normalization: {self.normalization}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x[:, -1].squeeze()
        x = x[:, :-1].float()

        t_out = timestep_embedding(t * self.step_scale, self.time_embed_dim)
        if self.sum_time_embed:
            t_out = self.t_module(t_out)

        if not self.conditional_model:
            x_out = self.x_module(x)
            if self.sum_time_embed:
                x_total = x_out + t_out
            else:
                x_total = torch.cat([x_out, t_out], dim=1)
            out = self.out_module(x_total)

        else:
            # Conditional-time model
            classes = x[:, -1]
            # Replace classes with lookup if necessary
            if self.lookup:
                classes = map_tensor_values(classes, self.mapping)
            classes = classes.long()
            x = x[:, :-1]

            # X-embedding
            x_out = self.x_module(x)

            if self.embedding_type == "ohe":
                e_out = torch.nn.functional.one_hot(classes, num_classes=self.n_classes)
            elif self.embedding_type == "free":
                e_out = self.embed_cond(classes)
            elif self.embedding_type in ["value", "value-all"]:
                e_out = self.e_module(classes.float().unsqueeze(1))

            # Time-embedding
            if self.sum_time_embed:
                x_total = x_out + t_out
            if self.sum_cond_embed:
                x_total = x_out + self.e_module(e_out)

            if not self.sum_time_embed:
                x_total = torch.cat([x_out, t_out], dim=1)
            if not self.sum_cond_embed:
                x_total = torch.cat([x_total, e_out], dim=1)

            if self.embedding_type == "value-all":
                out = self.out_module(x_total, e_out)
            else:
                out = self.out_module(x_total)

        return out


class MultiVectorFieldModel(nn.Module):

    def __init__(
        self,
        model_list,
        data_dim=2,
        x_latent_dim=128,
        time_embed_dim=None,
        cond_embed_dim=None,
        conditional_model=None,
        embedding_type=None,
        n_classes=None,
        label_list=None,
        normalization=None,
        activation=None,
        affine_transform=None,
        sum_time_embed=None,
        sum_cond_embed=None,
        max_norm_embedding=None,
        num_out_layers=None,
        spectral_norm=None,
        dropout=None,
        conditional_bias=None,
        keep_constants=None,
    ):
        super().__init__()

        self.model_list = model_list

        # We fit one model for each time interval and each condtion
        num_models = 0
        for c, times in self.model_list.items():
            num_models += len(times) - 1

        # Create a ModuleList to store N independent VectorFieldModel instances
        self.models = nn.ModuleList(
            [
                SimpleModel(data_dim, hidden=6, time_embed_dim=6)
                for _ in range(num_models + 1)
            ]
        )

        self.mapping_idx_to_tc = Dict()
        idx = 1
        for c, t_list in self.model_list.items():
            for i in range(len(t_list) - 1):
                self.mapping_idx_to_tc[c][(t_list[i], t_list[i + 1])] = idx
                idx += 1

        self.mapping_idx_to_tc[0][(0, 1)] = 0

    def _get_model_index(self, t, c):
        for (t1, t2), idx in self.mapping_idx_to_tc[c].items():
            if t1 <= t < t2:
                return idx
        # if t > t2, return the last model
        if t >= t2:
            return idx
        raise ValueError(f"No model found for time {t} and condition {c}")

    def forward(self, x):
        data = x[:, :-2]
        cond = x[:, -2]  # Second last column is the condition indicator
        t = x[:, -1]  # Last column is the time indicator

        idxs = []
        outputs = []
        for c, v in self.mapping_idx_to_tc.items():
            for from_to_, model_id in v.items():
                # Find all indices where the time is between from_ and to_
                idx = (from_to_[0] <= t) & (t < from_to_[1]) & (c == cond)
                if idx.sum() == 0:
                    continue
                model_output = self.models[model_id](
                    torch.cat([data[idx], t[idx].unsqueeze(1)], dim=1)
                )
                outputs.append(model_output)
                idxs.append(idx)
        # If all values in t are 1.0, then use last model
        if t.eq(1.0).all().item():
            model_output = self.models[-1](torch.cat([data, t.unsqueeze(1)], dim=1))
            outputs.append(model_output)
            idxs.append(t.eq(1.0))

        output = torch.cat(outputs, dim=0)

        # Resort the rows of output such that each row is at the same position as it was in the input
        # Convert boolean masks to indices
        idxs = [torch.where(idx)[0] for idx in idxs]
        idxs_flat = torch.cat(idxs)
        resorted_output = torch.zeros_like(output)
        resorted_output[idxs_flat] = output

        return resorted_output

    def __getitem__(self, idx):
        return self.models[idx]

def map_tensor_values(tensor, mapping_dict):
    device = tensor.device
    keys = torch.tensor(list(mapping_dict.keys()), device=device)
    values = torch.tensor(list(mapping_dict.values()), device=device)

    # Find the indices of the closest keys
    indices = torch.bucketize(tensor.flatten(), keys)
    mapped_values = values[indices]

    return mapped_values.reshape(tensor.shape)


def timestep_embedding(timesteps, dim, max_period=10000):
    """Create sinusoidal timestep embeddings.
    """
    if timesteps.dim() == 0:
        timesteps = timesteps.unsqueeze(0)
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
