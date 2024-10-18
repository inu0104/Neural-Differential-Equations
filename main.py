import torch
import torch.nn as nn
import torchdiffeq

class AttentiveNCDE(nn.Module):
    def __init__(self, input_dim, hidden_dim, attention_dim, output_dim):
        super(AttentiveNCDE, self).__init__()
        # Bottom NCDE for attention
        self.attention_func = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        self.attention_gru = nn.GRUCell(input_dim, hidden_dim)
        # Top NCDE for time-series processing
        self.func = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, time_points, initial_state, input_series):
        h = initial_state
        for t in range(len(time_points) - 1):
            delta_t = time_points[t + 1] - time_points[t]
            # Bottom NCDE for calculating attention
            h = self.attention_gru(input_series[t], h)
            attention = self.attention_func(h)
            # Element-wise multiplication of attention and input
            modified_input = input_series[t] * attention
            # Top NCDE for evolving hidden vector
            h = torchdiffeq.odeint(lambda t, x: self.func(x), h, torch.tensor([0, delta_t]))[-1]
        # Generate final output
        output = self.output_layer(h)
        return output

class SimpleNCDE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNCDE, self).__init__()
        # Simple NCDE without attention
        self.func = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.gru = nn.GRUCell(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, time_points, initial_state, input_series):
        h = initial_state
        for t in range(len(time_points) - 1):
            delta_t = time_points[t + 1] - time_points[t]
            h = self.gru(input_series[t], h)
            h = torchdiffeq.odeint(lambda t, x: self.func(x), h, torch.tensor([0, delta_t]))[-1]
        output = self.output_layer(h)
        return output
