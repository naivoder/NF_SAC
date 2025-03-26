import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal, TanhTransform

from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform


class FlowPolicyNetwork(nn.Module):
    def __init__(
        self,
        input_shape,
        n_actions,
        h1_size,
        h2_size,
        n_flows=2,
        learning_rate=3e-4,
        min_action=-1,
        max_action=1,
        chkpt_path="weights/flow_actor.pt",
    ):
        super(FlowPolicyNetwork, self).__init__()
        self.h1_size = h1_size
        self.h2_size = h2_size
        self.n_actions = n_actions
        self.min_action = min_action
        self.max_action = max_action
        self.checkpoint_path = chkpt_path

        # Base network for generating mean/log_std
        self.base_net = nn.Sequential(
            nn.Linear(*input_shape, h1_size),
            nn.ReLU(),
            nn.Linear(h1_size, h2_size),
            nn.ReLU(),
        )
        self.mean_layer = nn.Linear(h2_size, n_actions)
        self.log_std_layer = nn.Linear(h2_size, n_actions)

        # Build a CompositeTransform of multiple MaskedAffineAutoregressiveTransform
        transforms = []
        for _ in range(n_flows):
            transforms.append(
                MaskedAffineAutoregressiveTransform(
                    features=n_actions, hidden_features=h2_size
                )
            )
        composite_transform = CompositeTransform(transforms)

        # Build the flow as a distribution
        self.flow = Flow(
            transform=composite_transform,
            distribution=StandardNormal(shape=[n_actions]),
        )

        # Optimizer and device
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        # Action rescaling
        self.action_scale = torch.FloatTensor(
            (self.max_action - self.min_action) / 2.0
        ).to(self.device)
        self.action_bias = torch.FloatTensor(
            (self.max_action + self.min_action) / 2.0
        ).to(self.device)

    def forward(self, state):
        """Compute mean and log_std from the base network."""
        base_out = self.base_net(state)
        mean = self.mean_layer(base_out)
        log_std = torch.clamp(self.log_std_layer(base_out), -20, 2)
        return mean, log_std

    def sample_action(self, state):
        """Sample an action using the base distribution + normalizing flow + tanh squashing."""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        base_dist = Normal(mean, std)
        # Sample from base dist
        z = base_dist.rsample()

        # Pass z through the flow transforms
        action_flow, log_det = self.flow._transform(z)

        # Apply tanh to bound actions to [-1, 1]
        action = torch.tanh(action_flow)
        action_scaled = action * self.action_scale + self.action_bias

        # Compute log-prob of final actions:
        # 1) Base distribution log prob
        log_prob = base_dist.log_prob(z).sum(dim=-1, keepdim=True)
        # 2) Subtract log-determinant of flow transforms
        log_prob -= log_det.view(-1, 1)
        # 3) Subtract log-derivative from tanh transform
        #    (the usual correction for a TanhTransform if we were using TransformedDistribution)
        log_prob -= torch.log(self.action_scale * (1 - action.pow(2)) + 1e-6).sum(
            dim=-1, keepdim=True
        )

        return action_scaled, log_prob

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_path)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_path))
