import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_size=64, fc2_size=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        # KAE: define the first fully connected layer, mapping state_size to the first fc1 size (default=64)
        self.fc1 = nn.Linear(state_size, fc1_size)
        # KAE: define the second fully connected layer, mapping fc1 size (default=64) to the  fc2 size
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        # KAE: define the third connected layer, mapping fc2 size to the  action size
        self.fc3 = nn.Linear(fc2_size, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
#       input states into the top fc layer, output actions into the bottom fc layer with one fc1_size x fc2_size layer inbetween
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
