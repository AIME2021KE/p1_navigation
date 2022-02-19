import numpy as np
import random
from collections import namedtuple, deque

from dqn_model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#NOTE 2/18/2022 KAE: this class definition is essentially what was defined 
#  in the DQN mini-projectt with only minor modifications
# Sections without comments were pre-provided 
class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Networks -- both a local (internal) and a target)
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        # convert arraylike input state into unsqueezed value and post to device if available.
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        # evaluate the local network
        self.qnetwork_local.eval()
        # The with key word is new to me but it provides a cleaner implementation of try / finally by cleaning up on errors: 
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
        #NOTE: the original code provided in this function failed due to 
        #  lack of casting it into int32, which was added for the navigation project
        #  original retained as a comment
#            return np.argmax(action_values.cpu().data.numpy())
            return np.int32(np.argmax(action_values.cpu().data.numpy()))
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        # unpack the experiences tuple
        states, actions, rewards, next_states, dones = experiences

        ## TODONE KAE: compute and minimize the loss
        "*** YOUR CODE HERE ***"
        # get the next qtarget values obj based on next states:
        # Add some print statements so we can get insight into the various objects sizes, types, etc
        qt_nextobj = self.qnetwork_target(next_states)
        # we want a true copy of the qt_nextobj tensor, using detach to get it
        qt_nexttensor = qt_nextobj.detach()
        # next is tricky. We use max(1) to get the maximum in the tensor in dim=1 (implied?!)
        #  and we take the first value [0] in case more than one and finally
        #  unsqueeze the matrix to get a single value in a list of values. Including example from torch:
         # Thus the unsqueeze(1) gives us a list of lists as above, with a single (set) of values 
         # corresponding to the 1st dim of the tensor
        qt_next = qt_nexttensor.max(dim=1)[0].unsqueeze(1)
    
        # Next we get our qt but assuming we have many states, actions, rewards, dones, etc
        qt = rewards + (gamma * qt_next * (1 - dones))
        # then get our qlearning local network vlaues based on the current states
        qexpobj = self.qnetwork_local(states)
        # finally gather the data along the actions dimension
        qexp = qexpobj.gather(1,actions)
        
        # calculate loss (to be minimized). Note that torch.nn.functional is now F (not f as in model)
        # get the differences between expected and the acthived (target)
        loss = F.mse_loss(qexp, qt)
        
        # minimize loss by a) zero the gradients, b) prop loss backwards, and c) perform a step 
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        # unpack the local and target model parameters and copy them back into the target param data
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)