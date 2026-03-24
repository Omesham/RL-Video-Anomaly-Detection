import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class DQN(nn.Module):
    def __init__(self, input_dim, num_actions):
        super().__init__()
        self.net = nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, num_actions)
    )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, input_dim, num_actions=2, lr = 5e-4, gamma=0.9, epsilon=0.2):
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_actions = num_actions

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DQN(input_dim, num_actions).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)

        state = torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def update(self, state, action, reward, next_state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device).unsqueeze(0)

        q_values = self.model(state)
        next_q_values = self.model(next_state)

        target = q_values.clone().detach()

        # Bellman update
        target[0, action] = reward + self.gamma * torch.max(next_q_values)

        loss = self.loss_fn(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # ?? NEW: anomaly score
    def anomaly_score(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)
        q_values = self.model(state)
        
        # low confidence ? anomaly
        return -torch.max(q_values).item()