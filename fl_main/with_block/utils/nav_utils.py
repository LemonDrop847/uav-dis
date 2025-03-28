import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from models.arch import DQN
import matplotlib.pyplot as plt
import matplotlib.colors as colors

class GridWorld:
    """
    UAV navigation environment using a grid-based system.

    Grid Values:
    - 0: Free space
    - 1: Static obstacle
    - 2: UAV position
    - 3: Goal position
    - 4: Dynamic obstacle (can move)

    UAV Actions:
    - 0: Move Up
    - 1: Move Down
    - 2: Move Left
    - 3: Move Right
    """
    def __init__(self, grid_size=10, n_static=15, n_dynamic=5):
        self.grid_size = grid_size
        self.n_static = n_static
        self.n_dynamic = n_dynamic
        self.action_space = [0, 1, 2, 3]  # 0: up, 1: down, 2: left, 3: right
        self.n_actions = len(self.action_space)
        self.reset()

    def reset(self):
        # Create empty state with 4 channels: static obstacles, dynamic obstacles, agent, goal.
        self.state = np.zeros((self.grid_size, self.grid_size, 4), dtype=np.float32)
        # Place static obstacles
        self.static_positions = set()
        while len(self.static_positions) < self.n_static:
            pos = (np.random.randint(self.grid_size), np.random.randint(self.grid_size))
            if pos in [(0, 0), (self.grid_size - 1, self.grid_size - 1)]:
                continue
            self.static_positions.add(pos)
        for pos in self.static_positions:
            self.state[pos[0], pos[1], 0] = 1.0
        # Place dynamic obstacles
        self.dynamic_positions = []
        while len(self.dynamic_positions) < self.n_dynamic:
            pos = (np.random.randint(self.grid_size), np.random.randint(self.grid_size))
            if pos in self.static_positions or pos in [(0, 0), (self.grid_size - 1, self.grid_size - 1)]:
                continue
            self.dynamic_positions.append(pos)
        for pos in self.dynamic_positions:
            self.state[pos[0], pos[1], 1] = 1.0
        # Set agent and goal positions
        self.agent_pos = (0, 0)
        self.goal_pos = (self.grid_size - 1, self.grid_size - 1)
        self.state[self.agent_pos[0], self.agent_pos[1], 2] = 1.0
        self.state[self.goal_pos[0], self.goal_pos[1], 3] = 1.0
        self.steps = 0
        return self.get_state()

    def get_state(self):
        # Transpose to (C, H, W) format for CNN input
        return np.transpose(self.state, (2, 0, 1))

    def step(self, action):
        self.steps += 1
        old_dist = np.linalg.norm(np.array(self.agent_pos) - np.array(self.goal_pos))
        self.state[self.agent_pos[0], self.agent_pos[1], 2] = 0.0  # Remove agent marker

        new_pos = list(self.agent_pos)
        if action == 0:  # up
            new_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 1:  # down
            new_pos[0] = min(self.grid_size - 1, self.agent_pos[0] + 1)
        elif action == 2:  # left
            new_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 3:  # right
            new_pos[1] = min(self.grid_size - 1, self.agent_pos[1] + 1)
        new_pos = tuple(new_pos)

        done = False
        # Collision with obstacles
        if self.state[new_pos[0], new_pos[1], 0] == 1 or self.state[new_pos[0], new_pos[1], 1] == 1:
            reward = -10
            done = True
        elif new_pos == self.goal_pos:
            reward = 10
            done = True
        else:
            new_dist = np.linalg.norm(np.array(new_pos) - np.array(self.goal_pos))
            reward = old_dist - new_dist  # Shaping reward

        if not done:
            self.agent_pos = new_pos
        self.state[self.agent_pos[0], self.agent_pos[1], 2] = 1.0  # Set new agent position
        self._move_dynamic_obstacles()
        return self.get_state(), reward, done

    def _move_dynamic_obstacles(self):
        # Update dynamic obstacles positions
        for pos in self.dynamic_positions:
            self.state[pos[0], pos[1], 1] = 0.0
        new_dynamic = []
        for pos in self.dynamic_positions:
            moves = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
            dx, dy = random.choice(moves)
            new_pos = (np.clip(pos[0] + dx, 0, self.grid_size - 1),
                       np.clip(pos[1] + dy, 0, self.grid_size - 1))
            if new_pos in self.static_positions or new_pos == self.goal_pos or new_pos == self.agent_pos:
                new_pos = pos  # Stay if blocked
            new_dynamic.append(new_pos)
        self.dynamic_positions = new_dynamic
        for pos in self.dynamic_positions:
            self.state[pos[0], pos[1], 1] = 1.0

    def render(self, save_image=False, filename="gridworld.png"):
        """
        Renders the grid visually using matplotlib.
        - Static obstacles (S) are shown in black.
        - Dynamic obstacles (D) are shown in red.
        - The UAV (agent, A) is shown in blue.
        - The goal (G) is shown in green.
        If save_image is True, the image is saved to the specified filename.
        """
        # Create a numeric grid:
        # 0: free space, 1: static obstacle, 2: dynamic obstacle, 3: agent, 4: goal.
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        for pos in self.static_positions:
            grid[pos[0], pos[1]] = 1
        for pos in self.dynamic_positions:
            grid[pos[0], pos[1]] = 2
        # Place the agent and goal (agent overrides any other value)
        a_i, a_j = self.agent_pos
        grid[a_i, a_j] = 3
        g_i, g_j = self.goal_pos
        grid[g_i, g_j] = 4

        # Define a custom colormap:
        cmap = colors.ListedColormap(["white", "black", "red", "blue", "green"])
        bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        plt.figure(figsize=(self.grid_size/2, self.grid_size/2))
        plt.imshow(grid, cmap=cmap, norm=norm)

        # Annotate cells with letters
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if grid[i, j] == 1:
                    text = "S"
                elif grid[i, j] == 2:
                    text = "D"
                elif grid[i, j] == 3:
                    text = "A"
                elif grid[i, j] == 4:
                    text = "G"
                else:
                    text = ""
                # Choose contrasting text color:
                txt_color = "white" if grid[i, j] in [1, 2] else "black"
                plt.text(j, i, text, ha="center", va="center", color=txt_color, fontsize=12)

        plt.xticks([])
        plt.yticks([])
        plt.title("GridWorld Environment")
        plt.tight_layout()
        if save_image:
            plt.savefig(filename)
            print(f"Saved grid image to {filename}")
        plt.show()


# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
    def push(self, transition):
        self.memory.append(transition)
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)
    

class DQNAgent:
    def __init__(self, input_channels, n_actions, device):
        self.n_actions = n_actions
        self.device = device
        self.policy_net = DQN(input_channels, n_actions).to(device)
        self.target_net = DQN(input_channels, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.memory = ReplayBuffer(10000)
        self.batch_size = 64
        self.gamma = 0.99

    def select_action(self, state, epsilon):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        if random.random() > epsilon:
            with torch.no_grad():
                q_values = self.policy_net(state)
                action = int(torch.argmax(q_values).item())
        else:
            action = random.randrange(self.n_actions)
        return action

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = list(zip(*transitions))
        state_batch = torch.tensor(np.array(batch[0]), dtype=torch.float32).to(self.device)
        action_batch = torch.tensor(batch[1], dtype=torch.int64).unsqueeze(1).to(self.device)
        reward_batch = torch.tensor(batch[2], dtype=torch.float32).unsqueeze(1).to(self.device)

        next_state_batch = torch.tensor(np.array([s for s in batch[3] if s is not None]), dtype=torch.float32).to(self.device)
        done_batch = torch.tensor(batch[4], dtype=torch.float32).unsqueeze(1).to(self.device)

        current_q_values = self.policy_net(state_batch).gather(1, action_batch)
        next_q_values = torch.zeros(self.batch_size, 1).to(self.device)
        non_final_mask = torch.tensor([s is not None for s in batch[3]], dtype=torch.bool).to(self.device)
        if non_final_mask.sum() > 0:
            next_q_values[non_final_mask] = self.target_net(next_state_batch).max(1)[0].unsqueeze(1)
        target_q_values = reward_batch + (self.gamma * next_q_values * (1 - done_batch))
        loss = nn.MSELoss()(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

class DoubleDQNAgent(DQNAgent):
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = list(zip(*transitions))
        state_batch = torch.tensor(np.array(batch[0]), dtype=torch.float32).to(self.device)
        action_batch = torch.tensor(batch[1], dtype=torch.int64).unsqueeze(1).to(self.device)
        reward_batch = torch.tensor(batch[2], dtype=torch.float32).unsqueeze(1).to(self.device)
        next_state_batch = torch.tensor(np.array([s for s in batch[3] if s is not None]), dtype=torch.float32).to(self.device)
        done_batch = torch.tensor(batch[4], dtype=torch.float32).unsqueeze(1).to(self.device)

        current_q_values = self.policy_net(state_batch).gather(1, action_batch)
        next_q_values = torch.zeros(self.batch_size, 1).to(self.device)
        non_final_mask = torch.tensor([s is not None for s in batch[3]], dtype=torch.bool).to(self.device)
        if non_final_mask.sum() > 0:
            next_state_values = self.policy_net(next_state_batch)
            best_actions = torch.argmax(next_state_values, dim=1, keepdim=True)
            target_next_values = self.target_net(next_state_batch)
            next_q_values[non_final_mask] = target_next_values.gather(1, best_actions)
        target_q_values = reward_batch + (self.gamma * next_q_values * (1 - done_batch))
        loss = nn.MSELoss()(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()