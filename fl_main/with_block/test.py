# from utils.nav_utils import GridWorld, ReplayBuffer

# env = GridWorld(grid_size=10, n_static=15, n_dynamic=5)
# replay_buffer = ReplayBuffer(capacity=10000)
# env.reset()
# env.reset()

# env.render(save_image=True)

import torch
from torchsummary import summary
from torchviz import make_dot
from models.arch import DisasterModel, DQN

def print_model_summaries(device):
    # For DisasterModel, the expected input is (3, 222, 222)
    print("=== DisasterModel Summary ===")
    disaster_model = DisasterModel()
    disaster_model.to(device)
    summary(disaster_model, input_size=(3, 222, 222))
    
    # For DQN, assuming input_channels=4 and grid size is 10x10
    print("\n=== DQN Summary ===")
    dqn_model = DQN(input_channels=4, n_actions=4)
    dqn_model.to(device)
    summary(dqn_model, input_size=(4, 10, 10))

# --------------------- Architecture Diagrams ---------------------
def create_architecture_diagrams():
    # Create a diagram for DisasterModel
    disaster_model = DisasterModel()
    x_disaster = torch.randn(1, 3, 222, 222)
    y_disaster = disaster_model(x_disaster)
    dot_disaster = make_dot(y_disaster, params=dict(disaster_model.named_parameters()))
    dot_disaster.format = 'png'
    dot_disaster.render('disaster_model_arch', cleanup=True)
    print("Saved DisasterModel architecture diagram as 'disaster_model_arch.png'")

    # Create a diagram for DQN
    dqn_model = DQN(input_channels=4, n_actions=4)
    x_dqn = torch.randn(1, 4, 10, 10)
    y_dqn = dqn_model(x_dqn)
    dot_dqn = make_dot(y_dqn, params=dict(dqn_model.named_parameters()))
    dot_dqn.format = 'png'
    dot_dqn.render('dqn_model_arch', cleanup=True)
    print("Saved DQN architecture diagram as 'dqn_model_arch.png'")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    print_model_summaries(device=device)
    create_architecture_diagrams()