import torch
from torchviz import make_dot
from models.arch import CifarModel

# Define the CNN model
model = CifarModel()

# Create a dummy input tensor matching the input shape (MNIST: 1 channel, 28x28)
dummy_input = torch.randn(1, 3, 32, 32)

# Forward pass to generate the output
output = model(dummy_input)

# Generate a layered visual representation
model_graph = make_dot(output, params=dict(model.named_parameters()))

# Save the visualization as an image file (PNG format)
model_graph.render("cnn_model", format="png")  # Saves as cnn_model.png
print("Model visualization saved as cnn_model.png")
