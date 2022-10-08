import torch
from torch import nn

model = nn.Sequential(
	nn.Linear(28 * 28, 256),
	nn.ReLU(),
	nn.Linear(256, 10),
)

model = torch.jit.script(model)
print(model.code)
model.save("mnist-mlp.pt")