import src.singlelayer_gpt as gpt
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# find device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE =", device)

# set default device to cuda if available
torch.set_default_device(device)

# parameters
vocab_size = 4092
embed_dim = 2048
num_heads = 2
ff_hidden_layer = 2 * embed_dim
dropout = 0.1
context_length = 256
batch_size = 16

# initialise the model
model = gpt.TransformerDecoder(vocab_size, embed_dim, num_heads, ff_hidden_layer, dropout, device=device).to(device)

# create batch size and context length input tensor
input_tensor = torch.randint(0, vocab_size, (context_length, batch_size)).to(device)

output = model(input_tensor)

# get predicted word indices with argmax
predicted_indices = output.argmax(dim=-1)

# show trainable parameters
print(f"The model has {gpt.count_params(model):,} trainable parameters")

# convert the log probabilities to probabilities
distribution = torch.exp(output[0, 0, :])
distribution = distribution.detach().cpu().numpy()


