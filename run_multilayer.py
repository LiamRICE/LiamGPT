import src.multilayer_gpt as gpt
import torch

# find device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE =", device)

# set default device to cuda if available
torch.set_default_device(device)

# parameters
vocab_size = 16384
embed_dim = 2048
num_heads = 2
ff_hidden_layer = 8 * embed_dim
dropout = 0.1
num_layers = 32
context_length = 1024
batch_size = 16

# create input tensor
