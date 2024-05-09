import src.dataset_tools as dataset_tools
import torch

# set seed
torch.manual_seed(42)

# find device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE =", device)

# set default device to cuda if available
torch.set_default_device(device)

# hyperparameters
eval_interval = 100
max_iters = 500
learning_rate = 1e-4
batch_size = 3
block_size = 16

# model parameters
vocab_size = 16384
embed_dim = 2048
num_heads = 8
ff_hidden_layer = 8 * embed_dim
dropout = 0.1
num_layers = 16
context_length = 1024

text = dataset_tools.read_file("input.txt")
n_vocab, encode, decode = dataset_tools.byte_pair_encoding()

x, y = dataset_tools.get_batch(encode(text), block_size, batch_size, device)

dataset_tools.get_context_target(x, y, block_size, batch_size)

















