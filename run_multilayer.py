import src.multilayer_gpt as gpt
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
vocab_size = 16384
embed_dim = 2048
num_heads = 2
ff_hidden_layer = 8 * embed_dim
dropout = 0.1
num_layers = 16
context_length = 1024
batch_size = 1

# create input tensor
input_tensor = torch.randint(0, vocab_size, (context_length, batch_size)).to(device)

# initialise the model
model = gpt.MultiLayerTransformerDecoder(vocab_size, embed_dim, num_heads, ff_hidden_layer, dropout, num_layers, device)

print(f"the model has {gpt.count_params(model):,} trainable parameters")

output = model(input_tensor)

# Convert the log probabilities to probabilities for the first sequence in the batch and the first position in the sequence
distribution = torch.exp(output[0, 0, :])

# Convert the output tensor to numpy array
distribution = distribution.detach().cpu().numpy()

# Now plot the distribution
plt.figure(figsize=(12, 6))
plt.bar(np.arange(vocab_size), distribution)
plt.xlabel("Word Index")
plt.ylabel("Probability")
plt.title("Output Distribution over Vocabulary")
plt.show()
