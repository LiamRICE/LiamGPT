import src.gpt as gpt
import matplotlib.pyplot as plt
import seaborn as sns

# parameters
vocab_size = 1000
d_model = 512
num_heads = 1
ff_hidden_layer = 2 * d_model
dropout = 0.1
num_layers = 10
context_length = 50
batch_size = 1

# create mask
mask = gpt.generate_square_mask(size=5)

plt.figure(figsize=(5,5))
sns.heatmap(mask, cmap="crest", cbar=False, square=True)
plt.title("Mask for Transformer Decoder")
plt.show()



