import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd

# Load the NumPy array from the .npy file
# data = np.load('textchunksConsulting.npy')
data = np.load('bundle/chunks.npy')
# data = np.load('output/test_chunks_npy/test.npy')

# Convert the NumPy array to a pandas DataFrame
df = pd.DataFrame(data)
print("top 5 rows:")
print(df.head())

df['Score'] = np.random.randint(1, 6, size=len(df)) #not present originally

# df = pd.read_csv('textchunksConsulting.npy') (original comment out)
# matrix = df.ada_embedding.apply(eval).to_list()
matrix = df

# Create a t-SNE model and transform the data
tsne = TSNE(n_components=2, perplexity=15, random_state=42, init='random', learning_rate=200)
vis_dims = tsne.fit_transform(matrix.drop('Score', axis=1))

# colors = ["turquoise", "red", "darkorange", "gold", "darkgreen"]
colors = ["turquoise"]

# x = [x for x,y in vis_dims]
# y = [y for x,y in vis_dims]

x = vis_dims[:, 0]
y = vis_dims[:, 1]

color_indices = df.Score.values - 1
# color_indices = df['Score'] - 1

colormap = matplotlib.colors.ListedColormap(colors)
plt.scatter(x, y, c=color_indices, cmap=colormap, alpha=0.8)
plt.title("Consulting text Corpus using t-SNE")
plt.show()