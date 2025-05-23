# Use this script to dynamically use different clustering and dimensionality reduction methods to generate cluster plots using input text chunks.

import numpy as np
import pandas as pd
import openai, os, re, glob, umap
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
from sklearn.mixture import GaussianMixture
import sys
import scipy.stats as stats

# Configuration parameters
# complete_run = 0: Load saved labels and generate visualizations
# complete_run = 1: Generate clustering and labels from scratch, then visualize
complete_run = 0

# Configuration
n_dim = 2 # Set n_dim=2 for 2D plotting; set n_dim=3 for 3D plotting
dim_red_flag = 2  # 1=PCA, 2=t-SNE, 3=UMAP
cluster_algo_flag = 2  # 1=Agglomerative Clustering, 2=KMeans
num_clusters = 5  # Number of clusters

# OpenAI API key
path =r"input/APIkey.txt"
openai.api_key  = open(path).read()
open(path).close()

# Load dataset
# Text chunks are stored in a CSV file, and precomputed embeddings are loaded from a .npy file
df = pd.read_csv('bundle/chunks-originaltext.csv')
text_chunks = df['Text'].tolist()
embeddings = np.load('bundle/chunks.npy', allow_pickle=True)

# Set dimension reduction model
if dim_red_flag == 1:
    dim_red_mod = PCA(n_components=n_dim)
elif dim_red_flag == 2:
    dim_red_mod = TSNE(n_components=n_dim, random_state=0, perplexity=50)
elif dim_red_flag == 3:
    dim_red_mod = umap.UMAP(n_components=n_dim, random_state=42)
else:
    print("Invalid 'dim_red_flag'. Exiting.")
    sys.exit()

# Reduce embeddings dimensions
reduced_embeddings = dim_red_mod.fit_transform(embeddings)

# Optimize the number of clusters using Gaussian Mixture BIC scores
bic_scores = []
n_components_range = range(1, 100)

for n in n_components_range:
    gmm = GaussianMixture(n_components=n)
    gmm.fit(reduced_embeddings)
    bic_scores.append(gmm.bic(reduced_embeddings))

optimal_n_components = n_components_range[np.argmin(bic_scores)]
print("optimal_n_components = "+str(optimal_n_components))

num_clusters = optimal_n_components  # Update number of clusters

# Set clustering algorithm
if cluster_algo_flag == 1:
    cluster_algo = AgglomerativeClustering(n_clusters=num_clusters)
elif cluster_algo_flag == 2:
    cluster_algo = KMeans(n_clusters=num_clusters, random_state=42)
else:
    print("Invalid 'cluster_algo_flag'. Exiting.")
    sys.exit()

# Apply clustering algorithm to embeddings
cluster_labels = cluster_algo.fit_predict(embeddings)

# Helper function to interact with OpenAI API
def llm_response_generator(chunk, type='text'): 
    if (type=='text'):
        pre_text = "Provide a concise, meaningful label of not more than 8 words for the following text:"
    else:
        pre_text = "Read the following topics separated by '|' and provide 'one' meaningful topic label of not more than 8 words summarizing all without '|', ';' separator or full stop:"

    # Call OpenAI API to generate semantic label
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that generates concise meaningful labels for groups of text."
                },
                {
                    "role": "user",
                    "content": f"{pre_text}\n\n{chunk}"
                }
            ],
            max_tokens=100
        )
    return response['choices'][0]['message']['content'].strip()

# Group chunks by cluster and generate labels
clustered_chunks = {}
for chunk, label in zip(text_chunks, cluster_labels):
    if label not in clustered_chunks:
        clustered_chunks[label] = []
    clustered_chunks[label].append(chunk)

# Generate labels for each cluster
def generate_cluster_label(chunks):
    full_text = " ".join(chunks)
    # Counter to track the number of loops
    loop_count = 0
    labels.clear()
    for i in range(0, len(full_text), 15000):
    # Get only 15000 characters (4096 tokens ~ 16,384 characters is the maximum limit)
        chunk = full_text[i:i + 15000]
        # Pass chunk to OpenAI API method to generate semantic label
        response = llm_response_generator(chunk)
        labels.append(response)
        loop_count += 1
    print(f"count: {loop_count}")
    # Combine labels from batches to form the final label for the cluster
    return " | ".join(labels)

# Empty directory
def empty_dir(path):
    files = glob.glob(path)
    for file in files:
        os.remove(file)

# Generate labels for each cluster, minimizing API calls
labels = []
cluster_dict = {}
final_labels = {}

# Perform full processing if `complete_run == 1`
if (complete_run == 1):
    # Clear old files and save cluster text files
    empty_dir('labels/*.txt')
    empty_dir('cluster_text/textforcluster*.txt')
    for label, texts in clustered_chunks.items():
        with open(f'cluster_text/textforcluster{label}.txt', 'w') as file:
            file.write(str(texts))


    for label, texts in clustered_chunks.items():
        print(f"Processing cluster {label}...")  # Track which cluster is being worked on
        cluster_dict[label] = generate_cluster_label(texts)

    # Print the generated labels for each cluster
    for label, cluster_label in cluster_dict.items():
        str_label = re.sub(r"[',\"]", '', cluster_label)

        with open('labels/labelforcluster'+str(label)+".txt", 'w') as file:
            file.write(str_label)


    for label_file in glob.glob('labels/labelforcluster*.txt'):
        cluster_num = label_file.replace(".txt","").replace("labels/labelforcluster","")
        with open(label_file, 'r') as file:
            all_labels = file.read()

        final_labels[cluster_num] = llm_response_generator(all_labels,'labels')

        str_label = re.sub(r"[',\"]", '', final_labels[cluster_num])
        with open('labels/finallabelcluster'+str(cluster_num)+".txt", 'w') as file:
            file.write(str_label)
        print(f"Cluster {cluster_num}: {str_label}")

elif (complete_run == 0):
    print("Starting plot generation from saved labels ...")
else:
    print("Invalid 'complete_run' value. Exiting.")
    sys.exit()

# Load cluster labels
label_files = sorted([f for f in os.listdir("labels/") if f.startswith('finallabelcluster')])
read_labels = [open("labels/" + f).read() for f in label_files]
label_dict = {i: read_labels[i] for i in range(len(read_labels))}

# Plotting
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d') if n_dim == 3 else fig.add_subplot(111)
scatter_plots, text_labels = {}, {}

# Plot each cluster and add labels
for label, points in label_dict.items():
    cluster_points = reduced_embeddings[cluster_labels == label]
    scatter = ax.scatter(*cluster_points.T[:n_dim], label=label, s=40, alpha=0.6)
    scatter_plots[points[:25] + "..."] = scatter

    cluster_center = cluster_points.mean(axis=0)
    str_label = re.sub(r"['.,\"]", '', points)
    text = ax.text(*cluster_center[:n_dim], str_label, color='white', fontsize=8, ha='center',
                   bbox=dict(facecolor='black', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.15'), zorder=15)
    text_labels[points[:25] + "..."] = text

plt.subplots_adjust(left=0.3)

# Add CheckButtons to toggle cluster visibility
check_ax = plt.axes([0.05, 0.4, 0.2, 0.2])
check = CheckButtons(check_ax, list(scatter_plots.keys()), [True] * len(scatter_plots))

# Toggle visibility callback
def toggle_visibility(label):
    scatter, text = scatter_plots[label], text_labels[label]
    visible = not scatter.get_visible()
    scatter.set_visible(visible)
    text.set_visible(visible)
    fig.canvas.draw_idle()

check.on_clicked(toggle_visibility)

plt.legend(loc="best")
plt.show()
