#This code clusters textual data into semantic groups using SentenceTransformer embeddings and KMeans, generates concise and thematic cluster labels with OpenAI, calculates similarity scores for visualization, and plots the results interactively with Plotly

import numpy as np
import pandas as pd
import openai
from sentence_transformers import SentenceTransformer
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import re


# Load data
csv_file_path = 'bundle/chunks-originaltext.csv'
df = pd.read_csv(csv_file_path)
text_chunks = df['Text'].tolist()

# OpenAI API key
path = r"input/APIkey.txt"
openai.api_key = open(path).read()
open(path).close()

# Generate deterministic embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(text_chunks, convert_to_tensor=False)

# Clustering with KMeans for generating cluster labels
n_clusters = 5
kmeans_model = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans_model.fit_predict(embeddings)

# Generate concise labels for each cluster using OpenAI
def generate_cluster_label(chunks, max_input_tokens=4086):
    full_text = " ".join(chunks)
    labels = []
    for i in range(0, len(chunks), 15000):  # Split text if it's too long
        chunk = full_text[i:i + 15000]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates concise labels."},
                {"role": "user", "content": f"Provide a concise, meaningful label for the following text:\n\n{chunk}"}
            ],
            seed = 45,
            max_tokens=10
        )
        label = response['choices'][0]['message']['content'].strip()
        labels.append(label)
        print(labels)
    return labels

# Group chunks by cluster and generate labels
clustered_chunks = {label: [] for label in range(n_clusters)}
for chunk, label in zip(text_chunks, cluster_labels):
    clustered_chunks[label].append(chunk)
cluster_dict = {label: generate_cluster_label(texts)[0] for label, texts in clustered_chunks.items()}

# Map cluster labels
cluster_legend_labels = {i: f"Cluster {i+1}: {cluster_dict[i]}" for i in range(n_clusters)}

# Concatenate all cluster labels into one input for thematic anchor generation
all_cluster_labels = " | ".join(cluster_dict.values())

# Generate four thematic anchors based on cluster labels
def generate_thematic_anchors(cluster_labels):
    prompt = (
        f"Based on the following cluster labels, provide four distinct overarching thematic labels. "
        f"These should represent key dimensions that could span four quadrants for visualizing themes.\n\n"
        f"Cluster labels: {cluster_labels}\n\n"
        f"Provide four distinct theme labels."
    )
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Generate four distinct thematic labels for quadrants."},
            {"role": "user", "content": prompt}
        ],
        seed = 45,
        max_tokens=40
    )
    thematic_labels = response['choices'][0]['message']['content'].strip().split("\n")
    return [label.strip() for label in thematic_labels[:4]]

# Get dynamic thematic labels for the four quadrants
themes = generate_thematic_anchors(all_cluster_labels)
x_positive_theme, x_negative_theme, y_positive_theme, y_negative_theme = themes
x_positive_theme = (re.sub(r'\d+', '', (x_positive_theme.replace("+","")))).replace('"','')
x_negative_theme = (re.sub(r'\d+', '', (x_negative_theme.replace("-","")))).replace('"','')
y_positive_theme = (re.sub(r'\d+', '', (y_positive_theme.replace("+","")))).replace('"','')
y_negative_theme = (re.sub(r'\d+', '', (y_negative_theme.replace("-","")))).replace('"','')

# Generate embeddings for the dynamic anchors
anchor_texts = {
    "x_positive": x_positive_theme,
    "x_negative": x_negative_theme,
    "y_positive": y_positive_theme,
    "y_negative": y_negative_theme
}
anchor_embeddings = {key: model.encode([text], convert_to_tensor=False) for key, text in anchor_texts.items()}

# Calculate similarity scores for each point with respect to each axis
x_scores_positive = cosine_similarity(embeddings, anchor_embeddings["x_positive"]).flatten()
x_scores_negative = cosine_similarity(embeddings, anchor_embeddings["x_negative"]).flatten()
x_positions = x_scores_positive - x_scores_negative  # Positive values favor x_positive, negative favor x_negative

y_scores_positive = cosine_similarity(embeddings, anchor_embeddings["y_positive"]).flatten()
y_scores_negative = cosine_similarity(embeddings, anchor_embeddings["y_negative"]).flatten()
y_positions = y_scores_positive - y_scores_negative  # Positive values favor y_positive, negative favor y_negative

# Add new text prompt and map it
new_text_prompt ="Project Management and Stakeholder Engagement" #"Brand storytelling"
new_text_embedding = model.encode([new_text_prompt], convert_to_tensor=False)

# Calculate the x and y positions for the new text
new_x_score_positive = cosine_similarity(new_text_embedding, anchor_embeddings["x_positive"]).flatten()
new_x_score_negative = cosine_similarity(new_text_embedding, anchor_embeddings["x_negative"]).flatten()
new_x_position = new_x_score_positive - new_x_score_negative

new_y_score_positive = cosine_similarity(new_text_embedding, anchor_embeddings["y_positive"]).flatten()
new_y_score_negative = cosine_similarity(new_text_embedding, anchor_embeddings["y_negative"]).flatten()
new_y_position = new_y_score_positive - new_y_score_negative


# Prepare data for Plotly
plot_data = pd.DataFrame({
    "x": x_positions,
    "y": y_positions,
    "Cluster": [cluster_legend_labels[label] for label in cluster_labels],
    "Text": text_chunks
})



new_data = pd.DataFrame([{
    "x": new_x_position[0],
    "y": new_y_position[0],
    "Cluster": "New Text Prompt",
    "Text": new_text_prompt
}])


plot_data = plot_data.sort_values(by="Cluster")

# Create interactive Plotly scatter plot
fig = px.scatter(
    plot_data,
    x="x",
    y="y",
    color="Cluster",
    hover_data=["Text"],
    title="Quadrant Map of Text Data Based on Semantic Themes",
    opacity=1, 
)

# Add grid lines for quadrants
fig.add_shape(type="line", x0=0, y0=-1, x1=0, y1=1, line=dict(color="Black", width=1))
fig.add_shape(type="line", x0=-1, y0=0, x1=1, y1=0, line=dict(color="Black", width=1))

# Add custom annotations for x-axis labels
fig.add_annotation(
    x=1,  # Position for the +X label
    y=0,  # Align with the x-axis black line
    text=x_positive_theme,
    showarrow=False,
    font=dict(size=12),
    xanchor="center",
    yanchor="top"  # Place below the line
)

fig.add_annotation(
    x=-1,  # Position for the -X label
    y=0,  # Align with the x-axis black line
    text=x_negative_theme,
    showarrow=False,
    font=dict(size=12),
    xanchor="center",
    yanchor="top"  # Place below the line
)

# Add custom annotations for y-axis labels (optional, for symmetry)
fig.add_annotation(
    x=0,  # Align with the y-axis black line
    y=1,  # Position for the +Y label
    text=y_positive_theme,
    showarrow=False,
    font=dict(size=12),
    xanchor="left",
    yanchor="middle"
)

fig.add_annotation(
    x=0,  # Align with the y-axis black line
    y=-1,  # Position for the -Y label
    text=y_negative_theme,
    showarrow=False,
    font=dict(size=12),
    xanchor="left",
    yanchor="middle"
)

# Remove default tick labels if not needed
fig.update_layout(
    xaxis=dict(
        showticklabels=False
    ),
    yaxis=dict(
        showticklabels=False
    )
)

# Add the "New Text Prompt" marker as a separate trace
fig.add_trace(
    go.Scatter(
        x=new_data["x"],
        y=new_data["y"],
        mode="markers",
        marker=dict(
            symbol="x",  # Use 'x' for the marker symbol
            size=20,     # Larger size for prominence
            color="red", # Red color for high visibility
            opacity=1.0  # Full opacity to stand out
        ),
        name="New Text Prompt",  # Legend name
        text=new_data["Text"],   # Hover text
        hoverinfo="text"         # Show text on hover
    )
)

# Show the plot
fig.write_html("PerceptualMap.html")
fig.show()
print("Plotly chart created and saved")