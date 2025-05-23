# CorpusKey: Knowledge Graph Visualization of Semantic Clusters

**CorpusKey** is a project focused on clustering and visualizing large-scale text embeddings to reveal semantic structure within corpora. Leveraging state-of-the-art dimensionality reduction techniques, clustering algorithms, and large language models, this project generates interpretable cluster labels and perceptual maps that enhance clarity and decision-making in research, education, and analytics.

---

## Project Goals

* Cluster large text corpora meaningfully using embeddings
* Evaluate clustering quality using robust statistical metrics
* Visualize clusters with interpretable semantic dimensions
* Generate dynamic, labeled perceptual maps for intuitive insight

---

## Key Features

* **Text Embedding Clustering**: Uses pre-trained `all-MiniLM-L6-v2` models from SentenceTransformers for fast, meaningful vector generation.
* **Dimensionality Reduction**: Supports PCA, t-SNE, and UMAP for visualizing high-dimensional embeddings.
* **Clustering Algorithms**: Implements KMeans, Agglomerative Clustering, and Gaussian Mixture Models (GMMs) with BIC optimization.
* **Semantic Cluster Labeling**: Labels clusters using GPT-based summarization of representative text chunks.
* **Perceptual Mapping**: Generates 2D quadrant plots using cosine similarity to LLM-generated anchor themes (axes).
* **Interactive Plotting**: Uses Plotly for toggling clusters, overlaying prompts, and exploring thematic gaps.

---

## Technical Overview

### Clustering & Dimensionality Reduction

* **Evaluation Metrics**:

  * *Davies-Bouldin Index*: Lower is better (compact & well-separated).
  * *Silhouette Score*: Ideal ≥ 0.5, penalizes overlap.
  * *BIC (for GMMs)*: Selects optimal cluster count balancing fit and complexity.

* **Methods Compared**:

  | Method        | Shape Support | Scalability | Handles Noise |
  | ------------- | ------------- | ----------- | ------------- |
  | KMeans        | Spherical     | High        | No            |
  | Agglomerative | Arbitrary     | Low         | No            |
  | DBSCAN        | Arbitrary     | Medium      | Yes           |

* **Dimensionality Techniques**:

  * *PCA*: Fast, interpretable, linear.
  * *t-SNE*: Local focus, visually rich but stochastic.
  * *UMAP*: Balance of local + global structure, fast.

---

## Perceptual Maps

> Replaces traditional t-SNE/UMAP plots with interpretable semantic axes using LLM-generated themes.

### Visualization Process:

1. Generate 4 anchor themes: X+, X−, Y+, Y−
2. Compute cosine similarity of each embedding to the anchors.
3. Position = (X+ − X−, Y+ − Y−)
4. Overlay additional prompts for comparative analysis

**Example**:
Position (0.5, -0.1) implies:

* Strong alignment with “Profit Maximization”
* Slight preference for “Customer Satisfaction” over “Competitive Strategy”

---

## Tools & Libraries

* **Python 3.10+**
* `sentence-transformers` – Text embeddings
* `scikit-learn`, `umap-learn` – Clustering and reduction
* `openai` – Cluster and axis labeling (via GPT)
* `plotly` – Interactive visualization
* `numpy`, `pandas`, `matplotlib` – Data handling

---

## Repository Structure

```
.
├── CreateClustersAndLabels.py       # Clustering and dimensionality reduction
├── PerceptualMapWithPrompt.py       # Cosine similarity mapping and visualization
├── embeddings/                      # Input corpus embeddings
├── outputs/                         # Cluster visualizations and labels
├── README.md                        # Project documentation
```

---

## Future Enhancements

* Label refinement using Reinforcement Learning with Human Feedback (RLHF)
* Embedding-based keyword extraction for clusters
* Integration with enterprise LLM workflows for insight surfacing
* CI/CD integration and hosted web dashboard
