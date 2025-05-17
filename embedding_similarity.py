import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_from_disk


def load_embeddings_dataframe(path: str) -> pd.DataFrame:
    """
    Loads the fast-access embeddings dataset and returns it as a pandas DataFrame.

    Parameters:
        path (str): Directory path where the dataset is saved.

    Returns:
        pd.DataFrame: DataFrame containing at least 'clip_text_embedding' and 'qid'.
    """
    ds = load_from_disk(path)
    return ds.to_pandas()


def compute_cosine_similarity(embeddings: np.ndarray) -> np.ndarray:
    """
    Computes the cosine similarity matrix from the given embeddings.

    Parameters:
        embeddings (np.ndarray): 2D array where each row is an embedding vector.

    Returns:
        np.ndarray: Cosine similarity matrix.
    """
    return cosine_similarity(embeddings)


def build_similarity_graph(df: pd.DataFrame, sim_matrix: np.ndarray, threshold: float = 0.75,
                           top_k: int = 5) -> nx.Graph:
    """
    Builds a similarity graph where each node is an entity (identified by its QID) and
    edges connect entities whose cosine similarity (from text embeddings) is above the threshold.

    Parameters:
        df (pd.DataFrame): DataFrame with at least a 'qid' column.
        sim_matrix (np.ndarray): Cosine similarity matrix.
        threshold (float): Minimum cosine similarity to add an edge.
        top_k (int): Maximum number of similar entities to connect for each node.

    Returns:
        nx.Graph: The resulting similarity graph.
    """
    G = nx.Graph()
    qids = df["qid"].tolist()
    num_nodes = sim_matrix.shape[0]

    # Add all QIDs as nodes.
    for qid in qids:
        G.add_node(qid)

    for i in range(num_nodes):
        sims = sim_matrix[i]
        # Get sorted indices in descending order of similarity.
        similar_indices = np.argsort(-sims)
        count = 0
        for j in similar_indices:
            if i == j:
                continue
            if sims[j] >= threshold:
                G.add_edge(qids[i], qids[j], weight=sims[j])
                count += 1
            if count >= top_k:
                break
    return G


def visualize_graph_pretty(G: nx.Graph, output_file: str) -> None:
    """
    Visualizes the similarity graph using a spring layout with enhanced aesthetics, and saves the plot to a file.

    Parameters:
        G (nx.Graph): The similarity graph.
        output_file (str): Path to the output image file.
    """
    plt.figure(figsize=(15, 12))
    pos = nx.spring_layout(G, seed=42, k=0.15)

    # Draw nodes with custom styling.
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='skyblue', alpha=0.9,
                           linewidths=1.5, edgecolors='black')

    # Draw edges with widths proportional to their similarity weight.
    edges = G.edges(data=True)
    edge_widths = [data['weight'] * 5 for (_, _, data) in edges]
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, edge_color='gray')

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_color='darkblue', font_weight='bold')

    plt.title("Entity Similarity Graph", fontsize=16)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.show()


def main() -> None:
    embeddings_path: str = "/Data/MMEL/fast_access_embeddings"
    output_file: str = "entity_similarity_graph.png"

    df = load_embeddings_dataframe(embeddings_path)
    df = df.dropna(subset=["clip_text_embedding", "qid"])

    embeddings = np.array(df["clip_text_embedding"].tolist())
    sim_matrix = compute_cosine_similarity(embeddings)
    G = build_similarity_graph(df, sim_matrix, threshold=0.75, top_k=5)
    visualize_graph_pretty(G, output_file)
    print(f"Entity similarity graph saved to {output_file}")


if __name__ == "__main__":
    main()
