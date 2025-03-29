"""
Fraud Detection Pipeline using Graph Analysis and Machine Learning

This script performs the following tasks:
1. Loads transaction data from Parquet (or converts CSV to Parquet for efficiency).
2. Constructs a transaction graph using NetworkX, connecting transactions based on shared attributes.
3. Applies the Louvain method for community detection to identify fraud rings.
4. Visualizes fraud rings using Plotly.
5. Generates node embeddings via Node2Vec for machine learning input.
6. Trains a RandomForest model to classify fraudulent transactions.
7. Saves the trained model for future predictions.

Optimizations:
- Efficient graph construction using `itertuples()` instead of `iterrows()`.
- Fraud-based edge weight reinforcement for improved detection.
- Detailed logging for debugging and monitoring.
- Memory-efficient data handling (dropping NaNs early).
- Joblib-based model persistence.

"""
import os
import pandas as pd
import networkx as nx
import joblib
import plotly.graph_objects as go
import community.community_louvain as community
from node2vec import Node2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
import logging

# Initialize logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Define dataset paths
DATA_PATH = r"D:\FIT\Senior Year\SPRING 2025\BDM\Grand Project\CodeAndData\Data"
TRANSACTION_PARQUET = os.path.join(DATA_PATH, "train_transaction.parquet")
IDENTITY_PARQUET = os.path.join(DATA_PATH, "train_identity.parquet")
MODEL_PATH = "fraud_detection_model.pkl"

def load_data() -> pd.DataFrame:
    """Loads transaction datasets from Parquet, or converts CSV to Parquet first.

    If Parquet files exist, they are used. Otherwise, CSV files are converted.

    Returns:
        pd.DataFrame: Merged transaction dataset.
    """
    if os.path.exists(TRANSACTION_PARQUET) and os.path.exists(IDENTITY_PARQUET):
        logging.info("Loading data from Parquet files...")
        transactions = pd.read_parquet(TRANSACTION_PARQUET)
        identity = pd.read_parquet(IDENTITY_PARQUET)
    else:
        logging.info("Parquet files not found. Loading CSV and converting to Parquet...")
        transactions = pd.read_csv(os.path.join(DATA_PATH, "train_transaction.csv"))
        identity = pd.read_csv(os.path.join(DATA_PATH, "train_identity.csv"))

        transactions.to_parquet(TRANSACTION_PARQUET, index=False)
        identity.to_parquet(IDENTITY_PARQUET, index=False)
        logging.info("Parquet conversion complete.")

    df = transactions.merge(identity, on="TransactionID", how="left")
    logging.info(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
    
    return df

def construct_graph(df: pd.DataFrame) -> nx.Graph:
    """Builds a transaction graph, emphasizing fraud links.

    Args:
        df (pd.DataFrame): The transaction dataset.

    Returns:
        nx.Graph: A graph representing transaction relationships.
    """
    G = nx.Graph()
    logging.info("Constructing transaction graph...")

    # Add nodes with fraud labels
    for _, row in tqdm(df[["TransactionID", "isFraud"]].itertuples(index=False), total=df.shape[0], desc="Adding nodes"):
        G.add_node(row.TransactionID, fraud=row.isFraud)

    # Add edges based on shared features (e.g., card1 and addr1)
    edge_count = 0
    for _, row in tqdm(df[["card1", "addr1", "TransactionID"]].dropna().itertuples(index=False), total=df.shape[0], desc="Adding edges"):
        G.add_edge(row.card1, row.addr1, weight=1)
        edge_count += 1

    logging.info(f"Graph construction complete: {G.number_of_nodes()} nodes, {edge_count} edges.")

    # Strengthen fraud connections
    fraud_edges_updated = 0
    for u, v in tqdm(G.edges(), desc="Updating fraud-based weights"):
        if G.nodes[u]["fraud"] == 1 and G.nodes[v]["fraud"] == 1:
            G[u][v]["weight"] += 2
            fraud_edges_updated += 1

    logging.info(f"Fraud-based weights updated for {fraud_edges_updated} edges.")
    return G

def detect_communities(G: nx.Graph) -> dict:
    """Detects fraud rings using Louvain community detection.

    Args:
        G (nx.Graph): The transaction graph.

    Returns:
        dict: Mapping of nodes to community labels.
    """
    logging.info("Detecting communities using Louvain algorithm...")

    partition = community.best_partition(G)
    nx.set_node_attributes(G, partition, "community")

    num_communities = len(set(partition.values()))
    largest_community_size = max(partition.values(), key=list(partition.values()).count)

    logging.info(f"Community detection complete. Found {num_communities} communities.")
    logging.info(f"Largest community size: {largest_community_size} members.")

    return partition

def visualize_fraud_rings(G: nx.Graph):
    """Plots fraud rings using Plotly.

    Args:
        G (nx.Graph): The transaction graph.
    """
    logging.info("Visualizing fraud rings...")
    pos = nx.spring_layout(G, seed=42)

    node_x, node_y, node_color = [], [], []
    for node, (x, y) in pos.items():
        node_x.append(x)
        node_y.append(y)
        node_color.append("red" if G.nodes[node]["fraud"] == 1 else "blue")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers", marker=dict(color=node_color, size=10)))
    fig.show()
    logging.info("Visualization completed.")

def generate_embeddings(G: nx.Graph) -> dict:
    """Generates node embeddings using Node2Vec.

    Args:
        G (nx.Graph): The transaction graph.

    Returns:
        dict: Dictionary where keys are nodes and values are embedding vectors.
    """
    logging.info("Generating Node2Vec embeddings...")
    node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)

    embeddings = {node: model.wv[node] for node in G.nodes()}
    logging.info("Embeddings generated successfully.")
    return embeddings

def train_model(embeddings: dict, df: pd.DataFrame) -> RandomForestClassifier:
    """Trains a fraud detection model using Random Forest.

    Args:
        embeddings (dict): Dictionary mapping node IDs to vector embeddings.
        df (pd.DataFrame): DataFrame containing transaction data.

    Returns:
        RandomForestClassifier: The trained model.
    """
    logging.info("Preparing data for model training...")
    
    X, y = [], []
    valid_nodes = set(df["TransactionID"].values)
    
    for node, embedding in embeddings.items():
        if node in valid_nodes:
            X.append(embedding)
            y.append(df.loc[df["TransactionID"] == node, "isFraud"].values[0])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    logging.info(f"Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    logging.info(f"Model trained successfully! Accuracy: {acc:.4f}")
    logging.info("Classification Report:\n%s", classification_report(y_test, y_pred))

    return clf

def save_model(model: RandomForestClassifier, path: str):
    """Saves the trained model using joblib.

    Args:
        model (RandomForestClassifier): The trained fraud detection model.
        path (str): File path to save the model.
    """
    joblib.dump(model, path)
    logging.info(f"Model saved successfully at {path}")

if __name__ == "__main__":
    logging.info("Starting fraud detection pipeline...")

    df = load_data()
    G = construct_graph(df)
    detect_communities(G)
    visualize_fraud_rings(G)
    embeddings = generate_embeddings(G)
    model = train_model(embeddings, df)
    save_model(model, MODEL_PATH)

    logging.info("Fraud detection pipeline completed successfully! 🚀")
    logging.info(f"Model saved at {MODEL_PATH}")
