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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm
import logging

# Initialize logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Define dataset file paths
TRANSACTION_FILE = r"D:\FIT\Senior Year\SPRING 2025\BDM\Grand Project\CodeAndData\Data\train_transaction.parquet"
IDENTITY_FILE = r"D:\FIT\Senior Year\SPRING 2025\BDM\Grand Project\CodeAndData\Data\train_identity.parquet"
MODEL_PATH = "fraud_detection_model.pkl"


def load_and_preprocess_data() -> pd.DataFrame:
    """Loads and preprocesses transaction datasets.

    Returns:
        pd.DataFrame: Cleaned and feature-engineered dataset.
    """
    logging.info("Loading dataset...")

    transactions = pd.read_parquet(TRANSACTION_FILE)
    identity = pd.read_parquet(IDENTITY_FILE)

    df = transactions.merge(identity, on="TransactionID", how="left")
    logging.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns.")

    # Fix isFraud column
    df["isFraud"] = df["isFraud"].astype(str).str.strip().astype(int)

    # Convert TransactionDT to meaningful time features
    df["TransactionDT_days"] = df["TransactionDT"] // (24 * 60 * 60)
    df["TransactionDT_hours"] = df["TransactionDT"] // (60 * 60)

    # Scale TransactionAMT
    scaler = StandardScaler()
    df["TransactionAmt_scaled"] = scaler.fit_transform(df[["TransactionAmt"]])

    # Encode categorical features
    categorical_cols = [
        "ProductCD", "card1", "card2", "card3", "card4", "card5", "card6",
        "addr1", "addr2", "P_emaildomain", "R_emaildomain",
        "DeviceType", "DeviceInfo"
    ] + [f"id_{i}" for i in range(12, 39)]  # id_12 to id_38

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")  # Fill missing values with "Unknown"
            df[col] = df[col].astype(str)       # Convert all values to strings
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

    logging.info("Data preprocessing complete.")
    return df


def construct_graph(df: pd.DataFrame) -> nx.Graph:
    """Builds a transaction graph where nodes represent transactions, and edges represent relationships.

    Args:
        df (pd.DataFrame): The transaction dataset.

    Returns:
        nx.Graph: A graph representing transaction relationships.
    """
    logging.info("Constructing transaction graph...")
    G = nx.Graph()

    # Add nodes (transactions)
    for row in tqdm(df.itertuples(index=False), total=df.shape[0], desc="Adding nodes"):
        G.add_node(row.TransactionID, isFraud=row.isFraud)

    # Add edges based on shared attributes
    for col in ["card1", "addr1"]:
        groups = df.groupby(col)["TransactionID"].apply(list)
        for transactions in tqdm(groups, desc=f"Processing {col}"):
            for i in range(len(transactions)):
                for j in range(i + 1, len(transactions)):
                    G.add_edge(transactions[i], transactions[j], weight=1)

    # Add extra edges based on time and risk
    for row in tqdm(df.itertuples(index=False), total=df.shape[0], desc="Adding extra edges"):
        if row.D1 < 3:  # Transactions within 3 days
            G.add_edge(row.TransactionID, f"time_{row.D1}", weight=1)
        if row.C1 > 5:  # High-risk transactions
            G.add_edge(row.TransactionID, f"count_{row.C1}", weight=2)

    logging.info(f"Graph constructed: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    return G


def detect_communities(G: nx.Graph) -> dict:
    """Detects fraud rings using the Louvain method.

    Args:
        G (nx.Graph): The transaction graph.

    Returns:
        dict: Mapping of nodes to community labels.
    """
    logging.info("Detecting fraud communities using Louvain method...")
    partition = community.best_partition(G)
    nx.set_node_attributes(G, partition, "community")

    num_communities = len(set(partition.values()))
    logging.info(f"Community detection complete: {num_communities} communities found.")
    return partition


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
    logging.info("Node2Vec embeddings generated successfully.")
    return embeddings


def train_model(embeddings: dict, df: pd.DataFrame) -> RandomForestClassifier:
    """Trains a RandomForest model to classify fraudulent transactions.

    Args:
        embeddings (dict): Dictionary mapping node IDs to vector embeddings.
        df (pd.DataFrame): DataFrame containing transaction data.

    Returns:
        RandomForestClassifier: The trained model.
    """
    logging.info("Preparing data for training...")
    X, y = [], []
    valid_nodes = set(df["TransactionID"].values)

    for node, embedding in embeddings.items():
        if node in valid_nodes:
            X.append(embedding)
            y.append(df.loc[df["TransactionID"] == node, "isFraud"].values[0])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logging.info(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")

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
    logging.info(f"Model saved at {path}")


if __name__ == "__main__":
    logging.info("Starting fraud detection pipeline...")

    df = load_and_preprocess_data()
    G = construct_graph(df)
    detect_communities(G)
    embeddings = generate_embeddings(G)
    model = train_model(embeddings, df)
    save_model(model, MODEL_PATH)

    logging.info("Fraud detection pipeline completed successfully! 🚀")
