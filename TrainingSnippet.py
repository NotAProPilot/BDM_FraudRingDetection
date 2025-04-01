import os
import pandas as pd
import joblib
import logging
import time
from py2neo import Graph
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pushbullet import Pushbullet  # Pushbullet for notifications

# Try using rich for better progress bars, otherwise fallback to tqdm
try:
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
    USE_RICH = True
except ImportError:
    from tqdm import tqdm
    USE_RICH = False

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,  
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler()]
)

# Pushbullet API Key (replace with your actual API key)
PUSHBULLET_API_KEY = "o.hMiOPiMjcqBNFLCFdQWM9xaX1MKnOyy0"
pb = Pushbullet(PUSHBULLET_API_KEY)

# File Paths
TRANSACTION_FILE = r"D:\FIT\Senior Year\SPRING 2025\BDM\Grand Project\CodeAndData\Data\train_transaction_cut.csv"
IDENTITY_FILE = r"D:\FIT\Senior Year\SPRING 2025\BDM\Grand Project\CodeAndData\Data\train_identity.csv"
MODEL_PATH = r"D:\FIT\Senior Year\SPRING 2025\BDM\Grand Project\CodeAndData\Code\fraud_detection_model.pkl"

# Total number of pipeline stages
TOTAL_STEPS = 5  

def setup_progress_bar():
    """Sets up the progress bar."""
    if USE_RICH:
        return Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%")
        )
    return None

def load_and_preprocess_data():
    """
    Loads and preprocesses the transaction and identity datasets.

    Returns:
        pd.DataFrame: Merged and preprocessed dataset.
    """
    logging.info("📂 Loading dataset...")

    transactions = pd.read_csv(TRANSACTION_FILE)
    identity = pd.read_csv(IDENTITY_FILE)

    df = transactions.merge(identity, on="TransactionID", how="left")
    logging.info(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns.")

    # Ensure 'isFraud' is numeric
    df["isFraud"] = df["isFraud"].astype(str).str.strip().astype(int)

    # Convert TransactionDT to meaningful time features
    df["TransactionDT_days"] = df["TransactionDT"] // (24 * 60 * 60)
    df["TransactionDT_hours"] = df["TransactionDT"] // (60 * 60)

    # Scale TransactionAmt
    scaler = StandardScaler()
    df["TransactionAmt_scaled"] = scaler.fit_transform(df[["TransactionAmt"]])

    # Auto-detect non-numeric columns & convert
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    if categorical_cols:
        logging.info(f"🔍 Encoding categorical columns: {categorical_cols}")
        label_encoders = {}

        for col in categorical_cols:
            df[col] = df[col].fillna("Unknown")  # Handle missing values
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])  # Convert to numeric
            label_encoders[col] = le  # Store encoders if needed later

    logging.info("✅ Data preprocessing complete.")
    return df

def construct_graph(df: pd.DataFrame):
    """
    Constructs a transaction graph in Neo4j.

    Args:
        df (pd.DataFrame): Processed transaction data.

    Returns:
        Graph: Neo4j connection object.
    """
    logging.info("🔗 Connecting to Neo4j...")
    graph = Graph("bolt://localhost:7687", auth=("neo4j", "12345678"))

    logging.info("🛠️ Creating transaction nodes in Neo4j...")

    tx = graph.begin()
    try:
        for row in df.itertuples(index=False):
            tx.run(
                "MERGE (t:Transaction {id: $TransactionID, isFraud: $isFraud})",
                TransactionID=row.TransactionID, isFraud=row.isFraud
            )
        tx.commit()
    except Exception as e:
        tx.rollback()
        logging.error(f"❌ Error in Neo4j transaction: {e}")
        raise e

    logging.info("🔗 Creating relationships between transactions...")
    for col in ["card1", "addr1"]:
        groups = df.groupby(col)["TransactionID"].apply(list)
        for transactions in groups:
            for i in range(len(transactions)):
                for j in range(i + 1, len(transactions)):
                    graph.run(
                        """
                        MATCH (a:Transaction {id: $id1}), (b:Transaction {id: $id2})
                        MERGE (a)-[:LINKED_TO]->(b)
                        """,
                        id1=transactions[i], id2=transactions[j]
                    )

    logging.info("✅ Graph construction complete.")
    return graph

def train_model(df: pd.DataFrame):
    """
    Trains a RandomForest classifier for fraud detection.

    Args:
        df (pd.DataFrame): Processed dataset.

    Returns:
        RandomForestClassifier: Trained model.
    """
    logging.info("📊 Preparing data for training...")

    X = df.drop(columns=["isFraud", "TransactionID"])
    y = df["isFraud"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    logging.info(f"🔍 Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    logging.info(f"✅ Model trained successfully! Accuracy: {acc:.4f}")
    logging.info("📜 Classification Report:\n%s", classification_report(y_test, y_pred))
    return clf

def save_model(model: RandomForestClassifier):
    """
    Saves the trained model to a file and sends a Pushbullet notification.

    Args:
        model (RandomForestClassifier): The trained model.
    """
    joblib.dump(model, MODEL_PATH)
    logging.info(f"✅ Model saved at {MODEL_PATH}")

    # Send Pushbullet notification
    pb.push_note("Fraud Detection Training Complete", f"Model saved with accuracy: {accuracy_score:.4f}")

if __name__ == "__main__":
    logging.info("🚀 Starting fraud detection pipeline...")

    progress = setup_progress_bar()
    if progress:
        with progress:
            task = progress.add_task("🔄 Running Pipeline", total=TOTAL_STEPS)
            df = load_and_preprocess_data()
            progress.advance(task)

            graph = construct_graph(df)
            progress.advance(task)

            model = train_model(df)
            progress.advance(task)

            save_model(model)
            progress.advance(task)
    else:
        df = load_and_preprocess_data()
        graph = construct_graph(df)
        model = train_model(df)
        save_model(model)

    logging.info("🎉 Fraud detection pipeline completed successfully!")
    logging.info("🛑 Exiting...")
    time.sleep(2)  # Optional: Wait before exiting to see the final logs
    os._exit(0)  # Force exit to avoid hanging threads  