# Introduction
This is the repository for the code of Group 2-01 of class Big Data Mining, Spring 2025, on the topic of "Detecting Credit Card Fraud Ring using Graph Neural Network". Our project implements a graph-based approach to fraud detection, leveraging the relationships between transactions, devices, and users to identify suspicious patterns that might indicate fraud rings rather than just individual fraudulent transactions.

# Purpose
The purpose of this project is to:
1. Implement a graph-based neural network (GNN) approach to detect credit card fraud rings
2. Demonstrate how modeling transactions as a graph can reveal complex relationships and patterns not visible when analyzing transactions in isolation
3. Create a complete pipeline from data exploration to model training and testing
4. Provide a framework that can adapt to evolving fraud strategies

# How our codes work
The repository contains three main components:
## 1. Exploratory Data Analysis (EDA_FromTrainingData.py)
This script performs initial exploration of the dataset to understand its characteristics:
- Merges transaction and identity datasets
- Visualizes missing data using the missingno library
- Calculates correlation between features and fraud labels
- Analyzes distribution of transaction amounts by fraud status
- Provides visualizations to better understand the data patterns
The insights from this exploration inform the feature engineering and model development phases.
## 2. Training Pipeline (TrainingSnippet.py)
The training pipeline consists of several key steps:

- Data Loading and Preprocessing:
- Loads transaction and identity data
- Creates derived features from transaction datetime
- Scales numerical features
- Encodes categorical variables


## Graph Construction:

Creates a transaction graph in Neo4j where:

- Nodes represent individual transactions
- Edges connect related transactions (same card, address, etc.)
- Optimizes the process with batch operations to handle large datasets
- Uses indexing for faster lookups


### Model Training:
- Trains a Random Forest classifier on the processed data
- Uses both traditional features and graph-derived features
- Evaluates model performance with classification metrics
- Saves the trained model for later use

## 3. Testing Module (TestingSnippet.py)
The testing module allows evaluation of the trained model on new data:

- Loads and preprocesses test data in the same way as training data
- Applies the saved model to make predictions
- Evaluates performance if labels are available
- Outputs predictions to a CSV file if running in production mode

## Setup
### Prerequisites

1. Install Anaconda
2. Set up a Neo4j database
3. Clone the repository
4. Prepare your data files

### Step 1: Install Anaconda

1. Download and install Anaconda from the [official website](https://www.anaconda.com/download):
   - Choose the version appropriate for your operating system (Windows, macOS, or Linux)
   - Follow the installation prompts

2. Verify your installation by opening Anaconda Navigator or by running:
   ```bash
   conda --version
   ```

### Step 2: Clone the Repository and Set Up Environment

```bash
# Clone the repository
git clone https://github.com/your-username/fraud-detection-system.git
cd fraud-detection-system

# Create a new conda environment
conda create -n fraud-detection python=3.10
conda activate fraud-detection

# Install required packages
conda install -c conda-forge pandas scikit-learn matplotlib seaborn joblib py2neo
pip install missingno rich plotly
```

### Step 3: Set Up Neo4j Database

1. Download and install Neo4j Desktop from [Neo4j website](https://neo4j.com/download/)
2. Create a new database with the following settings:
   - Name: FraudDB (or any name of your choice)
   - Password: 12345678 (as used in the code)
   - Port: 7687 (default bolt port)
3. Start the database

## Step 4: File Structure

Organize your project with the following structure:

```
fraud-detection-system/
│
├── data/
│   ├── train_transaction.csv
│   ├── train_identity.csv
│   ├── test_transaction.csv
│   └── test_identity.csv
│
├── code/
│   ├── EDA_FromTrainingData.py
│   ├── TrainingSnippet.py
│   └── TestingSnippet.py
│
└── README.md
```

### Step 5: Configure File Paths

Update the file paths in each script to match your local directory structure:

1. In `EDA_FromTrainingData.py`:
   ```python
   train_transaction = pd.read_csv("./data/train_transaction.csv")
   train_identity = pd.read_csv("./data/train_identity.csv")
   ```

2. In `TrainingSnippet.py`:
   ```python
   TRANSACTION_FILE = "./data/train_transaction.csv"
   IDENTITY_FILE = "./data/train_identity.csv"
   MODEL_PATH = "./code/fraud_detection_model.pkl"
   ```

3. In `TestingSnippet.py`:
   ```python
   BASE_DIR = os.path.dirname(os.path.abspath(__file__))
   MODEL_PATH = os.path.join(BASE_DIR, "fraud_detection_model.pkl")
   TEST_TRANSACTION_FILE = os.path.join(BASE_DIR, "../data/test_transaction.csv")
   TEST_IDENTITY_FILE = os.path.join(BASE_DIR, "../data/test_identity.csv")
   ```

### Step 6: Run the Scripts

Run the scripts in the following order:

1. First, run the EDA script to explore the data:
   ```bash
   python code/EDA_FromTrainingData.py
   ```

2. Next, run the training script to build and save the model:
   ```bash
   python code/TrainingSnippet.py
   ```

3. Finally, run the testing script to evaluate the model:
   ```bash
   python code/TestingSnippet.py
   ```

### Notes

- Make sure Neo4j is running when executing the training script
- The training process may take some time depending on the size of your dataset
- The system will save the trained model as `fraud_detection_model.pkl` in the code directory
- Test results will be saved to `data/fraud_predictions.csv` if no labels are present in the test data

##3 Troubleshooting

- If you encounter memory issues during graph construction, try reducing batch sizes in the `construct_graph` function
- If the Neo4j connection fails, verify your database is running and check the connection parameters
- For visualization issues, ensure matplotlib, seaborn, and plotly are properly installed

