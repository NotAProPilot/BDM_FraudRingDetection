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


### Graph Construction:

Creates a transaction graph in Neo4j where:

- Nodes represent individual transactions
- Edges connect related transactions (same card, address, etc.)
- Optimizes the process with batch operations to handle large datasets
- Uses indexing for faster lookups


#### Model Training:
- Trains a Random Forest classifier on the processed data
- Uses both traditional features and graph-derived features
- Evaluates model performance with classification metrics
- Saves the trained model for later use

### 3. Testing Module (TestingSnippet.py)
The testing module allows evaluation of the trained model on new data:

- Loads and preprocesses test data in the same way as training data
- Applies the saved model to make predictions
- Evaluates performance if labels are available
- Outputs predictions to a CSV file if running in production mode
