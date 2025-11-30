# Graph Neural Networks for Node Prediction

A complete Python implementation of Node Classification using Graph Convolutional Networks (GCN). This project demonstrates how Graph Neural Networks leverage graph structure to improve predictions compared to traditional classifiers that only use node features.

## Overview

This project implements a Node Classification task using a Graph Neural Network (GCN) to predict the missing property (category/label) of nodes in a citation network based on the features of their neighbors and the structure of the graph.

### Key Features

- **GCN Model**: Implementation of a Graph Convolutional Network with two GCNConv layers
- **Cora Dataset**: Uses the classic Cora citation network dataset from PyTorch Geometric
- **Training Pipeline**: Complete training loop with train/validation/test splits
- **Baseline Comparison**: Compares GNN performance against Logistic Regression baseline
- **Reproducibility**: Fixed random seeds for consistent results

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Required Packages

Install all dependencies using pip:

```bash
pip install -r requirements.txt
```

Or install packages individually:

```bash
# Install PyTorch (CPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install PyTorch Geometric
pip install torch-geometric

# Install other dependencies
pip install numpy scikit-learn
```

For GPU support, visit [PyTorch Get Started](https://pytorch.org/get-started/locally/) for CUDA-specific installation instructions.

## Usage

Run the main script:

```bash
python gnn_prediction.py
```

### Expected Output

The script will:
1. Load the Cora dataset
2. Display dataset statistics
3. Train the GCN model for 200 epochs
4. Train a Logistic Regression baseline
5. Compare and display results

Example output:
```
============================================================
Graph Neural Network for Node Classification
============================================================

Using device: cpu
Loading Cora dataset...
Dataset Statistics:
  - Number of nodes: 2708
  - Number of edges: 10556
  - Number of features: 1433
  - Number of classes: 7
  - Training nodes: 140
  - Validation nodes: 500
  - Test nodes: 1000

...

============================================================
Results Summary
============================================================
Model                          Test Accuracy  
---------------------------------------------
Logistic Regression (baseline) 0.5740
GCN (Graph Neural Network)     0.8100
---------------------------------------------

✓ GCN outperforms Logistic Regression by 23.60 percentage points!
```

## Model Architecture

The GCN model consists of:

1. **GCNConv Layer 1**: Input features (1433) → Hidden dimension (16)
2. **ReLU Activation** + Dropout (0.5)
3. **GCNConv Layer 2**: Hidden dimension (16) → Number of classes (7)
4. **Log-Softmax** output

The GCNConv layers propagate information from neighboring nodes, allowing the model to learn representations based on both node features and graph structure.

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Hidden Dimension | 16 |
| Dropout | 0.5 |
| Learning Rate | 0.01 |
| Weight Decay | 5e-4 |
| Epochs | 200 |
| Optimizer | Adam |
| Loss Function | NLLLoss |

## Dataset

The Cora dataset is a citation network where:
- **Nodes**: 2,708 scientific publications
- **Edges**: 10,556 citation links
- **Features**: 1,433 (bag-of-words representation)
- **Classes**: 7 research topics

The task is to predict the research topic of each paper based on its content and citation relationships.

## Why GNNs Outperform Traditional Classifiers

Traditional classifiers like Logistic Regression only consider individual node features. GNNs, on the other hand:

1. **Aggregate Neighbor Information**: Each node's representation is enriched by its neighbors
2. **Learn Graph Structure**: The model learns which structural patterns are predictive
3. **Multi-hop Reasoning**: Information propagates through multiple layers of the network

This allows GNNs to achieve significantly higher accuracy on graph-structured data.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

- [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) - Kipf & Welling, 2017
- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
- [Cora Dataset](https://relational.fit.cvut.cz/dataset/CORA)