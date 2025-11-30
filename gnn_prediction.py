"""
Graph Neural Network for Node Classification

This script implements a Node Classification task using a Graph Convolutional Network (GCN).
It demonstrates how GNNs can leverage graph structure to improve predictions compared to
classifiers that only use node features.

Author: ML Scientist
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Set random seeds for reproducibility
def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class GCN(nn.Module):
    """
    Graph Convolutional Network (GCN) for Node Classification.
    
    Architecture:
        - GCNConv layer 1: input_features -> hidden_dim
        - ReLU activation + Dropout
        - GCNConv layer 2: hidden_dim -> num_classes
    
    The model propagates information from neighbors using the GCNConv layers,
    allowing nodes to learn representations based on their local graph structure.
    """
    
    def __init__(self, num_features, hidden_dim, num_classes, dropout=0.5):
        """
        Initialize the GCN model.
        
        Args:
            num_features (int): Number of input features per node
            hidden_dim (int): Number of hidden units
            num_classes (int): Number of output classes
            dropout (float): Dropout probability
        """
        super(GCN, self).__init__()
        
        # First GCN layer: input features to hidden dimension
        self.conv1 = GCNConv(num_features, hidden_dim)
        
        # Second GCN layer: hidden dimension to output classes
        self.conv2 = GCNConv(hidden_dim, num_classes)
        
        # Additional linear layers as per requirements
        self.linear1 = nn.Linear(num_features, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, num_classes)
        
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        """
        Forward pass of the GCN model.
        
        Args:
            x (Tensor): Node feature matrix of shape [num_nodes, num_features]
            edge_index (Tensor): Graph connectivity in COO format [2, num_edges]
        
        Returns:
            Tensor: Log-softmax probabilities for each class
        """
        # First GCN convolution + ReLU activation
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second GCN convolution
        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1)


def load_dataset(dataset_name='Cora', root='/tmp/data'):
    """
    Load a Planetoid dataset (Cora, CiteSeer, or PubMed).
    
    Args:
        dataset_name (str): Name of the dataset
        root (str): Root directory for downloading the dataset
    
    Returns:
        tuple: (dataset, data) where data contains node features, edges, labels, and masks
    """
    print(f"Loading {dataset_name} dataset...")
    dataset = Planetoid(root=root, name=dataset_name)
    data = dataset[0]
    
    print(f"Dataset Statistics:")
    print(f"  - Number of nodes: {data.num_nodes}")
    print(f"  - Number of edges: {data.num_edges}")
    print(f"  - Number of features: {dataset.num_features}")
    print(f"  - Number of classes: {dataset.num_classes}")
    print(f"  - Training nodes: {data.train_mask.sum().item()}")
    print(f"  - Validation nodes: {data.val_mask.sum().item()}")
    print(f"  - Test nodes: {data.test_mask.sum().item()}")
    
    return dataset, data


def train_gcn(model, data, optimizer, criterion):
    """
    Train the GCN model for one epoch.
    
    Args:
        model: GCN model
        data: Graph data containing features, edges, labels, and masks
        optimizer: PyTorch optimizer
        criterion: Loss function
    
    Returns:
        float: Training loss
    """
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    out = model(data.x, data.edge_index)
    
    # Compute loss only on training nodes
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item()


def evaluate_gcn(model, data, mask):
    """
    Evaluate the GCN model on a subset of nodes.
    
    Args:
        model: GCN model
        data: Graph data
        mask: Boolean mask indicating which nodes to evaluate
    
    Returns:
        float: Accuracy on the masked nodes
    """
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        correct = (pred[mask] == data.y[mask]).sum().item()
        accuracy = correct / mask.sum().item()
    return accuracy


def train_logistic_regression(data):
    """
    Train a Logistic Regression classifier using only node features.
    This serves as a baseline that ignores graph structure.
    
    Args:
        data: Graph data containing features, labels, and masks
    
    Returns:
        tuple: (model, test_accuracy)
    """
    print("\nTraining Logistic Regression baseline (ignores graph structure)...")
    
    # Convert to numpy arrays
    X = data.x.cpu().numpy()
    y = data.y.cpu().numpy()
    
    # Get training and test indices
    train_mask = data.train_mask.cpu().numpy()
    test_mask = data.test_mask.cpu().numpy()
    
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    
    # Train Logistic Regression
    clf = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
    clf.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return clf, accuracy


def main():
    """Main function to run the complete GNN node classification pipeline."""
    
    print("=" * 60)
    print("Graph Neural Network for Node Classification")
    print("=" * 60)
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Check device availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load dataset
    dataset, data = load_dataset('Cora')
    data = data.to(device)
    
    # Model hyperparameters
    hidden_dim = 16
    dropout = 0.5
    learning_rate = 0.01
    weight_decay = 5e-4
    epochs = 200
    
    print(f"\nHyperparameters:")
    print(f"  - Hidden dimension: {hidden_dim}")
    print(f"  - Dropout: {dropout}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Weight decay: {weight_decay}")
    print(f"  - Epochs: {epochs}")
    
    # Initialize GCN model
    model = GCN(
        num_features=dataset.num_features,
        hidden_dim=hidden_dim,
        num_classes=dataset.num_classes,
        dropout=dropout
    ).to(device)
    
    print(f"\nGCN Model Architecture:")
    print(model)
    
    # Optimizer and loss function
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    criterion = nn.NLLLoss()
    
    # Training loop
    print("\n" + "=" * 60)
    print("Training GCN Model")
    print("=" * 60)
    
    best_val_acc = 0
    best_test_acc = 0
    
    for epoch in range(1, epochs + 1):
        # Train
        loss = train_gcn(model, data, optimizer, criterion)
        
        # Evaluate
        train_acc = evaluate_gcn(model, data, data.train_mask)
        val_acc = evaluate_gcn(model, data, data.val_mask)
        test_acc = evaluate_gcn(model, data, data.test_mask)
        
        # Track best validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
        
        # Print progress every 20 epochs
        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}: Loss={loss:.4f}, "
                  f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, Test Acc={test_acc:.4f}")
    
    print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")
    print(f"Corresponding Test Accuracy: {best_test_acc:.4f}")
    
    # Final GCN evaluation
    final_test_acc = evaluate_gcn(model, data, data.test_mask)
    print(f"Final GCN Test Accuracy: {final_test_acc:.4f}")
    
    # Train and evaluate Logistic Regression baseline
    print("\n" + "=" * 60)
    print("Baseline Comparison")
    print("=" * 60)
    
    _, lr_accuracy = train_logistic_regression(data)
    print(f"Logistic Regression Test Accuracy: {lr_accuracy:.4f}")
    
    # Compare results
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"{'Model':<30} {'Test Accuracy':<15}")
    print("-" * 45)
    print(f"{'Logistic Regression (baseline)':<30} {lr_accuracy:.4f}")
    print(f"{'GCN (Graph Neural Network)':<30} {final_test_acc:.4f}")
    print("-" * 45)
    
    improvement = (final_test_acc - lr_accuracy) * 100
    if improvement > 0:
        print(f"\n✓ GCN outperforms Logistic Regression by {improvement:.2f} percentage points!")
        print("  This demonstrates the power of leveraging graph structure for node classification.")
    else:
        print(f"\n! Logistic Regression performs better by {-improvement:.2f} percentage points.")
        print("  Consider tuning GCN hyperparameters or training for more epochs.")
    
    return final_test_acc, lr_accuracy


if __name__ == "__main__":
    gnn_acc, baseline_acc = main()
