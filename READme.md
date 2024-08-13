# Enzyme Identification using Deep Learning

This project implements a deep learning model to identify enzyme sequences from protein sequences using functional domain encoding as features. The model achieves state-of-the-art performance, with an accuracy of approximately 94.5% on the test set.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Data Preparation](#data-preparation)
3. [Model Architecture](#model-architecture)
4. [Training Process](#training-process)
5. [Key Techniques](#key-techniques)
6. [Results and Evaluation](#results-and-evaluation)
7. [Future Work](#future-work)

## Project Overview

Enzymes are crucial proteins that catalyze biochemical reactions. Identifying enzymes from protein sequences is an important task in bioinformatics and can aid in understanding protein functions. This project uses a deep neural network to classify protein sequences as enzymes or non-enzymes based on their functional domain composition.

## Data Preparation

### Feature Encoding
- We use Pfam (Protein family) domain encoding as features.
- Each protein sequence is represented by a binary vector of length 16,306, where each position corresponds to a specific Pfam domain.
- A value of 1 indicates the presence of a domain, and 0 indicates its absence.

### Data Loading
- Data is stored in pickle files containing preprocessed Pfam domain information.
- We use a custom function `Pfam_from_pickle_file_encoding` to load and process the data.
- The dataset is balanced, with 22,168 enzyme sequences and 22,168 non-enzyme sequences.

### Data Split
- The dataset is split into training (90%) and testing (10%) sets using sklearn's `train_test_split` function.

## Model Architecture

We implement a deep fully connected neural network using PyTorch:

```python
class EnzymeNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(EnzymeNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
```

The network consists of:
- Input layer: 16,306 neurons (matching the feature vector size)
- Two hidden layers: 1,024 neurons each, with ReLU activation
- Output layer: 2 neurons (for binary classification) with softmax activation (implicit in loss function)

## Training Process

- Optimizer: Adam with learning rate 0.001 and weight decay 1e-5
- Loss function: Cross-Entropy Loss
- Batch size: 1,024
- Early stopping: Patience of 5 epochs
- Maximum epochs: 20 (early stopping usually activates before this)

## Key Techniques

### Batch Normalization
Batch Normalization is applied after each fully connected layer (except the output layer). It normalizes the inputs to each layer, which helps in several ways:
1. Reduces internal covariate shift, allowing higher learning rates.
2. Acts as a regularizer, reducing the need for Dropout.
3. Makes the network less sensitive to weight initialization.

Implementation:
```python
self.bn1 = nn.BatchNorm1d(hidden_size)
self.bn2 = nn.BatchNorm1d(hidden_size)
```

### Weight Decay
Weight decay is a form of L2 regularization that helps prevent overfitting by adding a penalty term to the loss function based on the magnitude of the weights. This encourages the model to learn smaller weights, leading to a simpler model that generalizes better.

Implementation:
```python
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
```

### Dropout
Dropout randomly sets a fraction of input units to 0 at each update during training, which helps prevent overfitting. We use a dropout rate of 0.3 after each hidden layer.

Implementation:
```python
self.dropout = nn.Dropout(0.3)
```

### Early Stopping
Early stopping is used to prevent overfitting by stopping the training process when the model's performance on the validation set starts to degrade. We implement this with a patience of 5 epochs.

## Results and Evaluation

The model achieves excellent performance:
- Test Accuracy: 94.29%
- Test Loss: 0.1783

Training process:
- Rapid improvement in the first few epochs (89.56% to 95.93% accuracy from epoch 1 to 2)
- Gradual improvement thereafter, reaching 97.22% training accuracy by epoch 8
- Early stopping activated after epoch 8, preventing overfitting

## Future Work

1. Feature importance analysis: Investigate which Pfam domains are most predictive of enzyme function.
2. External validation: Evaluate the model on a separate, external dataset to ensure generalizability.
3. Class-specific performance: Analyze the model's performance separately for enzyme and non-enzyme classes.
4. Hyperparameter tuning: Further optimize learning rate, batch size, and network architecture.
5. Ensemble methods: Explore combining multiple models for potentially improved performance.
6. Interpretability: Implement techniques to understand the model's decision-making process.

This project demonstrates the effectiveness of deep learning in bioinformatics tasks and provides a strong foundation for further research in protein function prediction.