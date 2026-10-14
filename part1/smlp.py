from scipy.sparse import csr_matrix
from torch.utils.data import Dataset
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm
import time
import scipy
import scipy.sparse

class SparseMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.2):
        """
        input_dim: dimensionality of the data representation space
        hidden_dim: dimensionality of the hidden layer
        num_classes: size of outcome space
        dropout: rate at which we 'drop' hidden activations to zero 
         (this is a form of regularisation of the model)
        """
        super().__init__()

        # We start with a simple 1-hidden layer neural network.
        # The input dimensions are the same as the vocabulary size.
        # The output dimensions are the number of classes.
        
        # We keep this weight dense, but the input will be sparse.
        self.weight = nn.Parameter(
            torch.empty(input_dim, hidden_dim)
        )
        self.bias = nn.Parameter(
            torch.zeros(hidden_dim)
        )

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Everything after the first layer is dense.
        self.output = nn.Linear(hidden_dim, num_classes)

        nn.init.xavier_uniform_(self.weight) # why xavier?

    def forward(self, x):
        # x is a sparse tensor: [batch, input_dim]
        x = torch.sparse.mm(x, self.weight) # It performs the first matrix multiplication while x remains sparse.
        x = x + self.bias

        x = self.activation(x)
        x = self.dropout(x)

        return self.output(x)

def scipy_to_torch_sparse(x):
    """
    Convert a SciPy sparse matrix to a PyTorch sparse tensor.
    """
    x = x.tocoo()
    indices = torch.tensor([x.row, x.col], dtype=torch.long)
    values = torch.tensor(x.data, dtype=torch.float32)

    return torch.sparse_coo_tensor(indices, values, size=x.shape).coalesce()

class SparseDataset(Dataset):
    def __init__(self, X, y):
        """Receives a sparse matrix and a list of labels,
        and returns a sparse dataset"""
        self.X = X
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def sparse_collate(batch):
    X_rows, y_rows = zip(*batch)

    # Stack SciPy sparse rows
    X_batch = scipy.sparse.vstack(X_rows)

    # Convert the complete batch to PyTorch sparse
    X_batch = scipy_to_torch_sparse(X_batch)

    y_batch = torch.stack(y_rows)

    return X_batch, y_batch


def train_one_epoch(model, dataloader, optimizer, criterion, device="cpu"):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for X_batch, y_batch in dataloader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        logits = model(X_batch)

        loss = criterion(logits, y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y_batch.size(0)

        predictions = logits.argmax(dim=1)

        total_correct += (predictions == y_batch).sum().item()
        total_examples += y_batch.size(0)

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }

def evaluate(model, dataloader, criterion, device="cpu"):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    all_preds = []
    with torch.no_grad():

        for X_batch, y_batch in dataloader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)

            loss = criterion(logits, y_batch)

            total_loss += loss.item() * y_batch.size(0)

            predictions = logits.argmax(dim=1)
            all_preds.append(predictions)
            
            total_correct += (predictions == y_batch).sum().item()
            total_examples += y_batch.size(0)

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples
    }, torch.cat([t.detach().cpu() for t in all_preds]).numpy()

def train_neural_network_model(training_set, validation_set, encoder, save_to,   
                                hidden_dim=128, dropout=0.1, batch_size=64, learning_rate=1e-3, 
                                weight_decay=1e-4, num_epochs=20):
    """Receives a training set, and a tokenizer, and returns a trained neural network model"""

    time_start = time.time()
    # Convert the training and validation sets to vectors
    print("Transforming documents...")
    training_vectors = [encoder(text) for text, label in training_set]
    training_labels = [int(label) for text, label in training_set]
    validation_vectors = [encoder(text) for text, label in validation_set]
    validation_labels = [int(label) for text, label in validation_set]

    time_document_to_vector = time.time()
    print(f"Document transformation done! Time taken: {time_document_to_vector - time_start:.3f} seconds")
    # Keep SciPy sparse here; conversion to torch happens in sparse_collate
    X_train_sparse = csr_matrix(training_vectors)
    X_val_sparse = csr_matrix(validation_vectors)

    train_dataset = SparseDataset(X_train_sparse, training_labels)
    val_dataset = SparseDataset(X_val_sparse, validation_labels)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=sparse_collate,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=sparse_collate,
    )

    time_dataloader = time.time()
    print(f"Dataloader done! Time taken: {time_dataloader - time_document_to_vector:.3f} seconds")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SparseMLP(input_dim=len(encoder), hidden_dim=hidden_dim, num_classes=len(set(training_labels)), dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    metrics = []
    print("Training...")
    predictions = None
    for epoch in tqdm(range(num_epochs)):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics, preds = evaluate(model, val_loader, criterion, device)
        metrics.append({"train_metrics": train_metrics, "val_metrics": val_metrics})

        print(
            f"Epoch {epoch + 1:02d} | "
            f"train loss: {train_metrics['loss']:.4f} | "
            f"train acc: {train_metrics['accuracy']:.4f} | "
            f"val loss: {val_metrics['loss']:.4f} | "
            f"val acc: {val_metrics['accuracy']:.4f}"
            )
        
        # Saving the best model according to validation loss
        if epoch != 0:
            if val_metrics["loss"] <= best_val_loss:
                best_val_loss = val_metrics["loss"]
                torch.save(model.state_dict(), f"{save_to}.pt")
                best_epoch = epoch
                predictions = preds
        else:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), f"{save_to}.pt")
            best_epoch = epoch
            predictions = preds
    
    time_training = time.time()
    print(f"Training done! Time taken: {time_training - time_dataloader:.3f} seconds")

    return metrics, best_epoch, predictions
    