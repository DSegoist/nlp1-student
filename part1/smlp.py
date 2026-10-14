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
    """One-hidden-layer MLP that accepts sparse input tensors.

    Inputs: feature vectors of size `input_dim` (e.g. bag-of-words).
    Process: sparse matmul into a dense hidden layer, ReLU + dropout, then a linear classifier.
    Output: logits over `num_classes`.
    """

    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.2):
        """
        input_dim: size of the input feature space (e.g. vocabulary size)
        hidden_dim: width of the hidden layer
        num_classes: number of output labels
        dropout: fraction of hidden units zeroed during training (regularisation)
        """
        super().__init__()

        # First-layer weights stay dense; inputs arriving here are sparse.
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
        """Map a sparse batch [batch, input_dim] to class logits [batch, num_classes]."""
        # First matmul keeps x sparse; result is dense from here on.
        x = torch.sparse.mm(x, self.weight)
        x = x + self.bias

        x = self.activation(x)
        x = self.dropout(x)

        return self.output(x)

def scipy_to_torch_sparse(x):
    """Convert a SciPy sparse matrix to a coalesced PyTorch sparse COO tensor.

    Input: SciPy sparse matrix.
    Output: torch sparse tensor with the same shape and non-zero values.
    """
    x = x.tocoo()
    indices = torch.tensor([x.row, x.col], dtype=torch.long)
    values = torch.tensor(x.data, dtype=torch.float32)

    return torch.sparse_coo_tensor(indices, values, size=x.shape).coalesce()

class SparseDataset(Dataset):
    """Wraps a SciPy sparse feature matrix and label list for use with DataLoader.

    Inputs: X (sparse matrix of examples), y (integer labels).
    Output per item: one sparse row and its label tensor.
    """

    def __init__(self, X, y):
        self.X = X
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def sparse_collate(batch):
    """Collate individual sparse rows into one PyTorch sparse batch.

    Input: list of (sparse row, label) pairs from SparseDataset.
    Process: stack SciPy rows, convert to a torch sparse tensor, stack labels.
    Output: (X_batch sparse tensor, y_batch long tensor).
    """
    X_rows, y_rows = zip(*batch)

    # Stack SciPy sparse rows, then convert the whole batch at once.
    X_batch = scipy.sparse.vstack(X_rows)
    X_batch = scipy_to_torch_sparse(X_batch)

    y_batch = torch.stack(y_rows)

    return X_batch, y_batch


def train_one_epoch(model, dataloader, optimizer, criterion, device="cpu"):
    """Run one training epoch: forward, loss, backward, optimizer step.

    Inputs: model, train dataloader, optimizer, loss function, device.
    Output: dict with average train loss and accuracy for the epoch.
    """
    model.train()  # enable dropout / training behaviour

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for X_batch, y_batch in dataloader:
        # move the batch to the device
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()  # clear gradients from the previous step

        logits = model(X_batch) # forward pass
        loss = criterion(logits, y_batch) # compute the loss

        loss.backward()   # compute gradients
        optimizer.step()  # update parameters

        # accumulate weighted loss and correct predictions for epoch averages
        total_loss += loss.item() * y_batch.size(0)
        predictions = logits.argmax(dim=1) # get the predicted class
        total_correct += (predictions == y_batch).sum().item() # count the number of correct predictions
        total_examples += y_batch.size(0) # count the number of examples

    return {
        "loss": total_loss / total_examples, # average loss
        "accuracy": total_correct / total_examples, # average accuracy
    }

def evaluate(model, dataloader, criterion, device="cpu"):
    """Evaluate the model without updating weights.

    Inputs: model, eval dataloader, loss function, device.
    Output: (metrics dict with loss/accuracy, numpy array of predicted labels).
    """
    model.eval()  # disable dropout / training behaviour

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    all_preds = []
    with torch.no_grad():  # no gradients needed for evaluation

        for X_batch, y_batch in dataloader:
            # move the batch to the device
            X_batch = X_batch.to(device) 
            y_batch = y_batch.to(device)

            logits = model(X_batch) # forward pass

            loss = criterion(logits, y_batch) # compute the loss

            # accumulate metrics; also keep predictions for later use
            total_loss += loss.item() * y_batch.size(0)
            predictions = logits.argmax(dim=1) # get the predicted class
            all_preds.append(predictions)

            total_correct += (predictions == y_batch).sum().item() # count the number of correct predictions
            total_examples += y_batch.size(0)

    # concatenate batch predictions into one array
    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples
    }, torch.cat([t.detach().cpu() for t in all_preds]).numpy()

def train_neural_network_model(training_set, validation_set, encoder, save_to,   
                                hidden_dim=128, dropout=0.1, batch_size=64, learning_rate=1e-3, 
                                weight_decay=1e-4, num_epochs=20):
    """Encode texts, train SparseMLP, and checkpoint the best validation model.

    Inputs: train/val (text, label) pairs, encoder (text -> sparse vector), path prefix to save weights,
            plus standard training hyperparameters.
    Process: vectorise documents, build sparse dataloaders, train for `num_epochs`, save best-by-val-loss.
    Output: per-epoch metrics list, best epoch index, and validation predictions from the best model.
    """

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

    train_dataset = SparseDataset(X_train_sparse, training_labels) # create the training dataset
    val_dataset = SparseDataset(X_val_sparse, validation_labels) # create the validation dataset

    # create the data loaders
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

    # Create the model based on the input dimension, chosen hidden dimension, number of classification classes and dropout rate
    model = SparseMLP(input_dim=len(encoder), hidden_dim=hidden_dim, num_classes=len(set(training_labels)), dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss() # create the loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay) # create the optimizer

    metrics = []
    print("Training...")
    predictions = None
    for epoch in tqdm(range(num_epochs)):
        # Train the model for one epoch
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        # Evaluate the model on the validation set
        val_metrics, preds = evaluate(model, val_loader, criterion, device)
        # Store the metrics for this epoch
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
    