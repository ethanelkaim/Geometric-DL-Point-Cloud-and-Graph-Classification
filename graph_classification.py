import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import DataLoader, InMemoryDataset
from sklearn.metrics import accuracy_score
import pandas as pd
from torch.optim.lr_scheduler import StepLR
from torch_geometric.nn import SAGEConv


# CustomGraphDataset class
class CustomGraphDataset(InMemoryDataset):
    def __init__(self, data_list):
        self.data_list = data_list
        super(CustomGraphDataset, self).__init__(root=None)

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

# Set seed for reproducibility
torch.manual_seed(42)

# loading the data
print("Loading the data")
train_data = torch.load('data/Q2/train.pt')
val_data = torch.load('data/Q2/val.pt')
test_data = torch.load('data/Q2/test.pt')

# Wrap the data with CustomGraphDataset
train_dataset = CustomGraphDataset(train_data)
val_dataset = CustomGraphDataset(val_data)
test_dataset = CustomGraphDataset(test_data)

# Define data loaders
print("Defining the data loaders")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Model Definition
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GraphSAGE, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return F.log_softmax(x, dim=1)

# Initialize model, optimizer, and loss function
in_channels = train_data[0].x.size(-1)
hidden_channels = 128
out_channels = len(set([data.y.item() for data in train_data]))

model = GraphSAGE(in_channels, hidden_channels, out_channels)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

criterion = torch.nn.CrossEntropyLoss()

# Training and Evaluation Functions
def train(model, loader):
    model.train()
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    all_confidences = []
    with torch.no_grad():
        for data in loader:
            out = model(data)
            probs = F.softmax(out, dim=1)
            pred = out.argmax(dim=1)
            confidence, _ = probs.max(dim=1)
            all_preds.extend(pred.tolist())
            all_labels.extend(data.y.tolist())
            all_confidences.extend(confidence.tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy, all_confidences

# Training Loop
epochs = 40
best_val_acc = 0

for epoch in range(1, epochs + 1):
    loss = train(model, train_loader)
    val_acc, val_confidences = evaluate(model, val_loader)
    print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Validation Accuracy: {val_acc:.4f}')
    if val_acc > 0.89:
        print("*****************************************************************")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')
    # scheduler.step()  # Update learning rate

print("Training complete!")

# Predict on Test Set
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
predictions = []

with torch.no_grad():
    for i, data in enumerate(test_loader):
        out = model(data)
        probs = F.softmax(out, dim=1)
        pred = out.argmax(dim=1)
        confidence, _ = probs.max(dim=1)
        for idx in range(data.num_graphs):
            predictions.append({
                'idx': i * data.num_graphs + idx,
                'label': pred[idx].item(),
                'confidence': confidence[idx].item()  # Add confidence score to predictions
            })

# Save predictions to CSV
df = pd.DataFrame(predictions)
df.to_csv('prediction.csv', index=False)
print("Predictions with confidence scores saved to prediction.csv!")
