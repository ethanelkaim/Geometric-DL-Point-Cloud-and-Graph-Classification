import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_max_pool
from torch_geometric.data import DataLoader
from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import SamplePoints, NormalizeScale, Compose
from sklearn.metrics import accuracy_score
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch_geometric.data import Batch
import torch_geometric.utils as utils


class DynamicEdgeConv(MessagePassing):
    def _init_(self, mlp, k=7, aggr='max'):
        super(DynamicEdgeConv, self)._init_(aggr=aggr)
        self.k = k
        self.mlp = mlp

    def forward(self, x, edge_index):
        edge_attr = self.get_edge_attr(x, edge_index)
        x = self.mlp(x)
        x = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return x

    def message(self, x_j, edge_attr):
        return x_j

    def get_edge_attr(self, x, edge_index):
        return None

class DGCNN(torch.nn.Module):
    def _init_(self, k=7, num_classes=10):
        super(DGCNN, self)._init_()
        self.k = k
        self.conv1 = DynamicEdgeConv(self.mlp(3, 64), k=self.k, aggr='max')
        self.conv2 = DynamicEdgeConv(self.mlp(64 + 3 , 128), k=self.k, aggr='max')
        self.conv3 = DynamicEdgeConv(self.mlp(128 + 3, 256), k=self.k, aggr='max')
        self.conv4 = DynamicEdgeConv(self.mlp(256 + 3, 512), k=self.k, aggr='max')
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, num_classes)
        )

    def mlp(self, in_channels, out_channels):
        return torch.nn.Sequential(
            torch.nn.Linear(in_channels, out_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(out_channels, out_channels)
        )

    def forward(self, data):
        x, batch = data.pos, data.batch

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        edge_index = self.get_knn_graph(x, k=self.k, batch=batch)

        x1 = self.conv1(x, edge_index)
        x2 = self.conv2(torch.cat([x, x1], dim=1), edge_index)
        x3 = self.conv3(torch.cat([x, x2], dim=1), edge_index)
        x4 = self.conv4(torch.cat([x, x3], dim=1), edge_index)

        x = global_max_pool(x4, batch)
        return F.log_softmax(self.fc(x), dim=1)

    def get_knn_graph(self, x, k, batch):
        # Calculer les k plus proches voisins pour chaque point
        num_points = x.size(0)
        x_np = x.cpu().numpy()
        kdtree = cKDTree(x_np)

        edge_index = []
        for i in range(num_points):
            distances, indices = kdtree.query(x_np[i], k=k + 1)  # k + 1 pour inclure le point lui-même
            for idx in indices[1:]:  # Ignorer le point lui-même
                edge_index.append([i, idx])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        max_edges = 1_000_000
        if edge_index.size(1) > max_edges:
            edge_index = edge_index[:, :max_edges]
        return edge_index


def train(model, optimizer, criterion, data_loader):
    model.train()
    total_loss = 0

    for data in data_loader:
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(data_loader)

def test(model, data_loader):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for data in data_loader:
            out = model(data)
            pred = out.max(dim=1)[1]
            y_true.extend(data.y.tolist())
            y_pred.extend(pred.tolist())

    return accuracy_score(y_true, y_pred)

torch.manual_seed(42)

dataset_path = '/content/sample_data/data/ModelNet10'
train_dataset = ModelNet(root=dataset_path, name='10', train=True, transform=Compose([SamplePoints(1750),
                                                                                          NormalizeScale()]))
test_dataset = ModelNet(root=dataset_path, name='10', train=False, transform=Compose([SamplePoints(1750),
                                                                                          NormalizeScale()]))

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

try:
    model = torch.load("DGCNN_model (10).pt")

except FileNotFoundError:
    num_classes = 10
    epochs = 15
    model = DGCNN(k=7, num_classes=num_classes)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_val_acc_score = 0

    for epoch in range(epochs):
        loss = train(model, optimizer, criterion, train_loader)
        print(f'Epoch {epoch + 1}, Loss: {loss}')

        val_acc_score = test(model, test_loader)
        print(f'Test Accuracy: {val_acc_score}\n')
        if val_acc_score > best_val_acc_score:
            torch.save(model, 'DGCNN_model.pt')
            best_val_acc_score = val_acc_score


def evaluate_and_plot_per_class(model, data_loader, num_correct=5, num_incorrect=5):
    """
    Évalue le modèle sur les données fournies et affiche des images de nuages de points
    bien classifiés et mal classifiés pour chaque classe.

    Parameters:
    - model : le modèle entraîné (DGCNN)
    - data_loader : DataLoader contenant les données de test
    - num_correct : nombre d'images bien classifiées à afficher par classe
    - num_incorrect : nombre d'images mal classifiées à afficher par classe
    """
    # Déterminer l'appareil (GPU ou CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Initialiser les dictionnaires pour stocker les échantillons
    correct_samples = {cls: [] for cls in range(10)}
    incorrect_samples = {cls: [] for cls in range(10)}

    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            out = model(data)
            preds = out.argmax(dim=1)

            # Convertir le batch en liste de graphes individuels
            data_list = data.to_data_list()

            for d, pred in zip(data_list, preds):
                true_label = d.y.item()
                pred_label = pred.item()

                if pred_label == true_label and len(correct_samples[true_label]) < num_correct:
                    correct_samples[true_label].append((d.pos.cpu().numpy(), true_label, pred_label))
                elif pred_label != true_label and len(incorrect_samples[true_label]) < num_incorrect:
                    incorrect_samples[true_label].append((d.pos.cpu().numpy(), true_label, pred_label))

                # Vérifier si toutes les classes ont suffisamment d'échantillons
                all_correct = all(len(samples) >= num_correct for samples in correct_samples.values())
                all_incorrect = all(len(samples) >= num_incorrect for samples in incorrect_samples.values())
                if all_correct and all_incorrect:
                    break
            if all_correct and all_incorrect:
                break

    # Fonction pour mapper les labels à des couleurs
    def get_color(label):
        cmap = plt.get_cmap('tab10')
        return cmap(label % 10)

    # Afficher les nuages de points pour chaque classe
    for cls in range(10):
        class_name = f"Class {cls}"  # Vous pouvez remplacer par le nom réel des classes si disponible

        # Afficher les nuages de points bien classifiés
        samples = correct_samples[cls]
        print(f"Affichage de {len(samples)} nuages de points bien classifiés pour {class_name}:")
        for idx, (pos, true_label, pred_label) in enumerate(samples):
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=[get_color(true_label)], s=10)
            ax.set_title(f'Bien classifié: True={true_label}, Pred={pred_label}')
            plt.show()

        # Afficher les nuages de points mal classifiés
        samples = incorrect_samples[cls]
        print(f"Affichage de {len(samples)} nuages de points mal classifiés pour {class_name}:")
        for idx, (pos, true_label, pred_label) in enumerate(samples):
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c='red', s=10)
            ax.set_title(f'Mal classifié: True={true_label}, Pred={pred_label}')
            plt.show()

evaluate_and_plot_per_class(model, test_loader, num_correct=5, num_incorrect=5)
