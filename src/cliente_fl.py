import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import copy

class MimicDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.output(x))
        return x

def treinar_cliente_fedprox(net, dataset, idxs, args, global_net):
    """
    Treina o modelo localmente usando a função de perda do FedProx.
    """
    net.train()
    # Cria DataLoader apenas com os índices deste cliente
    ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=args['bs'], shuffle=True)
    
    optimizer = optim.Adam(net.parameters(), lr=args['lr'])
    criterion = nn.BCELoss()
    
    epoch_loss = []
    
    for iter in range(args['local_ep']):
        batch_loss = []
        for batch_idx, (images, labels) in enumerate(ldr_train):
            net.zero_grad()
            log_probs = net(images)
            
            # Perda Original
            loss_original = criterion(log_probs, labels)
            
            # Termo Proximal do FedProx: (mu / 2) * ||w - w_t||^2
            proximal_term = 0.0
            for w, w_t in zip(net.parameters(), global_net.parameters()):
                proximal_term += (w - w_t).norm(2)
            
            loss = loss_original + (args['mu'] / 2) * proximal_term
            
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
        
        epoch_loss.append(sum(batch_loss)/len(batch_loss))
        
    return net.state_dict(), sum(epoch_loss)/len(epoch_loss)

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)
    def __len__(self): return len(self.idxs)
    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label