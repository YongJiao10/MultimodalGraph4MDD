import torch
import torch.nn as nn
import torch.nn.functional as F


def calculate_correlation(X, Y):

    mean_X = torch.mean(X, dim=-1, keepdim=True)
    mean_Y = torch.mean(Y, dim=-1, keepdim=True)

    X_centered = X - mean_X
    Y_centered = Y - mean_Y

    std_product = torch.sqrt(torch.sum(X_centered**2, dim=-1) * torch.sum(Y_centered**2, dim=-1))
    correlation = torch.sum(X_centered * Y_centered, dim=-1) / std_product

    return correlation.mean(0)


class AdaptConv(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0., is_last=False):
        super().__init__()
        self.is_last = is_last
        self.lin = nn.Linear(in_dim, out_dim)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU()
        
    def forward(self, x, edge_weight):
        x = self.lin(x)
        normalized_x = F.normalize(x, p=2., dim=-1)
        cos = torch.einsum('bij,bkj->bik', normalized_x, normalized_x)
        edge_weight = edge_weight * cos
        x = torch.einsum("npq,nqc->npc", edge_weight, x)
        if not self.is_last:
            x = self.drop(self.act(x))
        return x

class AdaptGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            is_last = i == num_layers - 1
            self.layers.append(AdaptConv(in_dim, hidden_dim, dropout, is_last=is_last))
    
    def forward(self, x, edge_weight):
        for layer in self.layers:
            x = layer(x, edge_weight)
        return x


class MyModel(nn.Module):
    
    def __init__(self, in_dim=100, hidden_dim=50, num_layers=2, dropout=0.5, num_nodes=100,
                 n_comp=16, gnn_drop=0.):
        super().__init__()
        
        self.num_nodes = num_nodes

        self.proj1 = nn.Parameter(torch.empty(num_nodes, n_comp), requires_grad=True)
        self.proj2 = nn.Parameter(torch.empty(num_nodes, n_comp), requires_grad=True)
        self.reset_parameters(self.proj1)
        self.reset_parameters(self.proj2)

        self.idx = torch.tril_indices(row=num_nodes, col=num_nodes, offset=-1)
        weight1 = nn.Parameter(torch.empty(num_nodes, num_nodes), requires_grad=False)
        self.reset_parameters(weight1)
        self.trainable_weight1 = nn.Parameter(weight1[self.idx[0], self.idx[1]], requires_grad=True)
        
        weight2 = nn.Parameter(torch.empty(num_nodes, num_nodes), requires_grad=False)
        self.reset_parameters(weight2)
        self.trainable_weight2 = nn.Parameter(weight2[self.idx[0], self.idx[1]], requires_grad=True)

        self.enc1 = AdaptGNN(in_dim, hidden_dim, num_layers, gnn_drop)
        self.enc2 = AdaptGNN(in_dim, hidden_dim, num_layers, gnn_drop)
        
        self.read = torch.nn.Conv2d(1, 1,(n_comp, 1))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim*2,128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def reset_parameters(self, weight):
        nn.init.normal_(weight)

    def loss(self, output, y, weight_decay, alpha, beta, gamma, corr_loss):

        mseloss = F.mse_loss(output.squeeze(), y)

        model_loss, edge_loss, node_loss = 0, 0, 0
        for name, param in self.named_parameters():
            if 'trainable' in name:
                edge_loss += torch.norm(param, p=1)
            elif 'proj' in name:
                node_loss += torch.norm(param, p=1)
            elif 'weight' in name:
                model_loss += torch.norm(param, p=2).pow(2)/2

        regularization_loss = weight_decay * model_loss + beta * edge_loss + gamma * node_loss
        total_loss = mseloss + regularization_loss + alpha * corr_loss
        return total_loss
    
    def forward(self, x, get_latent=False):
        '''
        Args:
            x: Tensor of shape (batch_size, 2, num_nodes, in_dim)
            - Dimension 1 (size=2): modality index, where
                x[:, 0, :, :] corresponds to fMRI input,
                x[:, 1, :, :] corresponds to EEG input.
            - Dimension 2: number of brain nodes (e.g., ROIs).
            - Dimension 3: input features per node.
        '''
        batch_size = x.size(0)

        edge_weight1 = torch.ones((self.num_nodes, self.num_nodes), device=x.device)
        edge_weight2 = torch.ones((self.num_nodes, self.num_nodes), device=x.device)
        edge_weight1[self.idx[0], self.idx[1]] = self.trainable_weight1
        edge_weight1[self.idx[1], self.idx[0]] = edge_weight1[self.idx[0], self.idx[1]]

        edge_weight2[self.idx[0], self.idx[1]] = self.trainable_weight2
        edge_weight2[self.idx[1], self.idx[0]] = edge_weight2[self.idx[0], self.idx[1]]
        edge_weight1 = edge_weight1.repeat(batch_size, 1, 1)
        edge_weight2 = edge_weight2.repeat(batch_size, 1, 1)

        x1 = self.enc1(x[:,0], edge_weight1)
        x2 = self.enc2(x[:,1], edge_weight2)

        x1t = x1.transpose(2,1)
        x2t = x2.transpose(2,1)
        X = x1t @ self.proj1
        Y = x2t @ self.proj2
        corr = calculate_correlation(X.transpose(2,1), Y.transpose(2,1))
        corr_loss = -corr.sum()
        x = torch.cat((X.transpose(2,1), Y.transpose(2,1)), dim=2)
        x = self.read(x.unsqueeze(1)).squeeze((1, 2))
        latent = x
        output = self.mlp(x)
        if get_latent:
            return output, corr_loss, latent
        return output, corr_loss
