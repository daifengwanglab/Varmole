import networkx as nx

import math
import pandas as pd
import numpy as np
import scipy as sp
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

# import matplotlib
# import matplotlib.pyplot as plt
# %matplotlib inline

import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader

from sampler import ImbalancedDatasetSampler

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

def preprocess(x, y):
    return x.float().to(device), y.int().reshape(-1, 1).to(device)

class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))

class GRNeQTL(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GRNeQTL, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input, adj):
        return input.matmul(self.weight.t() * adj) + self.bias

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class Net(nn.Module):
    def __init__(self, adj, D_in, H1, H2, H3, D_out):
        super(Net, self).__init__()
        self.adj = adj
        self.GRNeQTL = GRNeQTL(D_in, H1)
#         self.dropout = torch.nn.Dropout(0.5)
        self.linear2 = torch.nn.Linear(H1, H2)
        self.linear3 = torch.nn.Linear(H2, H3)
        self.linear4 = torch.nn.Linear(H3, D_out)

    def forward(self, x):
        h1 = self.GRNeQTL(x, self.adj).relu()
#         h1a = self.dropout(h1)
        h2 = self.linear2(h1).relu()
        h3 = self.linear3(h2).relu()
        y_pred = self.linear4(h3).sigmoid()
        return y_pred

def loss_batch(model, loss_fn, xb, yb, opt=None):
    yhat = model(xb)
    loss = loss_fn(yhat, yb.float())
    for param in model.parameters():
            loss += L1REG * torch.sum(torch.abs(param))

    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()
        
    yhat_class = np.where(yhat.detach().cpu().numpy()<0.5, 0, 1)
    accuracy = balanced_accuracy_score(yb.detach().cpu().numpy(), yhat_class)

    return loss.item(), accuracy

def fit(epochs, model, loss_fn, opt, train_dl, val_dl):
    train_loss = []
    train_accuracy = []
    val_loss = []
    val_accuracy = []
    
    for epoch in range(epochs):
        model.train()
        losses, accuracies = zip(
            *[loss_batch(model, loss_fn, xb, yb, opt) for xb, yb in train_dl]
        )
        train_loss.append(np.mean(losses))
        train_accuracy.append(np.mean(accuracies))

        model.eval()
        with torch.no_grad():
            losses, accuracies = zip(
                *[loss_batch(model, loss_fn, xb, yb) for xb, yb in val_dl]
            )
        val_loss.append(np.mean(losses))
        val_accuracy.append(np.mean(accuracies))
        
        if (epoch % 10 == 0):
            print("epoch %s" %epoch, np.mean(losses))
    
    return train_loss, train_accuracy, val_loss, val_accuracy

def load_data(gene, snp, label, eqtl, grn):
  eqtl = pd.read_csv(eqtl).drop(columns=['Unnamed: 0']).assign(weight=1)
  eqtl.columns = ['target','source','weight']

  eqtl_idx = eqtl[['source','target']].stack().reset_index(level=[0], drop=True).drop_duplicates().reset_index()
  col_idx = eqtl_idx[eqtl_idx['index']=='target'].index.values
  row_idx = eqtl_idx[eqtl_idx['index']=='source'].index.values

  gene_ls = eqtl_idx[eqtl_idx['index']=='target'][0].tolist()
  snp_ls = eqtl_idx[eqtl_idx['index']=='source'][0].tolist()

  G = nx.from_pandas_edgelist(eqtl, create_using=nx.DiGraph())
  eqtl_adj = nx.adjacency_matrix(G)

  eqtl_adj = sp.sparse.csr_matrix(
    eqtl_adj.tocsr()[row_idx,:][:,col_idx].todense()
  )

  grn = pd.read_csv(grn).drop(columns=['Unnamed: 0']).assign(weight=1)
  grn.columns = ['source','target','weight']

  G = nx.from_pandas_edgelist(grn, create_using=nx.DiGraph())
  grn_adj = nx.adjacency_matrix(G, nodelist=gene_ls)

  grn_adj_diag = grn_adj + sp.sparse.diags(np.ones(grn_adj.shape[1]), 0)

  adj = sp.sparse.vstack([eqtl_adj, grn_adj])
  adj_diag = sp.sparse.vstack([eqtl_adj, grn_adj_diag])

  expr = pd.read_csv(gene).drop(columns=['Unnamed: 0']).set_index('GeneName')
  expr = expr.reindex(gene_ls)#.sort_index(axis=1)

  snp = pd.read_csv(snp).drop(columns=['Unnamed: 0']).set_index('id')
  snp = snp.reindex(snp_ls)#.sort_index(axis=1)

  samples = list(set(expr.columns.tolist()).intersection(set(snp.columns.tolist())))
  expr = expr[samples]
  snp = snp[samples]

  obs = pd.concat([snp, expr]).sort_index(axis=1)

  label = pd.read_csv(label, sep='\t').set_index('id')
  label = label[samples].sort_index(axis=1)

  return obs, label, adj_diag

def train_Varmole(genotype, gene_expression, phenotype, GRN, eQTL, H2, H3, BS, L1REG, Opt, LR, L2REG=None, epoch, device='cpu'):
  device =device
  
  obs, label, adj = load_data(gene_expression, genotype, phenotype, eQTL, GRN)
  # obs = pd.read_pickle('/gpfs/scratch/ndnguyen/Varmole/new_data/obs_allSamples.pkl')#.sort_index()
  # label = pd.read_pickle('data/label_allSamples.pkl')
  # adj = sp.sparse.load_npz('/gpfs/scratch/ndnguyen/Varmole/new_data/adj_diag.npz')#.sort_index()

  X_train, X_test, y_train, y_test = train_test_split(obs.values.T, np.reshape(label.values, (-1, 1)),
      test_size=0.20, random_state=73)

  X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
      test_size=0.20, random_state=73)

  X_train, y_train, X_val, y_val = map(
      torch.tensor, (X_train, y_train, X_val, y_val)
  )

  BS = BS

  train_ds = TensorDataset(X_train, y_train)
  val_ds = TensorDataset(X_val, y_val)

  train_dl = DataLoader(dataset=train_ds, 
                            sampler = ImbalancedDatasetSampler(train_ds), 
                            batch_size=BS)
  val_dl = DataLoader(dataset=val_ds, 
                          batch_size=BS*2)

  train_dl = WrappedDataLoader(train_dl, preprocess)
  val_dl = WrappedDataLoader(val_dl, preprocess)

  D_in, H1, H2, H3, D_out = X_train.shape[1], adj.shape[1], H2, H3, 1
  a = torch.from_numpy(adj.todense()).float().to(device)
  model = Net(a, D_in, H1, H2, H3, D_out).to(device)

  L1REG = L1REG
  loss_fn = nn.BCELoss()

  LR = LR
  weight_decay=L2REG
  opt = opt(model.parameters(), lr=LR, weight_decay=L2REG)

  epochs = epoch
  train_loss, train_accuracy, val_loss, val_accuracy = fit(epochs, model, loss_fn, opt, train_dl, val_dl)

  # fig, ax = plt.subplots(2, 1, figsize=(12,8))

  # ax[0].plot(train_loss)
  # ax[0].plot(val_loss)
  # ax[0].set_ylabel('Loss')
  # ax[0].set_title('Training Loss')

  # ax[1].plot(train_accuracy)
  # ax[1].plot(val_accuracy)
  # ax[1].set_ylabel('Classification Accuracy')
  # ax[1].set_title('Training Accuracy')

  # plt.tight_layout()
  # plt.show()

  with torch.no_grad():
    x_tensor_test = torch.from_numpy(X_test).float().to(device)
    model.eval()
    yhat = model(x_tensor_test)
    y_hat_class = np.where(yhat.cpu().numpy()<0.5, 0, 1)
    test_accuracy = balanced_accuracy_score(y_test.reshape(-1,1), y_hat_class)
  print("Test Accuracy {:.2f}".format(test_accuracy))
  return test_accuracy
