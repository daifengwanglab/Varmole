# Varmole

## Abstract

Population studies such as GWAS have identified a variety of genomic variants associated with human diseases. To further understand potential mechanisms of disease variants, recent statistical methods associate functional omic data (e.g., gene expression) with genotype and phenotype and link variants to individual genes. However, how to interpret molecular mechanisms from such associations, especially across omics is still challenging. To address this, we develop an interpretable deep learning method, Varmole to simultaneously reveal genomic functions and mechanisms while predicting phenotype from genotype. In particular, Varmole embeds multi-omic networks into a deep neural network architecture and prioritizes variants, genes and regulatory linkages via drop-connect without needing prior feature selections.

## Description

Varmole is a Python script that uses the precompiled and pretrained a DropConnect-like Deep Neural Network in 
order to predict the disease outcome of the input SNPs and gene expressions, and to interpret the importance
of input features as well as the SNP-gene eQTL and gene-gene GRN connections.

## Installation

This script need no installation, but has the following requirements:
* PyTorch 0.4.1 or above
* Python3.6.5 or above


## Usage

### Command Line Tool
`python Varmole.py /path/to/input/file.csv`

The input file file.csv is the concatenation of genotype and gene expression over the samples.

The script will compute and output 4 output files:

* file_Predictions.csv: the disease prediction outcomes
* file_FeatureImportance.csv: the importance of SNPs and TFs input that gives rise to the prediction outcomes
* file_GeneImportance.csv: the importance of gene expressions that gives rise to the prediction outcomes
* file_ConnectionImportance.csv: the importance of eQTL/GRN connections that gives rise to the prediction outcomes

For more information:
    python Varmole.py -h

### Varmole Library
The Varmole architecture is also provided in the repository. It is the Biologically DropConnect Layer that users can import into a python program. The usage is as follow:

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader

from Varmole.model import GRNeQTL #import the Bio DropConnect Layer of Varmole

class Net(nn.Module):
    def __init__(self, adj, D_in, H1, H2, H3, D_out):
        super(Net, self).__init__()
        self.adj = adj
        self.GRNeQTL = GRNeQTL(D_in, H1)
        self.linear2 = torch.nn.Linear(H1, H2)
        self.linear3 = torch.nn.Linear(H2, H3)
        self.linear4 = torch.nn.Linear(H3, D_out)

    def forward(self, x):
        h1 = self.GRNeQTL(x, self.adj).relu()
        h2 = self.linear2(h1).relu()
        h3 = self.linear3(h2).relu()
        y_pred = self.linear4(h3).sigmoid()
        return y_pred
```
We also provide the imbalanced dataset sampler for user to deal with imbalanced data as follows:

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader

train_ds = TensorDataset(X_train, y_train)
val_ds = TensorDataset(X_val, y_val)

train_dl = DataLoader(dataset=train_ds, 
                          sampler = ImbalancedDatasetSampler(train_ds), # using sampler for imbalanced data
                          batch_size=BS)
val_dl = DataLoader(dataset=val_ds, 
                        batch_size=BS*2)
```                        
                        

## License
MIT License

Copyright (c) 2020 namtk

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
