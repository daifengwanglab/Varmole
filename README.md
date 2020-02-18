# Varmole

## Abstract

Population studies such as GWAS have identified a variety of genomic variants associated with human diseases. To further understand potential mechanisms of disease variants, recent statistical methods associate functional omic data (e.g., gene expression) with genotype and phenotype and link variants to individual genes. However, how to interpret molecular mechanisms from such associations, especially across omics is still challenging. To address this, we develop an interpretable deep learning method, Varmole to simultaneously reveal genomic functions and mechanisms while predicting phenotype from genotype. In particular, Varmole embeds multi-omic networks into a deep neural network architecture and prioritizes variants, genes and regulatory linkages via drop-connect without needing prior feature selections.

## Description

Varmole is a Python script that uses the precompiled and pretrained a DropConnect-like Deep Neural Network in 
order to predict the disease outcome of the input SNPs and gene expressions, and to interpret the importance
of input features as well as the SNP-gene eQTL and gene-gene GRN connections.

## Installation

This script need no installation, but has the following requirements:
    -PyTorch 0.4.1 or above
    -Python3.6.5 or above


## Usage
python Varmole.py /path/to/input/file.csv

The script will compute and output 4 output files:

* file_Predictions.csv: the disease prediction outcomes
* file_FeatureImportance.csv: the importance of SNPs and TFs input that gives rise to the prediction outcomes
* file_GeneImportance.csv: the importance of gene expressions that gives rise to the prediction outcomes
* file_ConnectionImportance.csv: the importance of eQTL/GRN connections that gives rise to the prediction outcomes

For more information:
    python Varmole.py -h

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
