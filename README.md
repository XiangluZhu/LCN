## Layer-wise Contrastive Network for Unsupervised Graph Representation Learning

This directory includes a PyTorch implementation of [Layer-wise Contrastive Network for Unsupervised Graph Representation Learning]. Note that the code is built on top of the official PyTorch implementation of GRACE (https://github.com/CRIPAC-DIG/GRACE).

### Dependencies
* Python 3.6 
* torch 1.6.0
* torch-geometric 1.6.1
* sklearn 0.21.3 

You can install the dependencies using the following command:
```bash
pip install -r requirements.txt
``` 
You can also install the appropriate version of torch-geometric according to your environment by referring to the official site (https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html). 

### Usage

After installing the dependencies, you can train and evaluate the model LCN, wide LCN by executing train-LCN.py, train-wide-LCN.py respectively. For example, if you would like to train and evaluate LCN and wide LCN on Cora and DBLP using GPU 0, you can execute:
```bash
python train-LCN.py --dataset Cora --gpu 0
python train-wide-LCN.py --dataset Cora --gpu 0
python train-LCN.py --dataset DBLP --gpu 0
python train-wide-LCN.py --dataset DBLP --gpu 0
``` 
