# Graph classification for molecular property prediction:

Training a message passing graph neural network (GNN) to perform graph classification on the `ogbg-molhiv` dataset (property prediction on molecular data). The dataset consists on ~40k molecules, labeled with 0 if they have no anti-HIV activity, or 1 if they have anti-HIV activity.

Main libraries:
- OGB library for the dataset, splits and evaluation. [(Docs)](https://ogb.stanford.edu/docs/home/)
- Pytorch Geometric for the implementation of the GNN. [(Docs)](https://pytorch-geometric.readthedocs.io/en/latest/)
- Pytorch Lightning for the training and evaluation of the model. [(Docs)](https://lightning.ai/docs/pytorch/stable/)
- Tensorboard for the metrics visualization. [(Docs)](https://www.tensorflow.org/tensorboard?hl=es-419)

This notebook has been run on Google Colab.

The particularities of the model and training scheme are detailed in the notebook. In summary, I have used:
- GIN layer (Graph Isomorphism Network).
- Weighted focal loss for training with imbalanced datasets and focusing on the difficult instances.
- Regularization methods such as weight decay and early stopping.

# Notebook with the model and training:

ogbg_molhiv_gnn.ipynb

# Model checkpoint:

checkpoints/epoch=15-step=4128.ckpt