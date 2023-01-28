# learning-pytorch
Experimenting and learning with pytorch, pytorch geometric, pytorch lightning...
Eventually, I'd like to work with the above tools and molecular graphs.

## Environment

I'm running this in a Conda environment with Python 3.10.9 and the following packages:
- PyTorch
- PyTorch Geometric (for graph neural networks)
- Pytorch-Lightning (automating training and evaluation loops, adding early stopping and other callbacks...)
- RDKit (managing molecules)
- Ax (library for bayesian optimization)
- OGB ([Open Graph Benchmark](https://ogb.stanford.edu/docs/home/))

Currently, Ax, RDKit and OGB are not necessary, I have added them for convenience.

Installed with:

```bash
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -y pyg -c pyg
conda install -y pytorch-lightning rdkit -c conda-forge
pip install ax-platform
pip install ogb
```