## pytorch_examples_vae.py

Adapted from the [PyTorch VAE example](https://github.com/pytorch/examples/blob/main/vae/main.py), with some modifications.

### Changes:

- Changes in model architecture:
    - Good practices:
        - Structured the model architecture into an encoder and a decoder, using nn.Sequential ([more info here](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/guide3/Debugging_PyTorch.html#Use-nn.Sequential-and-nn.ModuleList
    )).
        - Added in-place ReLU activation functions (memory optimization, [more info here](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/guide3/Debugging_PyTorch.html#In-place-activation-functions)).
        - Stacked the final two Linear layers of the encoder into a single layer (increases efficiency, [more info here](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/guide3/Debugging_PyTorch.html#Stack-layers/weights-with-same-input)).
    - Added customization on the input, hidden and latent dimensions (previously hardcoded).

- Changes in loss functions:
    - Split the loss function into the reconstruction and the KL divergence components, printing them during training and evaluation.
    - Added a KL loss coefficient (argument ```--kl-coeff```) to modulate the importance of the KL loss w.r.t. the reconstruction loss ([ref](https://arxiv.org/abs/1804.03599)).

- Additional input arguments.

### Usage:
```bash
python pytorch_examples_vae.py [additional args]
```

The script accepts the following optional arguments:

```
  --batch-size N    input batch size for training (default: 128)
  --epochs N        number of epochs to train (default: 10)
  --no-cuda         disables CUDA training
  --no-mps          disables macOS GPU training
  --seed S          random seed (default: 1)
  --log-interval N  how many batches to wait before logging training status
  --kl-coeff N      importance of the KL loss on the overall loss function (default: 1)
  --hidden-dim N    hidden layer dimension (encoder: input -> hidden -> latent, decoder: latent -> hidden -> input), (default: 400)
  --latent-dim N    latent dimension (encoder: input -> hidden -> latent, decoder: latent -> hidden -> input), (default: 20)
```

## pytorch_lightning_vae.py

The previous VAE adapted to the PyTorch Lightning framework.

### Changes:

- **pl.LightningDataModule**: downloading and transforming the dataset, creating the dataloaders. ([Docs](https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html))

- **pl.LightningModule**: defining the model architecture, forward, training loop and validation loop, and optimizer. ([Docs](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html))

- **Callbacks**: ([Docs](https://pytorch-lightning.readthedocs.io/en/stable/extensions/callbacks.html))
    - Model checkpoints.
    - Early stopping.
    - Custom callback for reconstruction and sampling images.

- **Automatic batch size finder and learning rate finder**: Commented in the code, can be uncommented to use them. ([Docs](https://pytorch-lightning.readthedocs.io/en/latest/advanced/training_tricks.html))

- Added/removed/modified some arguments.

### Usage:

```bash
python pytorch_lightning_vae.py
```

Optional arguments:
```
  -h, --help            show this help message and exit
  -b int, --batch-size int
                        input batch size for training (default: 128)
  -l float, --learning-rate float
                        learning rate for training (default: 1e-3)
  -e int, --epochs int  number of epochs to train (default: 10)
  -a str, --accelerator str
                        accelerator for the pl.Trainer ("cpu", "gpu", "mps", ...), (default: "gpu")
  -s int, --seed int    random seed (default: 1)
  --log-interval int    how many batches to wait before logging training status
  -k float, --kl-coeff float
                        importance of the KL loss on the overall loss function (default: 1)
  --hidden-dim int      hidden layer dimension (encoder: input -> hidden -> latent, decoder: latent -> hidden -> input), (default: 400)
  --latent-dim int      latent dimension (encoder: input -> hidden -> latent, decoder: latent -> hidden -> input), (default: 20)
  --num-workers int     number of workers (cpu cores) for the dataloaders (default: 16)
```

TO DO: 
- learning rate scheduler