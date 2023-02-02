## pytorch_examples_vae.py

Adapted from the [PyTorch VAE example](https://github.com/pytorch/examples/blob/main/vae/main.py), with some modifications:

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

The previous VAE adapted to the PyTorch Lightning framework. (In progress)

For now, the usage is the same as the previous one:

```bash
python pytorch_lightning_vae.py
```

TO DO: 
- explanations of things
- un-hardcode the num_workers and pin_memory from the dataloaders
- batch finder and learning rate finder
- save hyperparameters in the lightning (data) modules
- add additional arguments
- keep reading
