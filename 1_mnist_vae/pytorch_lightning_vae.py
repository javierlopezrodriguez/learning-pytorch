from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback


parser = argparse.ArgumentParser(description='VAE MNIST using Pytorch Lightning')
parser.add_argument('-b', '--batch-size', type=int, default=128, metavar='int',
                    help='input batch size for training (default: 128)')
parser.add_argument('-l', '--learning-rate', type=float, default=1e-3, metavar='float',
                    help='learning rate for training (default: 1e-3)')                
parser.add_argument('-e', '--epochs', type=int, default=10, metavar='int',
                    help='number of epochs to train (default: 10)')
parser.add_argument('-a', '--accelerator', default="gpu", metavar='str',
                    help='accelerator for the pl.Trainer ("cpu", "gpu", "mps", ...), (default: "gpu")')
parser.add_argument('-s', '--seed', type=int, default=1, metavar='int',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=50, metavar='int',
                    help='how many batches to wait before logging training status')
parser.add_argument('-k', '--kl-coeff', type=float, default=1, metavar='float',
                    help='importance of the KL loss on the overall loss function (default: 1)')
parser.add_argument('--hidden-dim', type=int, default=400, metavar='int',
                    help='hidden layer dimension (encoder: input -> hidden -> latent, decoder: latent -> hidden -> input), (default: 400)')
parser.add_argument('--latent-dim', type=int, default=20, metavar='int',
                    help='latent dimension (encoder: input -> hidden -> latent, decoder: latent -> hidden -> input), (default: 20)')
parser.add_argument('--num-workers', type=int, default=16, metavar='int',
                    help='number of workers (cpu cores) for the dataloaders (default: 16)')
args = parser.parse_args()

torch.manual_seed(args.seed)

# accelerator to use on pl.Trainer
if args.accelerator not in ["gpu", "mps", "cpu"]:
    print(f"Accelerator {args.accelerator} not supported.")
    args.accelerator = "cpu"
elif args.accelerator == "gpu" and not torch.cuda.is_available():
    print(f"CUDA not available.")
    args.accelerator = "cpu"
elif args.accelerator == "mps" and not torch.backends.mps.is_available():
    print(f"MPS not available.")
    args.accelerator = "cpu"

# See https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
# when working with more than two splits, to see where train, val and test go.

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir='../data', batch_size = 32, num_workers = 1, pin_memory = True):
        super().__init__()
        self.save_hyperparameters() # self.hparams.data_dir, self.hparams.batch_size, self.hparams.num_workers, self.hparams.pin_memory
        self.transform = transforms.ToTensor()

    def prepare_data(self) -> None:
        # Downloading the datasets if they are not already downloaded
        # for training (typically this dataset would be used for train + val)
        datasets.MNIST(self.hparams.data_dir, train=True, download=True)
        # for evaluation (in this example I'm using it for val, 
        # typically it would be used for test)
        datasets.MNIST(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage: str) -> None:
        # train / val (the only stage I'm using)
        if stage == 'fit':
            self.train_data = datasets.MNIST(self.hparams.data_dir, train=True, transform=self.transform)
            self.val_data = datasets.MNIST(self.hparams.data_dir, train=False, transform=self.transform)

        if stage == 'test':
            pass 
            # here it would be the dataset with train=False if I was using Trainer.test()
            # I am only using Trainer.fit() so I'm only defining the 'fit' stage.

    def train_dataloader(self):
        return DataLoader(self.train_data, 
                          batch_size=self.hparams.batch_size, 
                          shuffle=True,
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory,
                          )

    def val_dataloader(self):
        return DataLoader(self.val_data, 
                          batch_size=self.hparams.batch_size, 
                          shuffle=False,
                          num_workers = self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory,
                          )
    
# Loss functions:
def recon_loss(recon_x, x):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    return BCE

def kl_loss(mu, logvar):
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar, kl_coeff = 1):
    recon_l = recon_loss(recon_x, x)
    kl_l = kl_loss(mu, logvar)
    return (recon_l + kl_coeff * kl_l), recon_l, kl_l

from functools import partial
full_loss = partial(loss_function, kl_coeff = args.kl_coeff)

# Good practices:
# Use of nn.Sequential: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/guide3/Debugging_PyTorch.html#Use-nn.Sequential-and-nn.ModuleList
# nn.ReLU(inplace=True): https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/guide3/Debugging_PyTorch.html#In-place-activation-functions
# stacking mu and logvar linear layers: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/guide3/Debugging_PyTorch.html#Stack-layers/weights-with-same-input

# Lightning Module
class plVAE(pl.LightningModule):
    # Model architecture
    def __init__(self, input_dim = 784, hidden_dim = 400, latent_dim = 20, learning_rate = 1e-3):
        super(plVAE, self).__init__()
        self.save_hyperparameters() # self.hparams.input_dim, self.hparams.hidden_dim, self.hparams.latent_dim, self.hparams.learning_rate

        self.encoder = nn.Sequential(*[nn.Linear(self.hparams.input_dim, self.hparams.hidden_dim),
                                      nn.ReLU(inplace=True), 
                                      nn.Linear(self.hparams.hidden_dim, 2*self.hparams.latent_dim), 
                                      ]) 
                                      # returns the concatenation of mu and logvar
                                      # default init for Linear is kaiming_uniform which only depends on input size, not output
                                      # (https://discuss.pytorch.org/t/clarity-on-default-initialization-in-pytorch/84696/2)

        self.decoder = nn.Sequential(*[nn.Linear(self.hparams.latent_dim, self.hparams.hidden_dim),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(self.hparams.hidden_dim, self.hparams.input_dim),
                                      nn.Sigmoid(),
                                      ])
    # Reparameterization + Forward
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu_logvar = self.encoder(x.view(-1, self.hparams.input_dim)) # (batch_size, 2 * latent_dim)
        mu, logvar = mu_logvar[:,:self.hparams.latent_dim], mu_logvar[:,self.hparams.latent_dim:] # (batch_size, latent_dim) each
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    # training_step and validation_step note about previous bug:
    # I had previously called the model with 'model(data)', which worked but was incorrect.
    # It is called properly with 'self(data)'.
    # It only worked because I defined the 'model' variable at the end of the script and it was using that.

    # Training loop
    def training_step(self, batch, batch_idx):
        # Training process
        loss, recon_l, kl_l = self._common_step(batch, batch_idx) # Forward + loss calculation
        # Metrics
        # required "loss", actual loss that will be used for training
        metrics = {"loss": loss, "recon_loss": recon_l, "kl_loss": kl_l}
        # logs metrics for each training_step, 
        # and the average across the epoch, to the progress bar and logger
        self.log_dict(metrics, prog_bar = True, on_step = True, on_epoch = True, logger = True)
        return metrics

    # Evaluation loop
    def validation_step(self, batch, batch_idx):
        loss, recon_l, kl_l = self._common_step(batch, batch_idx) # Forward + loss calculation
        metrics = {"val_loss": loss, "val_recon_loss": recon_l, "val_kl_loss": kl_l} # Metrics
        # logs average metrics across the validation epoch, to the progress bar and logger
        self.log_dict(metrics, prog_bar = True, on_epoch = True, logger = True)
        return metrics

    def _common_step(self, batch, batch_idx):
        data, _ = batch
        recon_batch, mu, logvar = self(data) # forward
        loss, recon_l, kl_l = full_loss(recon_batch, data, mu, logvar) # loss calculation
        return loss / len(data), recon_l / len(data), kl_l / len(data)

    # Optimizer
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

# Callbacks:
# Checkpoint the model with the minimum val_loss
checkpoint_callback = ModelCheckpoint(dirpath="checkpoints/",
                                      monitor="val_loss", mode="min")
# Early stopping, stop training when val_loss does not improve
# (reduce more than min_delta=0.5) for patience=3 validation steps
early_stopping_callback = EarlyStopping(monitor="val_loss", mode="min",
                                        patience=3, min_delta=0.5)
# Custom callback for sampling and reconstruction images
class ImageCallback(Callback):
    # Sampling at the end of the validation epoch
    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        epoch = trainer.current_epoch
        sample = pl_module.decoder(torch.randn(64, pl_module.hparams.latent_dim, device=pl_module.device))
        save_image(sample.view(64, 1, 28, 28).detach(),
                    'results/sample_' + str(epoch) + '.png')
    
    # Reconstruction example
    def on_validation_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch, batch_idx, dataloader_idx) -> None:
        
        if batch_idx == 0:
            data, _ = batch
            recon_batch, _, _ = pl_module(data) # forward

            # Reconstruction example
            epoch = trainer.current_epoch
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n],
                                    recon_batch.view(data.size(0), 1, 28, 28)[:n]])
            save_image(comparison.detach(),
                        'results/reconstruction_' + str(epoch) + '.png', nrow=n)

### Main:
mnist_data = MNISTDataModule(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=(args.accelerator == "gpu"))
model = plVAE(hidden_dim = args.hidden_dim, latent_dim=args.latent_dim, learning_rate=args.learning_rate)
trainer = pl.Trainer(accelerator=args.accelerator, devices=1, 
                     max_epochs=args.epochs,
                     callbacks=[checkpoint_callback, early_stopping_callback, ImageCallback()],
                     log_every_n_steps=args.log_interval,
                    #  auto_scale_batch_size="binsearch",
                    #  auto_lr_find = True, 
                     )

# Automatic batch size finding:
# Uncomment auto_scale_batch_size in trainer (above) 
# and trainer.tune (below) for automatic batch_size finding
# This requires batch_size to be a hyperparameter (self.hparams) of the LightningDataModule
# (My result: 21399)

# Automatic learning rate finder:
# Uncomment auto_lr_find in trainer (above)
# and trainer.tune(below) for automatic learning rate finding
# This requires learning_rate or lr to be a hyperparameter (self.hparams) of the LightningModule
# (My result with batch_size 128: 0.002754228703338169)

# It would be sensible to include a learning rate scheduler,
# so that it starts with 2.7e-3 and gets progressively lower during training

# trainer.tune(model, datamodule=mnist_data)

trainer.fit(model, datamodule=mnist_data)








