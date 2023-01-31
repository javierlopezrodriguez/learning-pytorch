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
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--kl-coeff', type=int, default=1, metavar='N',
                    help='importance of the KL loss on the overall loss function (default: 1)')
parser.add_argument('--hidden-dim', type=int, default=400, metavar='N',
                    help='hidden layer dimension (encoder: input -> hidden -> latent, decoder: latent -> hidden -> input), (default: 400)')
parser.add_argument('--latent-dim', type=int, default=20, metavar='N',
                    help='latent dimension (encoder: input -> hidden -> latent, decoder: latent -> hidden -> input), (default: 20)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
use_mps = not args.no_mps and torch.backends.mps.is_available()

torch.manual_seed(args.seed)

# accelerator to use on pl.Trainer
if args.cuda:
    accelerator = "gpu"
elif use_mps:
    accelerator = "mps"
else:
    accelerator = "cpu"

# See https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
# when working with more than two splits, to see where train, val and test go.

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir='../data', batch_size = args.batch_size, num_workers = 16):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.ToTensor()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self) -> None:
        # Downloading the datasets if they are not already downloaded
        # for training (typically this dataset would be used for train + val)
        datasets.MNIST(self.data_dir, train=True, download=True)
        # for evaluation (in this example I'm using it for val, 
        # typically it would be used for test)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str) -> None:
        # train / val (the only stage I'm using)
        if stage == 'fit':
            self.train_data = datasets.MNIST(self.data_dir, train=True, transform=self.transform)
            self.val_data = datasets.MNIST(self.data_dir, train=False, transform=self.transform)

        if stage == 'test':
            pass 
            # here it would be the dataset with train=False if I was using Trainer.test()
            # I am only using Trainer.fit() so I'm only defining the 'fit' stage.

    def train_dataloader(self):
        return DataLoader(self.train_data, 
                          batch_size=self.batch_size, 
                          shuffle=True,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          )

    def val_dataloader(self):
        return DataLoader(self.val_data, 
                          batch_size=self.batch_size, 
                          shuffle=False,
                          num_workers = self.num_workers,
                          pin_memory=True,
                          )
    
# TO DO: un-hardcode this from the LightningDataModule:
# kwargs = {'num_workers': 16, 'pin_memory': True} if args.cuda else {}
    
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
    def __init__(self, input_dim = 784, hidden_dim = 400, latent_dim = 20):
        super(plVAE, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(*[nn.Linear(self.input_dim, self.hidden_dim),
                                      nn.ReLU(inplace=True), 
                                      nn.Linear(self.hidden_dim, 2*self.latent_dim), 
                                      ]) 
                                      # returns the concatenation of mu and logvar
                                      # default init for Linear is kaiming_uniform which only depends on input size, not output
                                      # (https://discuss.pytorch.org/t/clarity-on-default-initialization-in-pytorch/84696/2)

        self.decoder = nn.Sequential(*[nn.Linear(self.latent_dim, self.hidden_dim),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(self.hidden_dim, self.input_dim),
                                      nn.Sigmoid(),
                                      ])
    # Reparameterization + Forward
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu_logvar = self.encoder(x.view(-1, self.input_dim)) # (batch_size, 2 * latent_dim)
        mu, logvar = mu_logvar[:,:self.latent_dim], mu_logvar[:,self.latent_dim:] # (batch_size, latent_dim) each
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    # Training loop
    def training_step(self, batch, batch_idx):
        # Training process
        data, _ = batch
        recon_batch, mu, logvar = model(data) # forward (automatic calls to .train())
        loss, recon_l, kl_l = full_loss(recon_batch, data, mu, logvar) # loss calculation

        # Metrics
        metrics = {"loss": loss / len(data), # required, actual loss that will be used for training
                   "recon_loss": recon_l / len(data), 
                   "kl_loss": kl_l / len(data)}
        # logs metrics for each training_step, 
        # and the average across the epoch, to the progress bar and logger
        self.log_dict(metrics, prog_bar = True, on_step = True, on_epoch = True, logger = True)
        return metrics

    # Evaluation loop
    def validation_step(self, batch, batch_idx):
        # Validation process
        data, _ = batch
        recon_batch, mu, logvar = model(data) # forward (automatic calls to .eval())
        loss, recon_l, kl_l = full_loss(recon_batch, data, mu, logvar) # loss calculation

        # Metrics
        metrics = {"val_loss": loss / len(data), 
                   "val_recon_loss": recon_l / len(data), 
                   "val_kl_loss": kl_l / len(data)}
        # logs average metrics across the validation epoch, to the progress bar and logger
        self.log_dict(metrics, prog_bar = True, on_epoch = True, logger = True)

        # Reconstruction example
        epoch = self.current_epoch
        if batch_idx == 0:
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n],
                                    recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
            save_image(comparison.detach(),
                        'results/reconstruction_' + str(epoch) + '.png', nrow=n)
        return metrics

    # Sampling at the end of the validation epoch
    def validation_epoch_end(self, outputs):
        epoch = self.current_epoch
        sample = self.decoder(torch.randn(64, self.latent_dim, device=self.device))
        save_image(sample.view(64, 1, 28, 28).detach(),
                    'results/sample_' + str(epoch) + '.png')
    
    # Optimizer
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

# Callbacks:
# Checkpoint the model with the minimum val_loss
checkpoint_callback = ModelCheckpoint(dirpath="checkpoints/",
                                      monitor="val_loss", mode="min")
# Early stopping, stop training when val_loss does not improve
# (reduce more than min_delta=0.5) for patience=3 validation steps
early_stopping_callback = EarlyStopping(monitor="val_loss", mode="min",
                                        patience=3, min_delta=0.5)

mnist_data = MNISTDataModule()
model = plVAE(hidden_dim = args.hidden_dim, latent_dim=args.latent_dim)
trainer = pl.Trainer(accelerator=accelerator, devices=1, 
                     max_epochs=args.epochs,
                     callbacks=[checkpoint_callback, early_stopping_callback])
trainer.fit(model, datamodule=mnist_data)








