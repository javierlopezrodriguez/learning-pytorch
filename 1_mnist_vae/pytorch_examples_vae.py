# Adapted from the Pytorch VAE example: https://github.com/pytorch/examples/blob/main/vae/main.py

from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image


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

if args.cuda:
    device = torch.device("cuda")
elif use_mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=False, **kwargs)

# Good practices:
# Use of nn.Sequential: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/guide3/Debugging_PyTorch.html#Use-nn.Sequential-and-nn.ModuleList
# nn.ReLU(inplace=True): https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/guide3/Debugging_PyTorch.html#In-place-activation-functions
# stacking mu and logvar linear layers: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/guide3/Debugging_PyTorch.html#Stack-layers/weights-with-same-input

class VAE(nn.Module):
    def __init__(self, input_dim = 784, hidden_dim = 400, latent_dim = 20):
        super(VAE, self).__init__()

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

    def encode(self, x):
        return self.encoder(x.view(-1, self.input_dim))

    def decode(self, z):
        return self.decoder(z)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu_logvar = self.encode(x) # (batch_size, 2 * latent_dim)
        mu, logvar = mu_logvar[:,:self.latent_dim], mu_logvar[:,self.latent_dim:] # (batch_size, latent_dim) each
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE(hidden_dim = args.hidden_dim, latent_dim=args.latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

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
    return (recon_l + kl_coeff * kl_l), recon_l.item(), kl_l.item()

# Training loop
def train(epoch):
    model.train()
    train_loss = 0
    train_recon_loss = 0
    train_kl_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss, recon_l, kl_l = loss_function(recon_batch, data, mu, logvar, args.kl_coeff)
        loss.backward()

        train_loss += loss.item()
        train_recon_loss += recon_l
        train_kl_loss += kl_l

        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f} (Recon: {:.4f}, KL: {:.4f})'.format(
                epoch, 
                batch_idx * len(data), 
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data),
                recon_l / len(data),
                kl_l / len(data)))

    print('====> Epoch: {} Average loss: {:.4f} (Recon: {:.4f}, KL: {:.4f})'.format(
          epoch, 
          train_loss / len(train_loader.dataset),
          train_recon_loss / len(train_loader.dataset),
          train_kl_loss / len(train_loader.dataset)))

# Evaluation loop
def test(epoch):
    model.eval()
    test_loss = 0
    test_recon_loss = 0
    test_kl_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            loss, recon_l, kl_l = loss_function(recon_batch, data, mu, logvar, args.kl_coeff)
            test_loss += loss.item()
            test_recon_loss += recon_l
            test_kl_loss += kl_l

            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    test_recon_loss /= len(test_loader.dataset)
    test_kl_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f} (Recon: {:.4f}, KL: {:.4f})'.format(
          test_loss, 
          test_recon_loss, 
          test_kl_loss))

if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, args.latent_dim, device=device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')