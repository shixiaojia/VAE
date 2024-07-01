import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, image_size: int, in_channels: int, latent_dim: int, hid_dims: int = None):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        if not hid_dims:
            hid_dims = [32, 64, 128, 256]

        feature_size = image_size // (2**4)

        modules = []
        for h_d in hid_dims:
            modules.append(nn.Sequential(nn.Conv2d(in_channels, h_d, 3, 2, 1),
                            nn.BatchNorm2d(h_d),
                            nn.LeakyReLU()))
            in_channels = h_d

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hid_dims[-1]*feature_size**2, latent_dim)
        self.fc_var = nn.Linear(hid_dims[-1]*feature_size**2, latent_dim)

        # decoder
        self.decoder_input = nn.Linear(latent_dim, hid_dims[-1]*feature_size**2)
        hid_dims.reverse()

        modules = []
        for i in range(len(hid_dims)-1):
            modules.append(nn.Sequential(nn.ConvTranspose2d(hid_dims[i], hid_dims[i+1], 3, 2, 1, 1),
                           nn.BatchNorm2d(hid_dims[i+1]),
                           nn.LeakyReLU()))

        self.decoder = nn.Sequential(*modules)

        self.decoder_out = nn.Sequential(nn.ConvTranspose2d(hid_dims[-1], hid_dims[-1], 3, 2, 1, 1),
                                         nn.BatchNorm2d(hid_dims[-1]),
                                         nn.LeakyReLU(),
                                         nn.Conv2d(hid_dims[-1], 3, 3, 1, 1, 1),
                                         nn.Sigmoid())

    def encode(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        var = self.fc_var(x)
        return mu, var

    def decode(self, x):
        x = self.decoder_input(x)
        x = x.view(-1, 256, 6, 6)
        x = self.decoder(x)
        x = self.decoder_out(x)
        return x

    def re_parameterize(self, mu, log_var):
        std = torch.exp_(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + std*eps

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.re_parameterize(mu, log_var)
        out = self.decode(z)
        return out, mu, log_var

    def sample(self, n_samples, device):
        z = torch.randn((n_samples, self.latent_dim)).to(device)
        samples = self.decode(z)
        return samples


if __name__ == '__main__':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fake_input = torch.ones((1, 3, 96, 96))
    model = VAE(96, 3, 1024)
    out, *_ = model(fake_input)
    print(out.shape)
    print(model.sample(10, DEVICE).shape)


