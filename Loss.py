import torch
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self, kld_weight=0.03):
        super(Loss, self).__init__()
        self.kld_weight = kld_weight
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, input, output, mu, log_var):
        recon_loss = self.criterion(output, input)
        # kld_loss = torch.mean(-0.5*torch.sum(1+log_var-mu**2-log_var.exp(), dim=1), dim=0)
        kld_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        # loss = recon_loss + kld_loss
        # return {"loss": loss, "recon_loss": recon_loss.detach(), "kld_loss": kld_loss.detach()}
        return recon_loss + self.kld_weight*kld_loss
