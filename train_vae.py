import os
import numpy as np
import torch
from VAE import VAE
import argparse
from torch.utils.data import DataLoader
from PIL import Image
from torch.optim import Adam
from utils import MyDataset
from torchvision.utils import save_image
from Loss import Loss
from tqdm import tqdm


def args_parser():
    parser = argparse.ArgumentParser(description="Parameters of training vae model")
    parser.add_argument("-b", "--batch_size", type=int, default=128)
    parser.add_argument("-i", "--in_channels", type=int, default=3)
    parser.add_argument("-d", "--latent_dim", type=int, default=256)
    parser.add_argument("-l", "--lr", type=float, default=1e-3)
    parser.add_argument("-w", "--weight_decay", type=float, default=1e-5)
    parser.add_argument("-e", "--epoch", type=int, default=500)
    parser.add_argument("-v", "--snap_epoch", type=int, default=1)
    parser.add_argument("-n", "--num_samples", type=int, default=64)
    parser.add_argument("-p", "--path", type=str, default="./results_linear")
    return parser.parse_args()


def train(model, input_data, loss_fn, optimizer):
    optimizer.zero_grad()
    out, mu, log_var = model(input_data)
    total_loss = loss_fn(input_data, out, mu, log_var)
    total_loss.backward()
    optimizer.step()

    print("loss:", total_loss.item())


def save_images(path, epoch, imgs):
    N, C, H, W = imgs.shape
    os.makedirs(path, exist_ok=True)
    for i in range(N):
        img = torch.clamp(imgs[i].permute(1, 2, 0)*255, 0., 255.)

        img = img.detach().cpu().numpy().astype(np.uint8)
        save_path = os.path.join(path, f"epoch_{epoch}_id_{i}.png")
        Image.fromarray(img).save(save_path)


if __name__ == '__main__':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt = args_parser()

    loss_fn = Loss(kld_weight=0.03)

    dataset = MyDataset(img_path="../faces/", device=DEVICE)
    train_loader = DataLoader(dataset=dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)
    model = VAE(image_size=96, in_channels=opt.in_channels, latent_dim=opt.latent_dim)
    model.to(DEVICE)

    optimizer = Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

    for epoch in range(opt.epoch):
        model.train()
        data_bar = tqdm(train_loader)
        for step, data in enumerate(data_bar):
            train(model, data.to(DEVICE), loss_fn, optimizer)
            break

        if epoch % opt.snap_epoch == 0 or epoch == opt.epoch - 1:
            model.eval()
            images = model.sample(opt.num_samples, DEVICE)
            imgs = images.detach().cpu().numpy()
            saved_image_path = os.path.join(opt.path, "images")
            os.makedirs(saved_image_path, exist_ok=True)
            fname = './my_generated-images-{0:0=4d}.png'.format(epoch)
            save_image(images, fname, nrow=8)
            saved_model_path = os.path.join(opt.path, "models")
            os.makedirs(saved_model_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(saved_model_path, f"epoch_{epoch}.pth"))
