import os
import torch
from torch import nn
import torch.nn.functional as F
from modules.seqGAN.funcs import latent_to_img


def train_discriminator(discriminator,
                        generator,
                        optimizer_d,
                        train_loader,
                        dvae,
                        hidden_height,
                        heddin_width,
                        num_epochs,
                        device,
                        verbose=True,
                        print_iter=50):
    dvae.eval()
    dvae.to(device)

    generator.eval()
    generator.to(device)

    discriminator.train()
    discriminator.to(device)

    criterion = nn.BCELoss()

    iteration = 0
    for epoch in range(num_epochs):
        for img, label in train_loader:
            real = img.to(device)
            batch, _, _, _ = real.size()

            with torch.no_grad():
                latent_fake = generator.sample(n_samples=batch, device=device)
                fake = latent_to_img(latent_fake, dvae, hidden_height, heddin_width)

            inp = torch.cat([real, fake], dim=0)
            target = torch.cat([torch.ones(batch, device=device), torch.zeros(batch, device=device)], dim=0)

            optimizer_d.zero_grad()
            out = discriminator(inp)
            loss = criterion(out, target)
            loss.backward()
            optimizer_d.step()

            if verbose and (iteration % print_iter == 0):
                print("MLE Dis train: Epoch {} Iter {} Loss = {}".format(epoch, iteration, round(loss.item(), 5)))

            iteration += 1





