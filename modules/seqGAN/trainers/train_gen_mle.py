import os
import torch
from torch import nn
import torch.nn.functional as F


def train_generator_MLE(generator,
                        optimizer,
                        train_loader,
                        dvae,
                        num_epochs,
                        device,
                        verbose=True,
                        print_iter=50):
    criteriation = nn.CrossEntropyLoss()

    dvae.eval()
    dvae.to(device)

    generator.train()
    generator.to(device)

    iteration = 0
    for epoch in range(num_epochs):
        for img, label in train_loader:
            img = img.to(device)

            with torch.no_grad():
                latent = dvae.ng_q_encode(img)
            b, emb, h, w = latent.size()
            target = latent.view(b, emb, -1).permute(2, 0, 1)
            noise = torch.randn(1, b, emb, device=device)

            start_vector = torch.zeros(1, b, emb, device=target.device)
            inp = torch.cat([start_vector, target[:-1, :, :]], dim=0)

            output = generator(inp, noise)

            labels_pred = output.view(-1, emb)
            lables_true = target.argmax(dim=2).view(-1)
            loss = criteriation(labels_pred, lables_true)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            iteration += 1
            if verbose and (iteration % print_iter == 0):
                print("MLE Gen train: Epoch {} Iter {} Loss = {}".format(epoch, iteration, round(loss.item(), 5)))

