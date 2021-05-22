import os
import torch
from torch import nn
import torch.nn.functional as F


def train_discriminator(discriminator,
                        generator,
                        optimizer_d,
                        train_loader,
                        dvae,
                        num_epochs,
                        device,
                        verbose=True):
    dvae.eval()
    dvae.to(device)

    generator.eval()
    generator.to(device)

    discriminator.train()
    discriminator.to(device)

    criterion = nn.CrossEntropyLoss()

    iteration = 0
    for epoch in range(num_epochs):
        for img, label in train_loader:
            img = img.to(device)

            with torch.no_grad():
                latent_real = dvae.ng_q_encode(img)
                b, emb, h, w = latent_real.size()
                real = latent_real.view(b, emb, -1).permute(2, 0, 1)
                fake = generator.sample(n_samples=b, device=device)

            inp = torch.cat([real, fake], dim=1)
            target = torch.cat([torch.ones(b, device=device), torch.zeros(b, device=device)], dim=0)

            optimizer_d.zero_grad()
            out = discriminator(inp)
            loss = criterion(out, target)
            loss.backward()
            optimizer_d.step()

            iteration += 1
            if verbose:# and (iteration % 55 == 0):
                print("MLE Dis train: Epoch {} Iter {} Loss = {}".format(epoch, iteration, round(loss.item(), 5)))





    for d_step in range(d_steps):
        s = helpers.batchwise_sample(generator, POS_NEG_SAMPLES, BATCH_SIZE)
        dis_inp, dis_target = helpers.prepare_discriminator_data(real_data_samples, s, gpu=CUDA)
        for epoch in range(epochs):
            print('d-step %d epoch %d : ' % (d_step + 1, epoch + 1), end='')
            sys.stdout.flush()
            total_loss = 0
            total_acc = 0

            for i in range(0, 2 * POS_NEG_SAMPLES, BATCH_SIZE):
                inp, target = dis_inp[i:i + BATCH_SIZE], dis_target[i:i + BATCH_SIZE]
                dis_opt.zero_grad()
                out = discriminator.batchClassify(inp)
                loss_fn = nn.BCELoss()
                loss = loss_fn(out, target)
                loss.backward()
                dis_opt.step()

