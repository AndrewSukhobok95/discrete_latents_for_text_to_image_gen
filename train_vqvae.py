import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataload.cub import CubDataset, cub_collate
from vqvae.model import VQVAE


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = CubDataset(root_img_path="/home/andrey/Aalto/TA-VQVAE/data/CUB/CUB_200_2011/images",
                           root_text_path="/home/andrey/Aalto/TA-VQVAE/data/CUB/text",
                           imgs_list_file_path="/home/andrey/Aalto/TA-VQVAE/data/CUB/CUB_200_2011/images.txt")
train_loader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=True, collate_fn=cub_collate)

model = VQVAE(num_embeddings=1024, embedding_dim=256, commitment_cost=0.25, decay=0.99, num_x2downsamples=3)
optimizer = optim.Adam(model.parameters(), lr=1e-3, amsgrad=False)


if __name__ == '__main__':
    model.train()

    for imgs, _ in train_loader:
        imgs = imgs.to(device)
        optimizer.zero_grad()

        vq_loss, data_recon, perplexity = model(imgs)
        recon_error = F.mse_loss(data_recon, imgs)
        loss = recon_error + vq_loss
        loss.backward()

        optimizer.step()

    # x, _ = next(iter(train_loader))
    # print()
