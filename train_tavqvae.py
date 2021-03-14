import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer, BertModel
from datasets.cub import CubDataset
from datasets.common import Collater
from config import Config
from modules.tavqvae.generator import Generator
from modules.tavqvae.tagan_discriminator import Discriminator


def label_like(label, x):
    assert label == 0 or label == 1
    v = torch.zeros_like(x) if label == 0 else torch.ones_like(x)
    v = v.to(x.device)
    return v


def zeros_like(x):
    return label_like(0, x)


def ones_like(x):
    return label_like(1, x)


CONFIG = Config(local=False, model_path="models/tavqvae_e256x8138/")
CONFIG.save_config()

writer = SummaryWriter()

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

dataset = CubDataset(root_img_path=CONFIG.root_img_path,
                     root_text_path=CONFIG.root_text_path,
                     imgs_list_file_path=CONFIG.imgs_list_file_path,
                     img_size=CONFIG.img_size)

train_length = int(len(dataset) * 0.95)
test_length = len(dataset) - train_length
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, lengths=[train_length, test_length], generator=torch.Generator().manual_seed(42))

collater = Collater(tokenizer=bert_tokenizer)
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=CONFIG.BATCH_SIZE,
                          shuffle=True,
                          collate_fn=collater.collate_fn)


G = Generator(num_embeddings=CONFIG.vqvae_num_embeddings,
              img_embedding_dim=CONFIG.vqvae_embedding_dim,
              text_embedding_dim=bert_model.config.hidden_size,
              commitment_cost=CONFIG.vqvae_commitment_cost,
              decay=CONFIG.vqvae_decay,
              num_x2downsamples=CONFIG.vqvae_num_x2downsamples,
              num_resid_downsample_layers=CONFIG.vqvae_num_downsample_residual_layers,
              num_resid_bottleneck_layers=CONFIG.vqvae_num_bottleneck_residual_layers,
              text_rebuild_num_residual_layers=CONFIG.text_rebuild_num_residual_layers,
              use_batch_norm=True,
              vqvae_use_conv1x1=True)
G.load_vqvae_weights(CONFIG.load_vae_path, "VQVAE")

D = Discriminator(text_embedding_dim=bert_model.config.hidden_size)

optimizer_G = optim.Adam(G.get_rebuild_parameters(), lr=CONFIG.LR)
optimizer_D = optim.Adam(D.parameters(), lr=CONFIG.LR)

lr_scheduler_G = MultiStepLR(optimizer_G, milestones=CONFIG.step_LR_milestones, gamma=CONFIG.LR_gamma)
lr_scheduler_D = MultiStepLR(optimizer_D, milestones=CONFIG.step_LR_milestones, gamma=CONFIG.LR_gamma)


if __name__ == '__main__':
    print("Device in use: {}".format(CONFIG.DEVICE))

    G.train()
    D.train()
    bert_model.eval()

    G.to(CONFIG.DEVICE)
    D.to(CONFIG.DEVICE)
    bert_model.to(CONFIG.DEVICE)

    iteration = 0
    for epoch in range(CONFIG.NUM_EPOCHS):
        for imgs, text_info in train_loader:
            token_tensor, token_type_tensor, mask_tensor = text_info

            token_tensor = token_tensor.to(CONFIG.DEVICE)
            token_type_tensor = token_type_tensor.to(CONFIG.DEVICE)
            mask_tensor = mask_tensor.to(CONFIG.DEVICE)

            with torch.no_grad():
                # See the models docstrings for the detail of the inputs
                outputs = bert_model(token_tensor,
                                     attention_mask=mask_tensor,
                                     token_type_ids=token_type_tensor)
                texth = outputs[0]
                # batch x seq_len x emb_dim --> to seq_len x batch x emb_dim
                texth = texth.transpose(1, 0)

            imgs = imgs.to(CONFIG.DEVICE)
            texth = texth.to(CONFIG.DEVICE)

            texth_neg = torch.cat((texth[:, -1, :].unsqueeze(1), texth[:, :-1, :]), 1)
            mask_tensor_neg = torch.cat((mask_tensor[-1, :].unsqueeze(0), mask_tensor[:-1, :]), 0)

            ##### UPDATE DISCRIMINATOR #####
            optimizer_D.zero_grad()

            # real images
            real_logit, real_c_prob, real_c_prob_n = D(img=imgs, txt=texth, len_txt=mask_tensor.sum(dim=1), negative=True)

            real_loss = F.binary_cross_entropy_with_logits(real_logit, ones_like(real_logit))
            real_c_pos_loss = F.binary_cross_entropy(real_c_prob, ones_like(real_c_prob))
            real_c_neg_loss = F.binary_cross_entropy(real_c_prob_n, zeros_like(real_c_prob_n))
            real_c_loss = (real_c_pos_loss + real_c_neg_loss) / 2
            real_loss = real_loss + CONFIG.tagan_lambda_cond_loss * real_c_loss

            writer.add_scalar('D/real_loss', real_loss.item(), iteration)
            writer.add_scalar('D/real_c_pos_loss', real_c_pos_loss.item(), iteration)
            writer.add_scalar('D/real_c_neg_loss', real_c_neg_loss.item(), iteration)

            real_loss.backward()

            # synthesized images
            fake, _ = G(imgh=imgs, texth=texth_neg, text_mask=mask_tensor_neg)
            fake_logit, _ = D(img=fake.detach(), txt=texth_neg, len_txt=mask_tensor_neg.sum(dim=1))

            fake_loss = F.binary_cross_entropy_with_logits(fake_logit, zeros_like(fake_logit))

            writer.add_scalar('D/fake_loss', fake_loss.item(), iteration)

            fake_loss.backward()

            optimizer_D.step()

            ##### UPDATE GENERATOR #####
            optimizer_G.zero_grad()

            fake, _ = G(imgh=imgs, texth=texth_neg, text_mask=mask_tensor_neg)
            fake_logit, fake_c_prob = D(fake, txt=texth_neg, len_txt=mask_tensor_neg.sum(dim=1))

            print(fake_c_prob)

            fake_loss = F.binary_cross_entropy_with_logits(fake_logit, ones_like(fake_logit))
            fake_c_loss = F.binary_cross_entropy(fake_c_prob, ones_like(fake_c_prob))

            G_loss = fake_loss + CONFIG.tagan_lambda_cond_loss * fake_c_loss

            writer.add_scalar('G/fake_loss', fake_loss.item(), iteration)
            writer.add_scalar('G/fake_c_loss', fake_c_loss.item(), iteration)
            writer.add_scalar('G/G_loss', G_loss.item(), iteration)

            G_loss.backward()

            # reconstruction for matching input
            recon, perplexity = G(imgh=imgs, texth=texth, text_mask=mask_tensor)

            recon_loss = F.l1_loss(recon, imgs)
            G_loss = CONFIG.tagan_lambda_recon_loss * recon_loss

            writer.add_scalar('G/recon_loss', recon_loss.item(), iteration)

            G_loss.backward()

            optimizer_G.step()

            print("Epoch: {} Iter: {}".format(epoch, iteration))

            iteration += 1

        lr_scheduler_G.step()
        lr_scheduler_D.step()

        G.save_model(root_path=CONFIG.save_model_path, model_name="TAVQVAE")
        text_rebuild_path = os.path.join(CONFIG.save_model_path, "TAVQVAE_discriminator.pth")
        torch.save(D.state_dict(), text_rebuild_path)

        print()




