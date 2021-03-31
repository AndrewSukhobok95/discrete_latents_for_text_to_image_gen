import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from transformers import BertTokenizer, BertModel
from torch.utils.tensorboard import SummaryWriter
from modules.tadvae_generator.model import Generator
from modules.tadvae_discriminator.model import Discriminator
from config_reader import ConfigReader
from train_utils.utils import zeros_like, ones_like
from train_utils.data_utils import get_cub_dataloaders


# CONFIG = ConfigReader(config_path="/home/andrey/Aalto/thesis/TA-VQVAE/configs/tadvae_cub_local.yaml")
CONFIG = ConfigReader(config_path="/u/82/sukhoba1/unix/Desktop/TA-VQVAE/configs/tadvae_cub_remote.yaml")
CONFIG.print_config_info()

writer = SummaryWriter()

BERT_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
BERT_model = BertModel.from_pretrained('bert-base-uncased')

G = Generator(
    img_embedding_dim=8192,
    text_embedding_dim=BERT_model.config.hidden_size,
    n_trd_blocks=4,
    num_trd_block_for_mask=3,
    n_attn_heads=4,
    linear_hidden_dim=1024,
    dropout_prob=0.1,
    n_img_hidden_positions=32*32)
G.load_dvae_weights(CONFIG.load_dvae_path, CONFIG.dvae_model_name)

D = Discriminator(
    txt_in_dim=BERT_model.config.hidden_size,
    hidden_dim=512)

optimizer_G = optim.Adam(G.get_rebuild_params(), lr=CONFIG.LR)
optimizer_D = optim.Adam(D.parameters(), lr=CONFIG.LR)

lr_scheduler_G = MultiStepLR(optimizer_G, milestones=CONFIG.step_LR_milestones, gamma=CONFIG.LR_gamma)
lr_scheduler_D = MultiStepLR(optimizer_D, milestones=CONFIG.step_LR_milestones, gamma=CONFIG.LR_gamma)

train_loader, _ = get_cub_dataloaders(tokenizer=BERT_tokenizer,
                                      root_img_path=CONFIG.root_img_path,
                                      root_text_path=CONFIG.root_text_path,
                                      imgs_list_file_path=CONFIG.imgs_list_file_path,
                                      img_size=CONFIG.img_size,
                                      batch_size=CONFIG.BATCH_SIZE)

if __name__ == '__main__':
    print("Device in use: {}".format(CONFIG.DEVICE))

    G.dvae.eval()
    G.text_rebuild_block.train()
    D.train()
    BERT_model.train()

    G.to(CONFIG.DEVICE)
    D.to(CONFIG.DEVICE)
    BERT_model.to(CONFIG.DEVICE)

    iteration = 0
    for epoch in range(CONFIG.NUM_EPOCHS):
        for imgs, text_info in train_loader:
            token_tensor, token_type_tensor, mask_tensor, _ = text_info

            imgs = imgs.to(CONFIG.DEVICE)
            token_tensor = token_tensor.to(CONFIG.DEVICE)
            token_type_tensor = token_type_tensor.to(CONFIG.DEVICE)
            mask_tensor = mask_tensor.to(CONFIG.DEVICE)

            with torch.no_grad():
                imgs_recon = G.dvae(imgs)

            bert_output = BERT_model(token_tensor, attention_mask=mask_tensor, token_type_ids=token_type_tensor)
            text_embeddings = bert_output[0]

            text_embeddings_neg = torch.cat((text_embeddings[:, -1, :].unsqueeze(1), text_embeddings[:, :-1, :]), 1)
            mask_tensor_neg = torch.cat((mask_tensor[-1, :].unsqueeze(0), mask_tensor[:-1, :]), 0)

            # UPDATE DISCRIMINATOR: REAL IMAGE
            real_logit, real_c_prob = D(img=imgs_recon, txt=text_embeddings, txt_mask=mask_tensor)
            real_c_prob_neg = D(img=imgs_recon, txt=text_embeddings_neg, txt_mask=mask_tensor_neg, only_conditional=True)

            D_real_loss = F.binary_cross_entropy_with_logits(real_logit, ones_like(real_logit))
            real_c_pos_loss = F.binary_cross_entropy(real_c_prob, ones_like(real_c_prob))
            real_c_neg_loss = F.binary_cross_entropy(real_c_prob_neg, zeros_like(real_c_prob_neg))
            D_real_loss = D_real_loss + CONFIG.lambda_cond_loss * (real_c_pos_loss + real_c_neg_loss) / 2

            writer.add_scalar('D/D_real_loss', D_real_loss.item(), iteration)
            writer.add_scalar('D/D_real_c_pos_loss', real_c_pos_loss.item(), iteration)
            writer.add_scalar('D/D_real_c_neg_loss', real_c_neg_loss.item(), iteration)

            D_real_loss.backward(retain_graph=True)

            # UPDATE DISCRIMINATOR: SYNTHESIZED IMAGE
            with torch.no_grad():
                gen_img = G(img=imgs, txt_h=text_embeddings_neg, txt_pad_mask=mask_tensor_neg)
            fake_logit, _ = D(img=gen_img, txt=text_embeddings_neg, txt_mask=mask_tensor_neg)

            D_fake_loss = F.binary_cross_entropy_with_logits(fake_logit, zeros_like(fake_logit))

            writer.add_scalar('D/D_fake_loss', D_fake_loss.item(), iteration)

            D_fake_loss.backward(retain_graph=True)

            optimizer_D.step()
            D.zero_grad()

            # UPDATE GENERATOR: CONDITIONAL
            gen_img = G(img=imgs, txt_h=text_embeddings_neg, txt_pad_mask=mask_tensor_neg)
            fake_logit, fake_c_prob = D(gen_img, txt=text_embeddings_neg, txt_mask=mask_tensor_neg)

            fake_loss = F.binary_cross_entropy_with_logits(fake_logit, ones_like(fake_logit))
            fake_c_loss = F.binary_cross_entropy(fake_c_prob, ones_like(fake_c_prob))

            G_cond_loss = fake_loss + CONFIG.lambda_cond_loss * fake_c_loss

            writer.add_scalar('G/G_cond_fake_loss', fake_loss.item(), iteration)
            writer.add_scalar('G/G_cond_fake_c_loss', fake_c_loss.item(), iteration)
            writer.add_scalar('G/G_cond_loss', G_cond_loss.item(), iteration)

            G_cond_loss.backward(retain_graph=True)

            # UPDATE GENERATOR: RECONSTRUCTION
            gen_img = G(img=imgs, txt_h=text_embeddings, txt_pad_mask=mask_tensor)

            recon_loss = F.l1_loss(gen_img, imgs_recon)
            G_recon_loss = CONFIG.lambda_recon_loss * recon_loss

            writer.add_scalar('G/G_recon_loss', recon_loss.item(), iteration)

            G_recon_loss.backward(retain_graph=True)

            optimizer_G.step()
            G.zero_grad()

            print("Epoch: {} Iter: {}".format(epoch, iteration))

            iteration += 1

        lr_scheduler_G.step()
        lr_scheduler_D.step()

        G.save_rebuild_model(root_path=CONFIG.save_model_path, model_name=CONFIG.save_model_name)
        D.save_model(root_path=CONFIG.save_model_path, model_name=CONFIG.save_model_name)

        #print()




