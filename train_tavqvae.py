import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer, BertModel
from datasets.cub import CubDataset, CubCollater
from config import Config


CONFIG = Config(local=True, model_path="models/tavqvae_e512x8138/")
CONFIG.save_config()

writer = SummaryWriter()

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_model.eval()

train_dataset = CubDataset(root_img_path=CONFIG.root_img_path,
                           root_text_path=CONFIG.root_text_path,
                           imgs_list_file_path=CONFIG.imgs_list_file_path)
collater = CubCollater(tokenizer=bert_tokenizer)
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=CONFIG.BATCH_SIZE,
                          shuffle=True,
                          collate_fn=collater.collate)


if __name__ == '__main__':
    print("Device in use: {}".format(CONFIG.DEVICE))

    iteration = 0
    for epoch in range(CONFIG.NUM_EPOCHS):
        for imgs, text_info in train_loader:
            token_tensor, token_type_tensor, mask_tensor = text_info

            print()




