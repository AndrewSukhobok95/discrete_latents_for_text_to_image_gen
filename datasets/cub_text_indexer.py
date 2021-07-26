import os
import re
import json
from datasets.datasets.cub import CubDataset
from collections import Counter

class TextIndexer:
    def __init__(self, vocab_file_path):
        self.vocab_file_path = vocab_file_path
        with open(self.vocab_file_path) as json_file:
            self.word_to_id = json.load(json_file)
        self.id_to_word = dict([(v, k) for k, v in self.word_to_id.items()])
        self.pad_token = "[PAD_TOKEN]"
        self.sos_token = "[SOS_TOKEN]"
        self.eos_token = "[EOS_TOKEN]"

    def collect_cub_vocab(self, root_text_path, imgs_list_file_path):
        dataset = CubDataset(
            root_img_path=None,
            root_text_path=root_text_path,
            imgs_list_file_path=imgs_list_file_path,
            transforms=None,
            img_size=256,
            return_all_texts=True)

        # tokens_dict = dict()
        tokens_set = set()
        for t in range(len(dataset)):
            texts = dataset.get_texts_only(t)
            new_tokens = sum(map(self.tokenize, texts), [])
            tokens_set.update(new_tokens)
            # td = dict(Counter(new_tokens))
            # for k,v in td.items():
            #     if k in tokens_dict.keys():
            #         tokens_dict[k] = tokens_dict[k] + v
            #     else:
            #         tokens_dict[k] = v

        tokens_list = [self.pad_token, self.sos_token, self.eos_token] + list(tokens_set)
        word_to_id = dict(zip(tokens_list, range(len(tokens_list))))
        with open(self.vocab_file_path, 'w') as outfile:
            json.dump(word_to_id, outfile, indent=4)

    def tokenize(self, s):
        s = s.replace('\n', '')
        tokens = re.findall(r'\b\w+\b', s)
        return tokens

    def text2ids(self, text):
        tokens = self.tokenize(text)
        tokens_ids = []
        for t in tokens:
            tokens_ids.append(self.word_to_id.get(t))
        return tokens_ids

    def ids2text(self, tokens):
        text = []
        for t in tokens:
            text.append(self.id_to_word.get(t))
        return ' '.join(text)


if __name__ == "__main__":
    ti = TextIndexer(vocab_file_path="/home/andrey/Aalto/thesis/TA-VQVAE/data/CUB/vocab.json")
    ti.ids2text([[1,2,3], [3,4,5]])
    ti.text2ids('solid black bird with a medium beak and a yellow eye.')


