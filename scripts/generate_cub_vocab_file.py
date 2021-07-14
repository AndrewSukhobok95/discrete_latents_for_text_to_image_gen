from datasets.cub_text_indexer import TextIndexer


if __name__ == "__main__":
    ti = TextIndexer(vocab_file_path="/home/andrey/Aalto/thesis/TA-VQVAE/data/CUB/vocab.json")
    ti.collect_cub_vocab(
        root_text_path="/home/andrey/Aalto/thesis/TA-VQVAE/data/CUB/text",
        imgs_list_file_path="/home/andrey/Aalto/thesis/TA-VQVAE/data/CUB/CUB_200_2011/images.txt"
    )
