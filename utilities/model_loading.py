from modules.dvae.model import DVAE
from modules.clip.model import CLIP, DVAECLIP
from modules.transformer_gen.ar_cond_1stream.generator import LatentGenerator as LatentGenerator1s
from modules.transformer_gen.ar_cond_2stream.generator import LatentGenerator as LatentGenerator2s


def define_CLIP(config, eval=False, load=False):
    model = CLIP(
        img_height=config.img_height,
        img_width=config.img_width,
        img_channels=config.img_channels,
        patch_height=config.patch_height,
        patch_width=config.patch_width,
        txt_max_length=config.txt_max_length,
        txt_vocab_size=config.txt_vocab_size,
        embed_dim=config.embed_dim,
        num_blocks=config.num_blocks,
        hidden_dim=config.hidden_dim,
        n_attn_heads=config.n_attn_heads,
        dropout_prob=config.dropout_prob,
        device=config.DEVICE)
    if eval:
        model.eval()
    else:
        model.train()
    if load:
        model.load_model(
            root_path=config.save_model_path,
            model_name=config.save_model_name)
    return model


def define_DVAE(config, eval=False, load=False, compound_config=False):
    model = DVAE(
        in_channels=config.in_channels,
        vocab_size=config.vocab_size,
        num_x2downsamples=config.num_x2downsamples,
        num_resids_downsample=config.num_resids_downsample,
        num_resids_bottleneck=config.num_resids_bottleneck,
        hidden_dim=config.hidden_dim,
        device=config.DEVICE)
    if eval:
        model.eval()
    else:
        model.train()
    if load and compound_config:
        model.load_model(
            root_path=config.vae_model_path,
            model_name=config.vae_model_name)
    elif load and not compound_config:
        model.load_model(
            root_path=config.save_model_path,
            model_name=config.save_model_name)
    return model


def define_LatentGenerator1s(config, eval=False, load=False, load_to_continue=False):
    model = LatentGenerator1s(
        hidden_width=config.hidden_width,
        hidden_height=config.hidden_height,
        embedding_dim=config.vocab_size,
        num_blocks=config.num_blocks,
        cond_seq_size=config.cond_seq_size,
        cond_vocab_size=config.cond_vocab_size,
        hidden_dim=config.hidden_dim,
        n_attn_heads=config.n_attn_heads,
        dropout_prob=config.dropout_prob,
        device=config.DEVICE)
    if eval:
        model.eval()
    else:
        model.train()
    if load and load_to_continue:
        model.load_model(
            root_path=config.save_model_path,
            model_name=config.save_model_name)
    elif load and not load_to_continue:
        model.load_model(
            root_path=config.load_model_path,
            model_name=config.load_model_name)
    return model


def define_LatentGenerator2s(config, eval=False, load=False, load_to_continue=False):
    model = LatentGenerator2s(
        hidden_width=config.hidden_width,
        hidden_height=config.hidden_height,
        embedding_dim=config.vocab_size,
        num_blocks=config.num_blocks,
        cond_num_blocks=config.cond_num_blocks,
        cond_seq_size=config.cond_seq_size,
        cond_vocab_size=config.cond_vocab_size,
        hidden_dim=config.hidden_dim,
        n_attn_heads=config.n_attn_heads,
        dropout_prob=config.dropout_prob,
        device=config.DEVICE)
    if eval:
        model.eval()
    else:
        model.train()
    if load and load_to_continue:
        model.load_model(
            root_path=config.save_model_path,
            model_name=config.save_model_name)
    elif load and not load_to_continue:
        model.load_model(
            root_path=config.load_model_path,
            model_name=config.load_model_name)
    return model
