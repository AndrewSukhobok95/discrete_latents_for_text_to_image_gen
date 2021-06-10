# TA-VQVAE


## Config Rules

dvae_{dataset}_v{vocab}_ds{num_downsample}\_{machine}.yaml

Example:
- dvae_mnistmd_v256_ds2_remote.yaml
- dvae_mnistmd_v256_ds3_remote.yaml

## Model Storing Rules

- models
    - dataset
        - dvae_v{vocab}_ds{num_downsample}


## Base implementations

- TAGAN: https://github.com/woozzu/tagan
- DALLE: https://github.com/openai/DALL-E

## Data

#### CUB

- Text captions: https://github.com/taoxugit/AttnGAN
- Images: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html

#### Multi-Digit MNIST

- https://github.com/shaohua0116/MultiDigitMNIST
