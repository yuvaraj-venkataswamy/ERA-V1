# ERA V1 - Session 20 Assignment

## Stable Diffusion
Stable Diffusion is based on a particular type of diffusion model called Latent Diffusion. General diffusion models are machine learning systems that are trained to denoise random gaussian noise step by step, to get to a sample of interest, such as an image.

Latent diffusion can reduce the memory and compute complexity by applying the diffusion process over a lower dimensional latent space, instead of using the actual pixel space. This is the key difference between standard diffusion and latent diffusion models: in latent diffusion the model is trained to generate latent (compressed) representations of the images.

There are three main components in latent diffusion.

1. An autoencoder (VAE).
2. A U-Net.
3. A text-encoder, e.g. CLIP's Text Encoder.

**1. The autoencoder (VAE)**

The VAE model has two parts, an encoder and a decoder. The encoder is used to convert the image into a low dimensional latent representation, which will serve as the input to the *U-Net* model. The decoder transforms the latent representation back into an image.

**2. The U-Net**

The U-Net has an encoder part and a decoder part both comprised of ResNet blocks.
The encoder compresses an image representation into a lower resolution image representation and the decoder decodes the lower resolution image representation back to the original higher resolution image representation that is supposedly less noisy.

**3. The Text-encoder**

The text-encoder is responsible for transforming the input prompt, *e.g.* "An astronout riding a horse" into an embedding space that can be understood by the U-Net. It is usually a simple *transformer-based* encoder that maps a sequence of input tokens to a sequence of latent text-embeddings.

![alt text](https://github.com/yuvaraj-venkataswamy/ERA-V1/blob/main/session19/Images/segment-anything-pipeline.gif)

The stable diffusion model takes both a latent seed and a text prompt as an input. The latent seed is then used to generate random latent image representations of size  64×64  where as the text prompt is transformed to text embeddings of size  77×768  via CLIP's text encoder.

Next the U-Net iteratively denoises the random latent image representations while being conditioned on the text embeddings. The output of the U-Net, being the noise residual, is used to compute a denoised latent image representation via a scheduler algorithm. 

## Result 1
```
prompt = "many astronauts walking in the moon"
image = pipe(prompt, height=512, width=768).images[0]
image
```
![alt text](https://github.com/yuvaraj-venkataswamy/ERA-V1/blob/main/session19/Images/segment-anything-pipeline.gif)

## Result 2
```
prompt = "A girl cutting pizza"
image = pipe(prompt, height=512, width=768).images[0]
image
```
![alt text](https://github.com/yuvaraj-venkataswamy/ERA-V1/blob/main/session19/Images/segment-anything-pipeline.gif)

## Result 3
```
prompt = "A boy painting a picture of tree"
image = pipe(prompt, height=512, width=768).images[0]
image
```
![alt text](https://github.com/yuvaraj-venkataswamy/ERA-V1/blob/main/session19/Images/segment-anything-pipeline.gif)

## References
1. https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/stable_diffusion.ipynb
2. https://arxiv.org/abs/2112.10752
