# GAN_generate
generate GAN pics based on StudioGAN

I have push the ckpt and other "BIG" files using git Large File Storage (FLS), you may need to pull it using lfs too.

```
nvidia-smi
```
use this to get the free gpu cuda number

then
```
./generate [CUDA] [MODEL] [number of image to save] [version(optional)]
```
for example
```
./generate 0 BigGAN-Deep 100
```
will generate BigGAN-Deep models with random ckpt version running on CUDA 0, and save 100 generated pics

```
./generate 0 BigGAN-Deep 10 56_49
```
will generate BigGAN-Deep models with the first ckpt version having the name "56_49" running on CUDA 0, and save 10 pics

The generated pics will locate on ./samples/MODEL_FOLDER/fake/*/*.png

If you need anything, contact me directly which will help me making this readme more readable and friendly.
