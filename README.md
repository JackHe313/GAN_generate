# GAN_generate
generate GAN pics based on StudioGAN

I have push the ckpt and other "BIG" files using git Large File Storage (FLS), you may need to pull it using lfs too.

## requirement
First, install PyTorch meeting your environment (at least 1.7):
```
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```
Then, use the following command to install the rest of the libraries:
```
pip install tqdm ninja h5py kornia matplotlib pandas sklearn scipy seaborn wandb PyYaml click requests pyspng imageio-ffmpeg timm
```

## Run
Get the free gpu cuda number
```
nvidia-smi
```

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

The generated pics will locate on ```./samples/MODEL_FOLDER/fake/*/*.png```


If you need anything, contact me directly which will help me making this readme more readable and friendly.
