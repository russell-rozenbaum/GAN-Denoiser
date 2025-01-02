# Denoising Adversarial Network (DAN)

Denoising Autoencoder trained on a generative adversarial network (GAN). Autoencoder for use in denoising
audio signals.


## Current State

Capable of being trained on denoising sinusoidal composition signals corrupted with Gaussian White Noise (GWN):
- Sample Rate: 1024
- Min/Max Frequencies: 10Hz/256Hz

These are notable constraints, and even with these, the training process is computationally expensive when 
training on a standard laptop. 

Can assumably be extended to train on sample rates of up to 44100Hz when given proper computational power.


## Setup

Setup data folder as follows:

```
GAN-Denoiser
├── data
│   ├── resampled
│   ├── train
│   │   ├── clean
│   │   ├── mixed
│   │   └── noise
│   └── validation
│       ├── clean
│       ├── mixed
│       └── noise
└── ...
```

Then run create_data.py
This should result in folders being filled with data, and displaying plots for a random signal from each folder.
Signals should look similar to these
<!-- TODO: Insert example signal plots -->

Then run dataset.py to verify that everything is working properly (this won't actually do anything to memory)

Now we can run train_gan.py
<!-- TODO: finish training and performance anlysis implementations -->

### Except
Except I am still working on properly implementing the GAN trainingand evaluation process... trained denoiser outputs are currently... not the greatest

## Ideas

Given this is a trivial dataset, generated through random uniform and gaussian distributions, this model will by no means be efficient on real-world data alone. 

However, it may serve as a good initialization in more complex models, e.g. for use in denoising bird calls in nature settings where noise is highly abundant. Given we have full control at curating our dataset, and essentially being able to create as much data as we need, we can fine-tune dataset creation parameters like min/max frequency (to match frequnecy range of bird vocalizations), and noise distributions (to match noise found in rainforests (rain, wind, etc)).

Furthermore, there's been notable advancements in denoising methods using similar architectural methods, such as deep feature loss as utilized by [Zhang et al. 2024](https://www.sciencedirect.com/science/article/pii/S1574954124000591). 