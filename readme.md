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

GAN-Denoiser
|_ data
   |_ resampled
   |_ train
      |_ clean
      |_ mixed
      |_ noise
   |_ validation
      |_ clean
      |_ mixed
      |_ noise
|_ ...

Then run create_data.py
This should result in folders being filled with data, and displaying plots for a random signal from each folder.
Signals should look similar to these
TODO: Insert example signal plots

Then run dataset.py to verify that everything is working properly (this won't actually do anything to memory)

Now we can run train_gan.py
TODO: finish training and performance anlysis implementations