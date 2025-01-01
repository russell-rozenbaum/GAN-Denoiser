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


### ...hmm

