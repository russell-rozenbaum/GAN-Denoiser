# GAN Denoiser (or DAN - Denoising Adversarial Network)

Denoising architecture utilizing the methods behind GANs. 

## Ideas

A scratchpad for a speculative specification:
Initial baby steps...
- Generate clean audio and noise audio. Mix to create noisey dataset.
- Generator acts as denoiser. Discriminator is fed denoised and clean waveforms, attempts to catch
the signal who was originally noisey
Transfer learning! 
- Pretrain on source task involving sinusoidal waves, 400-8000Hz (common bird frequency ranges). Apply noise in this source task using gaussian white noise (GWN), attempting to closely
imitate that of rainforest noise such as rain, wind, etc. 
- Train for target task on smaller dataset involving clean bird sounds and noisey rainforest sounds. Mix these, and analyze performance.

Further ideas:
- Look into 2D spectrograms rather than 1D


### Short version

