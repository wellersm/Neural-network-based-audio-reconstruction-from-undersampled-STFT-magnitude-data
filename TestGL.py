import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import time
import torch
import torchaudio
import torchaudio.transforms as T

from torch import nn
from torch.utils.data import Dataset

# FGL function
def griffinlim(specgram, window, n_fft, hop_length, win_length, power, n_iter, momentum, length, angles):
    
    # momentum assert
    if not 0 <= momentum < 1:
        raise ValueError("momentum must be in range [0, 1). Found: {}".format(momentum))
    momentum = momentum / (1 + momentum)

    # pack batch
    shape = specgram.size()
    specgram = specgram.reshape([-1] + list(shape[-2:]))
    specgram = specgram.pow(1 / power)

    # initialize the previous iterate to 0
    tprev = torch.tensor(0.0, dtype=specgram.dtype, device=specgram.device)
    for _ in range(n_iter):
        # Invert with our current estimate of the phases
        inverse = torch.istft(
            specgram * angles, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, length=length
        )

        # Rebuild the spectrogram
        rebuilt = torch.stft(
            input=inverse,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )

        # Update our phase estimates
        angles = rebuilt
        if momentum:
            angles = angles - tprev.mul_(momentum)
        angles = angles.div(angles.abs().add(1e-16))

        # Store the previous iterate
        tprev = rebuilt

    # Return the final phase estimates
    waveform = torch.istft(
        specgram * angles, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, length=length
    )

    # unpack batch
    waveform = waveform.reshape(shape[:-2] + waveform.shape[-1:])

    return waveform

# Set seeds for reproducability
np.random.seed(17)
torch.manual_seed(17)

# Neural network model
class nn_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=32379, out_features=18000,
                    bias=True),
            nn.SELU(),
            nn.Linear(in_features=18000, out_features=18000,
                    bias=True),
            nn.SELU(),
            nn.Linear(in_features=18000, out_features=8000,
                    bias=True)
        )

    def forward(self, x: torch.Tensor):
        return self.layers(x)
        
# Dataset class for phase retrieval
class AudioMNISTSpectrogram(Dataset):
    def __init__(self, annotations_file):
        self.annotations = pd.read_csv(annotations_file, header=None, names=['Path', 'Label'], delimiter=',')

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):

        # Load the audio signal
        audio, _ = torchaudio.load(self.annotations.iloc[idx, 0])

        # Resample the audio signal
        resampler = T.Resample(48000, 8000)
        audio = resampler(audio)

        # Pad the audio signal to one second
        padding = 8000 - audio.size(1)
        audio = nn.functional.pad(audio, (0, padding), value=0)

        # Standardise the audio signal
        audio = (audio - torch.mean(audio))/torch.std(audio)

        # Compute the measurements
        Spectrogram = T.Spectrogram(n_fft = 256, hop_length = 32, power = 1)
        measurements = Spectrogram(audio)

        return measurements, audio

# Load the test data
test_data = AudioMNISTSpectrogram('Test.csv')

# Define Spectral Convergence and the Quotient Space Metric
Spectrogram = T.Spectrogram(n_fft = 256, hop_length = 32, power = 1).cuda(0)
def SpectralConvergence(input, prediction):
    INPUT = Spectrogram(input.squeeze())
    return 20*torch.log10( torch.linalg.matrix_norm( INPUT - Spectrogram(prediction.squeeze()) ) / torch.linalg.matrix_norm(INPUT) )
def QuotientSpaceMetric(input, prediction):
    input, prediction = input.squeeze(), prediction.squeeze()
    aligned_prediction = torch.sign( torch.inner(input, prediction) ) * prediction
    return 20*torch.log10( torch.linalg.vector_norm(input - aligned_prediction) / 89.437128755 )
        
# Load the model
model = torch.load("Models/32af.pt")
model.eval()

# Allocate space for the metrics
spectral_convergence, quotient_metric = torch.zeros(len(test_data)).cuda(0), torch.zeros(len(test_data)).cuda(0)
RawSpectrogram = T.Spectrogram(n_fft = 256, hop_length = 32, power = None).cuda(0)

# Check the performance on the test set.
with torch.no_grad():
    for i, (measurements, inputs) in enumerate(test_data):
        
        # Predict audio
        measurements = measurements.cuda(0)
        initialisations = model( measurements )
        Initialisations = RawSpectrogram( initialisations )
        angles = Initialisations.div(Initialisations.abs().add(1e-16))

        # Apply Griffin-Lim
        predictions = griffinlim(measurements, torch.hann_window(256).cuda(0), 256, 32, 256, 1, 0, 0.99, None, angles)

        # Compute the metrics
        inputs = inputs.cuda(0)
        spectral_convergence[i] = SpectralConvergence(inputs, predictions)
        quotient_metric[i] = QuotientSpaceMetric(inputs, predictions)

# Print the errors
print(f"- | sample mean | sample sd")
print(f"Spectral convergence [db] | {spectral_convergence.mean():.3f} | {spectral_convergence.std():.3f}")
print(f"Quotient space metric [db] | {quotient_metric.mean():.3f} | {quotient_metric.std():.3f}")

# Check the performance on the test set.
model_time = 0
with torch.no_grad():
    for i, (measurements, inputs) in enumerate(test_data):
        
        # Predict audio
        measurements = measurements.cuda(0)
        start_time = time.time()
        initialisations = model( measurements )
        model_time += (time.time()-start_time)
        Initialisations = RawSpectrogram( initialisations )
        angles = Initialisations.div(Initialisations.abs().add(1e-16))

        # Apply Griffin-Lim
        predictions = griffinlim(measurements, torch.hann_window(256).cuda(0), 256, 32, 256, 1, 1024, 0.99, None, angles)

        # Compute the metrics
        inputs = inputs.cuda(0)
        spectral_convergence[i] = SpectralConvergence(inputs, predictions)
        quotient_metric[i] = QuotientSpaceMetric(inputs, predictions)

# Print the errors
print(f"- | sample mean | sample sd")
print(f"Spectral convergence [db] | {spectral_convergence.mean():.3f} | {spectral_convergence.std():.3f}")
print(f"Quotient space metric [db] | {quotient_metric.mean():.3f} | {quotient_metric.std():.3f}")
print(f"Model elapsed time (on {i+1} signals): {model_time:.1f} s")

# Check the performance on the test set for regular GL
GL_time = 0
with torch.no_grad():
    for i, (measurements, inputs) in enumerate(test_data):
        
        # Predict audio
        measurements = measurements.cuda(0)
        start_time = time.time()
        angles = torch.rand(measurements.size(), dtype=torch.cfloat, device=measurements.device)

        # Apply Griffin-Lim
        predictions = griffinlim(measurements, torch.hann_window(256).cuda(0), 256, 32, 256, 1, 1024, 0.99, None, angles)
        GL_time += (time.time() - start_time)

        # Compute the metrics
        inputs = inputs.cuda(0)
        spectral_convergence[i] = SpectralConvergence(inputs, predictions)
        quotient_metric[i] = QuotientSpaceMetric(inputs, predictions)

# Print the errors
print(f"- | sample mean | sample sd")
print(f"Spectral convergence [db] | {spectral_convergence.mean():.3f} | {spectral_convergence.std():.3f}")
print(f"Quotient space metric [db] | {quotient_metric.mean():.3f} | {quotient_metric.std():.3f}")
print(f"Total elapsed time (on {i+1} signals): {GL_time:.1f} s")