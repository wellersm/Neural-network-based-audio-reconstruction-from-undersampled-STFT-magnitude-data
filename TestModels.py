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

# Set seeds for reproducability
np.random.seed(17)
torch.manual_seed(17)

# The settings for our neural networks
settings = [[4, 32, 18000, "32af.pt", 1],
            [2, 64, 22000, "16af.pt", 1],
            [1, 64, 25000, "8af.pt", 1],
            [0.5, 128, 27000, "4af.pt", 1]]
for s in range(8):

    # Set up the parameters
    oversampling = settings[s][0]
    a = settings[s][1]
    hidden_features = settings[s][2]
    modelname = settings[s][3]
    pow = settings[s][4]

    # Phase retrieval model
    class nn_model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=int((oversampling*a+1)*(8000//a + 1)), out_features=hidden_features,
                        bias=True),
                nn.SELU(),
                nn.Linear(in_features=hidden_features, out_features=hidden_features,
                        bias=True),
                nn.SELU(),
                nn.Linear(in_features=hidden_features, out_features=8000,
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
            Spectrogram = T.Spectrogram(n_fft = int(2*oversampling*a), hop_length = a, power = pow)
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
    model = torch.load("Models/"+modelname)
    model.eval()

    # Allocate space for the metrics
    spectral_convergence, quotient_metric = torch.zeros(len(test_data)).cuda(0), torch.zeros(len(test_data)).cuda(0)

    # Check the performance on the test set.
    start_time = time.time()
    with torch.no_grad():
        for i, (measurements, inputs) in enumerate(test_data):
            
            # Predict audio
            predictions = model( measurements.cuda(0) )

            # Compute the metrics
            inputs = inputs.cuda(0)
            spectral_convergence[i] = SpectralConvergence(inputs, predictions)
            quotient_metric[i] = QuotientSpaceMetric(inputs, predictions)

    # Print the errors
    print(modelname)
    print(f"- | sample mean | sample sd")
    print(f"Spectral convergence [db] | {spectral_convergence.mean():.3f} | {spectral_convergence.std():.3f}")
    print(f"Quotient space metric [db] | {quotient_metric.mean():.3f} | {quotient_metric.std():.3f}")
    print(f"Total elapsed time (on {i+1} signals): {(time.time()-start_time):.1f} s")
    # ---------- ---------- ----------


    # ---------- Saving some audio signals to listen to ----------
    # Make results directory if necessary
    if not os.path.exists("Results"):
        os.mkdir("Results")
    if not os.path.exists("Results/Test_"+modelname[:3]):
        os.mkdir("Results/Test_"+modelname[:3])

    # Plotting loop
    fig, axs = plt.subplots(2,2)
    with torch.no_grad():
        for i in range(2):
            for j in range(2):

                # Making the prediction
                prediction = model( test_data[2*i+j][0].cuda(0) ).cpu()

                # Save the audio and prediction
                torchaudio.save("Results/Test_"+modelname[:3]+f"/original_{2*i+j+1}.wav", test_data[2*i+j][1], 8000)
                torchaudio.save("Results/Test_"+modelname[:3]+f"/GriffinLim_{2*i+j+1}.wav", prediction, 8000)

                # Plotting
                axs[i,j].plot(np.linspace(0, 1, 8000, endpoint=False), 
                                test_data[2*i+j][1].squeeze(),
                                np.linspace(0, 1, 8000, endpoint=False),
                                prediction.squeeze(), alpha=0.8)        
    fig.savefig("Results/Test_"+modelname[:3]+"/rec.png", bbox_inches="tight")
    # ---------- ---------- ----------