import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import torch
import torchaudio
import torchaudio.transforms as T

from torch import nn
from torch.utils.data import DataLoader, Dataset


# ---------- Routines for computing the measurements and the approximate spectrogram ----------
def comp_measurements(input):
    spec_subroutine = T.Spectrogram(n_fft = 2*64, hop_length = 64, power = 1)
    return spec_subroutine(input)
def comp_spectrogram(input):
    spec_subroutine = T.Spectrogram(n_fft = 8*32, hop_length = 32, power = 1)
    spec_subroutine.cuda(0)
    return spec_subroutine(input)
# ---------- ---------- ----------


# ---------- Dataset class for phase retrieval ----------
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
        measurements = comp_measurements(audio)

        return measurements, audio
# ---------- ---------- ----------


# ---------- Neural network model ----------
class nn_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=8190, out_features=25000,
                    bias=True),
            nn.SELU(),
            nn.Linear(in_features=25000, out_features=25000,
                    bias=True),
            nn.SELU(),
            nn.Linear(in_features=25000, out_features=8000,
                    bias=True)
        )

    def forward(self, x: torch.Tensor):
        return self.layers(x)
# ---------- ---------- ----------


# ---------- Setup ----------
# Set seeds for reproducability
np.random.seed(17)
torch.manual_seed(17)

# Set up the dataloaders
training_data = AudioMNISTSpectrogram('Train.csv')
test_data = AudioMNISTSpectrogram('Test.csv')
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=False)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)

# Set up the model, loss, and optimiser
model = nn_model().cuda(0)
loss_fn = nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=2.5e-5)
# ----------- ---------- ----------


# ----------- Training - Phase I ----------
# Train loss and test loss saving
train_loss, test_loss = np.zeros(50), np.zeros(50)
opt_loss = torch.finfo(torch.torch.float32).max

# Timing
start_time = time.time()
print(f"--- Training - Phase I ---")

# Training loop
for epoch in range(50):

    # Make sure gradient tracking is on
    model.train(True)

    # Train for one epoch
    for measurements, audio in train_dataloader:

        # Computing predictions and losses
        measurements = measurements.cuda(0)
        audio = audio.cuda(0).squeeze()
        predictions = model(measurements)
        loss = loss_fn(comp_spectrogram(predictions), comp_spectrogram(audio))

        # Backpropagation
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()

        # Store the loss
        train_loss[epoch] += loss

    # Average loss
    train_loss[epoch] /= len(train_dataloader)

    # Set the model to evaluation mode 
    model.eval()

    # Check the performance on the test set.
    with torch.no_grad():
        for measurements, audio in test_dataloader:

            measurements = measurements.cuda(0)
            audio = audio.cuda(0).squeeze()
            predictions = model(measurements)
            test_loss[epoch] += loss_fn(comp_spectrogram(predictions), comp_spectrogram(audio))

    # Average the test loss
    test_loss[epoch] /= len(test_dataloader)

    # Print the train and test losses after the epoch 
    print(f"Epoch {epoch+1} | Train loss = {train_loss[epoch]:.3f}, Test loss = {test_loss[epoch]:.3f}")

    # Save the model and optimal loss
    if epoch == 0 or test_loss[epoch] < opt_loss:
        opt_loss = test_loss[epoch]
        epochs_phaseI = epoch+1
        time_phaseI = time.time() - start_time
        timeperepoch_phaseI = time_phaseI/epochs_phaseI
        torch.save(model, f"Results8000/model_phaseI.pt")

# Time the code
total_time = time.time()-start_time
print(f"Total training time = {total_time:.1f}")
print(f"Training time per epoch = {(total_time/50):.1f}")

# Some output
test_loss_phaseI = test_loss[:epochs_phaseI]
train_loss_phaseI = train_loss[:epochs_phaseI]
print(f"Phase I finished after {epochs_phaseI} epochs ({time_phaseI:.1f} s, {timeperepoch_phaseI:.1f} s per epoch).")
print(f"Reached a train loss of {test_loss_phaseI[-1]:.1f}")

# Plot the resulting loss curve 
epochs = list(range(1, 51))
fig = plt.figure(figsize=(10, 6))
plt.plot(epochs, test_loss, label='Test loss', linestyle='-')
plt.plot(epochs, train_loss, label='Training loss', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
fig.savefig(f"Results8000/plt_loss_phaseI.png", bbox_inches="tight")
# ---------- ---------- ----------


# ----------- Training - Phase II ----------
# Load model and decrease learning rate
model = torch.load("Results8000/model_phaseI.pt")
optimiser = torch.optim.Adam(model.parameters(), lr=2.5e-5/2)

# Train loss and test loss saving
train_loss, test_loss = np.zeros(int(epochs_phaseI//2+1)), np.zeros(int(epochs_phaseI//2+1))

# Presave the model in case no learning occurs
torch.save(model, "Results8000/model_phaseII.pt")
epochs_phaseII = 0

# Timing
start_time = time.time()
print(f"--- Training - Phase II ---")

# Training loop
for epoch in range(int(epochs_phaseI//2+1)):

    # Make sure gradient tracking is on
    model.train(True)

    # Train for one epoch
    for measurements, audio in train_dataloader:

        # Computing predictions and losses
        measurements = measurements.cuda(0)
        audio = audio.cuda(0).squeeze()
        predictions = model(measurements)
        loss = loss_fn(comp_spectrogram(predictions), comp_spectrogram(audio))

        # Backpropagation
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()

        # Store the loss
        train_loss[epoch] += loss

    # Average loss
    train_loss[epoch] /= len(train_dataloader)

    # Set the model to evaluation mode 
    model.eval()

    # Check the performance on the test set.
    with torch.no_grad():
        for measurements, audio in test_dataloader:

            measurements = measurements.cuda(0)
            audio = audio.cuda(0).squeeze()
            predictions = model(measurements)
            test_loss[epoch] += loss_fn(comp_spectrogram(predictions), comp_spectrogram(audio))

    # Average the test loss
    test_loss[epoch] /= len(test_dataloader)

    # Print the train and test losses after the epoch 
    print(f"Epoch {epoch+epochs_phaseI+1} | Train loss = {train_loss[epoch]:.3f}, Test loss = {test_loss[epoch]:.3f}")

    # Save the model and optimal loss
    if test_loss[epoch] < opt_loss:
        opt_loss = test_loss[epoch]
        epochs_phaseII = epoch+1
        time_phaseII = time.time() - start_time
        timeperepoch_phaseII = time_phaseII/epochs_phaseII
        torch.save(model, f"Results8000/model_phaseII.pt")

# Time the code
total_time = time.time()-start_time
print(f"Total training time = {total_time:.1f}")
print(f"Training time per epoch = {(total_time/int(epochs_phaseI//2+1)):.1f}")

# Some output
if epochs_phaseII > 0:
    test_loss_phaseII = test_loss[:epochs_phaseII]
    train_loss_phaseII = train_loss[:epochs_phaseII]
    print(f"Phase II finished after {epochs_phaseII} epochs ({time_phaseII:.1f} s, {timeperepoch_phaseII:.1f} s per epoch).")
    print(f"Reached a train loss of {test_loss_phaseII[-1]:.1f}")
else:
    test_loss_phaseII = np.array([])
    train_loss_phaseII = np.array([])
    print("No learning in phase II.")

# Plot the resulting loss curve 
epochs = list(range(epochs_phaseI+1, epochs_phaseI+int(epochs_phaseI//2+1)+1))
fig = plt.figure(figsize=(10, 6))
plt.plot(epochs, test_loss, label='Test loss', linestyle='-')
plt.plot(epochs, train_loss, label='Training loss', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
fig.savefig(f"Results8000/plt_loss_phaseII.png", bbox_inches="tight")
# ---------- ---------- ----------


# ----------- Training - Phase III ----------
# Compute the number of epochs needed to reach machine precision and load the model
num_epochs = int(23+np.log2(2.5e-5/2))
model = torch.load("Results8000/model_phaseII.pt")

# Train loss and test loss saving
train_loss, test_loss = np.zeros(num_epochs), np.zeros(num_epochs)

# Pre-save the model in case no learning happens
torch.save(model, "Results8000/final.pt")
epochs_phaseIII = []
time_phaseIII = 0

# Timing
start_timeI = time.time()
start_timeII = time.time()
print(f"--- Training - Phase III ---")

# Training loop
for epoch in range(num_epochs):

    # Scale down the learning rate appropriately
    lr1 = 2.5e-5/2 / 2**(epoch+1)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr1)

    # Make sure gradient tracking is on
    model.train(True)

    # Train for one epoch
    for measurements, audio in train_dataloader:

        # Computing predictions and losses
        measurements = measurements.cuda(0)
        audio = audio.cuda(0).squeeze()
        predictions = model(measurements)
        loss = loss_fn(comp_spectrogram(predictions), comp_spectrogram(audio))

        # Backpropagation
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()

        # Store the loss
        train_loss[epoch] += loss

    # Average loss
    train_loss[epoch] /= len(train_dataloader)

    # Set the model to evaluation mode 
    model.eval()

    # Check the performance on the test set.
    with torch.no_grad():
        for measurements, audio in test_dataloader:

            measurements = measurements.cuda(0)
            audio = audio.cuda(0).squeeze()
            predictions = model(measurements)
            test_loss[epoch] += loss_fn(comp_spectrogram(predictions), comp_spectrogram(audio))

    # Average the test loss
    test_loss[epoch] /= len(test_dataloader)

    # Print the train and test losses after the epoch 
    print(f"Epoch {epoch+epochs_phaseI+epochs_phaseII+1} | Train loss = {train_loss[epoch]:.3f}, Test loss = {test_loss[epoch]:.3f}")

    # Save the model and optimal loss
    if test_loss[epoch] < opt_loss:

        opt_loss = test_loss[epoch]
        epochs_phaseIII.append(epoch)
        time_phaseIII += time.time() - start_timeII
        torch.save(model, f"Results8000/final.pt")
        start_timeII = time.time()

    else:

        model = torch.load("Results8000/final.pt")
        start_timeII = time.time()

# Time the code
total_time = time.time()-start_timeI
print(f"Total training time = {total_time:.1f}")
print(f"Training time per epoch = {(total_time/num_epochs):.1f}")

# Some output
if len(epochs_phaseIII):
    test_loss_phaseIII = test_loss[epochs_phaseIII]
    train_loss_phaseIII = train_loss[epochs_phaseIII]
    timeperepoch_phaseIII = time_phaseIII/len(epochs_phaseIII)
    print(f"Phase III finished after {len(epochs_phaseIII)} epochs ({time_phaseIII:.1f} s, {timeperepoch_phaseIII:.1f} s per epoch).")
    print(f"Reached a train loss of {test_loss_phaseIII[-1]:.1f}")
else:
    print("No improvement in phase III.")
    test_loss_phaseII = np.array([])
    train_loss_phaseII = np.array([])

# Plot the resulting loss curve 
epochs = list(range(epochs_phaseI+epochs_phaseII+1, epochs_phaseI+epochs_phaseII+num_epochs+1))
fig = plt.figure(figsize=(10, 6))
plt.plot(epochs, test_loss, label='Test loss', linestyle='-')
plt.plot(epochs, train_loss, label='Training loss', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
fig.savefig(f"Results8000/plt_loss_phaseIII.png", bbox_inches="tight")
# ---------- ---------- ----------


# Some final plotting
epochs = list(range(1, epochs_phaseI+epochs_phaseII+len(epochs_phaseIII)+1))
fig = plt.figure(figsize=(10, 6))
plt.plot(epochs, np.concatenate((test_loss_phaseI, test_loss_phaseII, test_loss_phaseIII)), label='Test loss', linestyle='-')
plt.plot(epochs, np.concatenate((train_loss_phaseI, train_loss_phaseII, train_loss_phaseIII)), label='Training loss', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
fig.savefig(f"Results8000/plt_loss.png", bbox_inches="tight")

# Plotting some examples
# Extract data
measurements, signals = next(iter(test_dataloader))

# Plotting
fig, axs = plt.subplots(3,3)
model.eval()
with torch.no_grad():

    # Plotting the results
    for i in range(3):
        for j in range(3):
            axs[i,j].plot(np.linspace(0, 1, 8000, endpoint=False), 
                          signals[3*i+j].squeeze(),
                          np.linspace(0, 1, 8000, endpoint=False),
                          model(measurements[3*i+j].cuda(0)).cpu().squeeze(), alpha=0.8)        
    fig.savefig(f"Results8000/rec.png", bbox_inches="tight")

    # Clearing the figure
    fig, axs = plt.subplots(3,3)

    # Plotting the measurements
    for i in range(3):
        for j in range(3):
            axs[i,j].imshow(measurements[3*i+j].squeeze(),
                            origin='lower',
                            aspect = 'auto',
                            extent=(0,1,0,4001)
            )
    fig.savefig(f"Results8000/meas.png", bbox_inches="tight")