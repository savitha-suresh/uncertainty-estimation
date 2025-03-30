import torch
import torch.nn as nn
import torch.optim as optim
from ucimlrepo import fetch_ucirepo
import pandas as pd
from models.v2 import DiffusionModelV2
from models.v1 import DiffusionModel

diffusion_steps = 250  # Number of steps in the diffusion process

# Set noising variances betas as in Nichol and Dariwal paper (https://arxiv.org/pdf/2102.09672.pdf)
s = 0.008
timesteps = torch.tensor(range(0, diffusion_steps), dtype=torch.float32)
schedule = torch.cos((timesteps / diffusion_steps + s) / (1 + s) * torch.pi / 2)**2

baralphas = schedule / schedule[0]
betas = 1 - baralphas / torch.concatenate([baralphas[0:1], baralphas[0:-1]])
alphas = 1 - betas



def noise(Xbatch, t):
    eps = torch.randn_like(Xbatch)
    noised = (baralphas[t] ** 0.5).repeat(1, Xbatch.shape[1]) * Xbatch + ((1 - baralphas[t]) ** 0.5).repeat(1, Xbatch.shape[1]) * eps
    return noised, eps

def train_model(model, data, diffusion_steps, device):
    epochs = 100
    batch_size = 10

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    
    for epoch in range(epochs):
        epoch_loss = steps = 0
        for i in range(0, len(data), batch_size):
            Xbatch = data[i:i+batch_size]
            timesteps = torch.randint(0, diffusion_steps, size=[len(Xbatch), 1])
            noised, eps = noise(Xbatch, timesteps)
            predicted_noise = model(noised.to(device), timesteps.to(device))
            loss = loss_fn(predicted_noise, eps.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()    
            epoch_loss += loss
            steps += 1
        if epoch % 10 == 0:
            print(f"Epoch {epoch} loss = {epoch_loss / steps}")


def sample_ddpm(model, nsamples, nfeatures, mc_samples=30, device='cpu'):
    """Sampler following the Denoising Diffusion Probabilistic Models method by Ho et al (Algorithm 2)"""
    model.train()
    samples = torch.zeros(mc_samples, nsamples, nfeatures).to(device)
    with torch.no_grad():
        for i in range(mc_samples):
            x = torch.randn(size=(nsamples, nfeatures)).to(device)
            xt = [x]
            for t in range(diffusion_steps-1, 0, -1):
                predicted_noise = model(x, torch.full([nsamples, 1], t).to(device))
                # See DDPM paper between equations 11 and 12
                x = 1 / (alphas[t] ** 0.5) * (x - (1 - alphas[t]) / ((1-baralphas[t]) ** 0.5) * predicted_noise)
                if t > 1:
                    # See DDPM paper section 3.2.
                    # Choosing the variance through beta_t is optimal for x_0 a normal distribution
                    variance = betas[t]
                    std = variance ** (0.5)
                    x += std * torch.randn(size=(nsamples, nfeatures)).to(device)
                xt += [x]
            samples[i] = x
    return x, xt, samples


def main():
    air_quality = fetch_ucirepo(id=360)
    X = air_quality.data.features
    X['Time'] = pd.to_datetime(X['Time'], format='%H:%M:%S')
    X['Hour'] = X['Time'].dt.hour
    X = X.drop('Date', axis=1)
    X = X.drop('Time', axis=1)
    X = X.astype('float32')


    X = torch.tensor(X.values)

    nfeatures = X.shape[1]
    y = air_quality.data.targets
    model = DiffusionModelV2(nfeatures=nfeatures, nblocks=8, hidden_layer=512)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    X.to(device)
    train_model(model, X, diffusion_steps, device)
    X_last, X_hist, mc_samples = sample_ddpm(model, 10000, nfeatures)
    torch.save(mc_samples, 'mc_samples.pt')
    torch.save(X_last, 'X_last.pt')

if __name__ == "__main__":
    main()