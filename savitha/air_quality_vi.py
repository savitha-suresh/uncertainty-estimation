from air_quality_uncertainty_estimation import get_data_air_quality, diffusion_steps, noise, sample_ddpm
from models.dm_vi import DiffusionModelVI
from blitz.utils import variational_estimator
import torch 
import torch.nn as nn
import torch.optim as optim
import logging
import sys
import numpy as np


logging.basicConfig(level=logging.INFO, stream=sys.stdout)

@variational_estimator
class WrappedDiffusionModel(nn.Module):
    def __init__(self, diffusion_model):
        super().__init__()
        self.diffusion_model = diffusion_model

    def forward(self, inputs):
        x, t = inputs  # Unpack the tuple
        return self.diffusion_model(x, t)

def train_model(model, data, diffusion_steps, device):
    epochs = 20
    batch_size = 16
    model.train()
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    
    for epoch in range(epochs):
        epoch_loss = steps = 0
        for i in range(0, len(data), batch_size):
            Xbatch = data[i:i+batch_size]
            timesteps = torch.randint(0, diffusion_steps, size=[len(Xbatch), 1])
            noised, eps = noise(Xbatch, timesteps)
            # predicted_noise = model(noised.to(device), timesteps.to(device))
            # loss = loss_fn(predicted_noise, eps.to(device))
            loss = model.sample_elbo(
                    inputs=(noised.to(device), timesteps.to(device)),
                    labels=eps.to(device),
                    criterion=loss_fn,
                    sample_nbr=3,
                    complexity_cost_weight=1e-6
                )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()    
            epoch_loss += loss
            steps += 1
        logging.info(f"Epoch {epoch} loss = {epoch_loss / steps}")

  

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    X = get_data_air_quality()
    X = torch.tensor(X.values)

    nfeatures = X.shape[1]

   
    base_model = DiffusionModelVI(nfeatures=nfeatures, nblocks=8, hidden_layer=512)
    model = WrappedDiffusionModel(base_model)
    model = model.to(device)
    
    X.to(device)
    train_model(model, X, diffusion_steps, device)
    X_last, X_hist, mc_samples = sample_ddpm(model, 10000, nfeatures, device=device)
    torch.save(mc_samples, 'mc_samples.pt')
    torch.save(X_last, 'X_last.pt')



if __name__ == "__main__":
    main()