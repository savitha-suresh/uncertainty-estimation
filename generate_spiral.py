import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset

# 1. Generate the Spiral Dataset
def generate_spiral_data(n_points=1000, noise_std=0.1):
    theta = np.linspace(0, 4 * np.pi, n_points)
    r = np.linspace(0, 1, n_points)
    x = r * np.sin(theta) + noise_std * np.random.randn(n_points)
    y = r * np.cos(theta) + noise_std * np.random.randn(n_points)
    return np.stack((x, y), axis=-1)

# 2. Diffusion Model with MC Dropout
class DiffusionModel(nn.Module):
    def __init__(self, dropout_prob=0.1):
        super(DiffusionModel, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 64)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x, t):
        t = t.float().unsqueeze(1)
        x = torch.cat([x, t], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)  # Dropout added
        x = self.fc3(x)
        return x

# 3. Prepare Data and Model
spiral_data = generate_spiral_data()
spiral_data_tensor = torch.tensor(spiral_data, dtype=torch.float32)
dataset = TensorDataset(spiral_data_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

epochs = 100
timesteps = 100

def linear_schedule(timesteps):
    beta_start = 1e-4
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

betas = linear_schedule(timesteps)
alphas = 1 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

model = DiffusionModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
for epoch in range(epochs):
    for batch in dataloader:
        data = batch[0]
        optimizer.zero_grad()
        loss = 0.0
        batch_size = data.shape[0]

        for t in range(timesteps):
            noise = torch.randn_like(data)
            sqrt_alpha_cumprod_t = torch.sqrt(alphas_cumprod[t])
            sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1 - alphas_cumprod[t])
            noisy_data = sqrt_alpha_cumprod_t * data + sqrt_one_minus_alpha_cumprod_t * noise
            t_tensor = torch.tensor([t] * batch_size)
            predicted_noise = model(noisy_data, t_tensor)
            loss += nn.MSELoss()(predicted_noise, noise)

        loss = loss / timesteps
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}")

# Monte Carlo Dropout Sampling

def generate_mc_samples(model, timesteps, alphas, alphas_cumprod, n_samples=500, mc_samples=30):
    model.train()  # Enable dropout during inference
    samples = torch.zeros(mc_samples, n_samples, 2)
    with torch.no_grad():
        for i in range(mc_samples):
            x_t = torch.randn(n_samples, 2)
            for t in reversed(range(timesteps)):
                t_tensor = torch.tensor([t] * n_samples)
                predicted_noise = model(x_t, t_tensor)

                beta_t = betas[t]
                if t > 0:
                    alpha_t_minus_one = alphas_cumprod[t-1]
                    alpha_t = alphas_cumprod[t]
                    sigma_t = torch.sqrt(beta_t*(1-alpha_t_minus_one)/(1-alpha_t))
                else:
                    sigma_t = torch.sqrt(betas[0])

                noise = torch.randn_like(x_t)
                mean = (1 / torch.sqrt(alphas[t])) * (x_t - (betas[t] / torch.sqrt(1 - alphas_cumprod[t])) * predicted_noise)
                x_t = mean + sigma_t * noise
            samples[i] = x_t
    
    return samples, samples.mean(dim=0), samples.std(dim=0)  # Mean and standard deviation

# Generate Samples and Uncertainty
samples, mean_samples, uncertainty = generate_mc_samples(model, timesteps, alphas, alphas_cumprod, n_samples=500)

# Plot Uncertainty Heatmap
plt.figure(figsize=(8, 6))
all_x = samples[:, :, 0].flatten()
all_y = samples[:, :, 1].flatten()
sns.kdeplot(x=all_x, y=all_y, cmap="coolwarm", fill=True)

plt.scatter(spiral_data[:, 0], spiral_data[:, 1], s=5, alpha=0.2, label="Original Spiral Data")
#plt.scatter(mean_samples[:, 0].numpy(), mean_samples[:, 1].numpy(), c='red', s=5, label="Generated Data")
plt.title("Uncertainty Heatmap of Generated Data")
plt.legend()
plt.show()
