import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid", context="talk")

# 1. Data Preparation
from ucimlrepo import fetch_ucirepo

# Fetch dataset - id=186 (wine quality)
wine_quality = fetch_ucirepo(id=186)
X = wine_quality.data.features.values   # shape (6497, 11)
y = wine_quality.data.targets.values      # shape (6497, 1)

# metadata and variable information
print("Metadata:\n", wine_quality.metadata)
print("Variables:\n", wine_quality.variables)

# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# transform
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# dataloader
batch_size = 128
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. MDN Model
class MDN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_mixtures, output_dim=1, dropout_rate=0.3):
        """
        A MDN network.

        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Number of hidden units.
            num_mixtures (int): Number of Gaussian mixture components.
            output_dim (int): Dimension of the target.
            dropout_rate (float): Dropout probability.
        """
        super(MDN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        self.dropout = nn.Dropout(dropout_rate)

        # Output layers: mixture weights (pi), means (mu), and standard deviations (sigma)
        self.pi = nn.Linear(hidden_dim, num_mixtures)
        self.mu = nn.Linear(hidden_dim, num_mixtures * output_dim)
        self.sigma = nn.Linear(hidden_dim, num_mixtures * output_dim)

        self.num_mixtures = num_mixtures
        self.output_dim = output_dim

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)

        # Mixture coeffs
        pi = torch.softmax(self.pi(x), dim=1)

        # mean for each mixture component
        mu = self.mu(x)
        mu = mu.view(-1, self.num_mixtures, self.output_dim)

        # std for each mixture component.
        sigma = self.sigma(x)
        sigma = sigma.view(-1, self.num_mixtures, self.output_dim)
        sigma = torch.exp(sigma)  # for positivity

        return pi, mu, sigma

# 3. Loss Function
def mdn_loss(pi, mu, sigma, target):
    """
    Negative log-likelihood of the target.

    Args:
        pi: Tensor of shape (batch, num_mixtures) with mixing coefficients.
        mu: Tensor of shape (batch, num_mixtures, output_dim) with means.
        sigma: Tensor of shape (batch, num_mixtures, output_dim) with std deviations.
        target: Tensor of shape (batch, output_dim).

    Returns:
        loss.
    """
    # (batch, num_mixtures, output_dim)
    target = target.unsqueeze(1).expand_as(mu)
    m = torch.distributions.Normal(loc=mu, scale=sigma)
    # log
    log_prob = m.log_prob(target)  # shape: (batch, num_mixtures, output_dim)
    log_prob = log_prob.sum(dim=2)  # shape: (batch, num_mixtures)

    # log probs by mixture coeffs.
    weighted_log_prob = torch.log(pi + 1e-8) + log_prob
    # Log-sum-exp trick over mixtures.
    log_sum = torch.logsumexp(weighted_log_prob, dim=1)

    loss = -torch.mean(log_sum)
    return loss

# 4. Init
input_dim = X_train_tensor.shape[1]  # 11 features
hidden_dim = 256
num_mixtures = 8                   # play around with this ...
output_dim = 1

model = MDN(input_dim, hidden_dim, num_mixtures, output_dim, dropout_rate=0.3).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)

# 5. Train Loop
num_epochs = 500
loss_history = []
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        pi, mu, sigma = model(batch_x)
        loss = mdn_loss(pi, mu, sigma, batch_y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_x.size(0)

    epoch_loss = running_loss / len(train_dataset)
    loss_history.append(epoch_loss)

    scheduler.step(epoch_loss)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# 6. Inference
def predict(model, X):
    model.eval()
    with torch.no_grad():
        pi, mu, sigma = model(X)
        # squeeze the last dimension.
        mu = mu.squeeze(2)  # shape: (batch, num_mixtures)
        # sum_i pi_i * mu_i.
        expectation = (pi * mu).sum(dim=1, keepdim=True)
    return expectation

# 7. Evaluation
X_test_tensor = X_test_tensor.to(device)
y_test_tensor = y_test_tensor.to(device)

# Get preds.
y_pred_tensor = predict(model, X_test_tensor)
y_pred_np = y_pred_tensor.cpu().numpy()
y_test_np = y_test_tensor.cpu().numpy()

# inv transform to original scale
y_pred_inv = scaler_y.inverse_transform(y_pred_np)
y_test_inv = scaler_y.inverse_transform(y_test_np)

# Compute Mean Squared Error.
mse_val = mean_squared_error(y_test_inv, y_pred_inv)
r2 = r2_score(y_test_inv, y_pred_inv)
mae = mean_absolute_error(y_test_inv, y_pred_inv)

# accuracy
tolerance = 0.5  # adjust as needed
accuracy = np.mean(np.abs(y_test_inv - y_pred_inv) < tolerance)

print("\nTest Metrics:")
print("Mean Squared Error: {:.4f}".format(mse_val))
print("R² Score: {:.4f}".format(r2))
print("Mean Absolute Error: {:.4f}".format(mae))
print("Custom Accuracy (within tolerance {:.2f}): {:.2f}%".format(tolerance, accuracy * 100))

# 8. Visualizations

# (a) Training Loss Curve
plt.figure(figsize=(10, 6))
plt.plot(loss_history, color='blue', linewidth=2)
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.title("MDN Training Loss Over Epochs", fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.show()

# (b) KDE Plot: True vs Predicted Target Distribution
plt.figure(figsize=(10, 6))
sns.kdeplot(y_test_inv.flatten(), label='True Distribution', shade=True, color="skyblue", linewidth=2)
sns.kdeplot(y_pred_inv.flatten(), label='Predicted Distribution', shade=True, color="salmon", linewidth=2)
plt.xlabel("Target Value", fontsize=14)
plt.ylabel("Density", fontsize=14)
plt.title("KDE Plot: True vs Predicted Target Distribution", fontsize=16)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

# (c) Scatter Plot: True vs Predicted
plt.figure(figsize=(8, 8))
plt.scatter(y_test_inv, y_pred_inv, alpha=0.6, edgecolor='k', s=70)
min_val = min(y_test_inv.min(), y_pred_inv.min())
max_val = max(y_test_inv.max(), y_pred_inv.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label="Ideal")
plt.xlabel("True Target", fontsize=14)
plt.ylabel("Predicted Target", fontsize=14)
plt.title("Scatter Plot: True vs Predicted", fontsize=16)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

# (d) Residual Error Distribution
residuals = y_test_inv - y_pred_inv
plt.figure(figsize=(10, 6))
sns.histplot(residuals.flatten(), bins=30, kde=True, color='purple')
plt.xlabel("Residual Error (True - Predicted)", fontsize=14)
plt.title("Residual Error Distribution", fontsize=16)
plt.tight_layout()
plt.show()

# (e) Mixture Density Components for a Random Test Sample
def plot_mixture_density(model, sample_x, sample_y, scaler_y):
    """
    Get sample, plot Gaussian mixture components + overall mixture density.
    Return vals in og scale

    Args:
        model: Trained MDN model.
        sample_x: Input sample tensor of shape (1, features).
        sample_y: True target value (in original scale) for the sample.
        scaler_y: The scaler used to standardize targets.
    """
    model.eval()
    with torch.no_grad():
        pi, mu, sigma = model(sample_x)
        # remove batch dimension.
        pi = pi.cpu().numpy()[0]               # shape: (num_mixtures,)
        mu = mu.cpu().numpy()[0].squeeze(-1)     # shape: (num_mixtures,)
        sigma = sigma.cpu().numpy()[0].squeeze(-1)

    # std scale to og scale
    y_mean = scaler_y.mean_[0]
    y_std = scaler_y.scale_[0]
    mu_orig = mu * y_std + y_mean
    sigma_orig = sigma * y_std

    # Range
    x_min = sample_y - 4 * np.max(sigma_orig)
    x_max = sample_y + 4 * np.max(sigma_orig)
    x_vals = np.linspace(x_min, x_max, 1000)

    total_density = np.zeros_like(x_vals)
    plt.figure(figsize=(10, 6))
    for i in range(len(pi)):
        # Density for each Gaussian component.
        density = pi[i] * (1/(np.sqrt(2*np.pi)*sigma_orig[i])) * \
                  np.exp(-0.5 * ((x_vals - mu_orig[i]) / sigma_orig[i])**2)
        total_density += density
        plt.plot(x_vals, density, linestyle='--', label=f'Component {i+1}')

    # Plot overall mixture density.
    plt.plot(x_vals, total_density, label="Mixture Density", color='black', linewidth=2)
    plt.axvline(x=sample_y, color='red', linestyle='-', linewidth=2, label="True Value")
    plt.xlabel("Target Value", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.title("Mixture Density Components for a Sample", fontsize=16)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

import random
idx = random.randint(0, X_test_tensor.shape[0]-1)
sample_x = X_test_tensor[idx:idx+1]
# Convert to og scale.
sample_y = scaler_y.inverse_transform(y_test_tensor[idx:idx+1].cpu().numpy())[0, 0]
plot_mixture_density(model, sample_x, sample_y, scaler_y)
