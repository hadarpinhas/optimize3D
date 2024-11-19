import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from skimage import measure
import trimesh

def save_voxel_grid_as_stl(voxel_grid, filename='generated_shape.stl'):
    """
    Convert the voxel grid to a mesh and save as an STL file.
    """
    # Use the marching cubes algorithm to extract the surface mesh
    verts, faces, normals, values = measure.marching_cubes(
        voxel_grid, level=0.5, spacing=(1.0, 1.0, 1.0)
    )
    # Create a mesh
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    # Export the mesh to an STL file
    mesh.export(filename)
    print(f"Mesh saved as '{filename}'")

def create_box_with_holes(width, height, depth, hole_diameter, grid_size=32):
    """
    Create a voxel grid representing a box with holes based on input dimensions.
    """
    # Initialize empty grid
    voxel_grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    
    # Compute the scaling factor
    max_dim = max(width, height, depth)
    scale = (grid_size - 2) / max_dim  # Subtract 2 to leave a margin
    
    # Scale dimensions to grid size
    w = int(width * scale)
    h = int(height * scale)
    d = int(depth * scale)
    hole_r = int((hole_diameter * scale) / 2)
    
    # Compute the starting indices
    x_start = (grid_size - w) // 2
    y_start = (grid_size - h) // 2
    z_start = (grid_size - d) // 2
    
    # Create box
    voxel_grid[
        x_start:x_start + w,
        y_start:y_start + h,
        z_start:z_start + d
    ] = 1.0
    
    # Create holes in the center
    xx, yy, zz = np.meshgrid(
        np.arange(grid_size),
        np.arange(grid_size),
        np.arange(grid_size),
        indexing='ij'
    )
    center = grid_size // 2
    distance = np.sqrt(
        (xx - center)**2 +
        (yy - center)**2 +
        (zz - center)**2
    )
    voxel_grid[distance < hole_r] = 0.0
    
    return voxel_grid

def visualize_voxel_grid(voxel_grid, threshold=0.5):
    """
    Visualize the voxel grid using matplotlib's 3D plotting.
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    # Get the coordinates of the voxels
    filled = voxel_grid > threshold
    x, y, z = filled.nonzero()
    ax.scatter(x, y, z, zdir='z', c='red', marker='s')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def generate_dataset(num_samples=1000, grid_size=32):
    """
    Generate a synthetic dataset of voxel grids and corresponding parameters.
    """
    data = []
    params = []
    for _ in range(num_samples):
        # Random parameters within a specified range
        width = np.random.uniform(5, 15)
        height = np.random.uniform(5, 15)
        depth = np.random.uniform(5, 15)
        hole_diameter = np.random.uniform(0.5, min(width, height, depth) / 2)
        
        voxel_grid = create_box_with_holes(width, height, depth, hole_diameter, grid_size)
        data.append(voxel_grid)
        params.append([width, height, depth, hole_diameter])
    
    data = np.array(data)
    params = np.array(params)
    return data, params

class ConditionalVoxelVAE(nn.Module):
    def __init__(self, latent_dim=128, param_dim=4, input_shape=(32, 32, 32)):
        super(ConditionalVoxelVAE, self).__init__()
        self.latent_dim = latent_dim
        self.param_dim = param_dim
        self.input_shape = input_shape

        # Encoder
        self.enc_conv1 = nn.Conv3d(1, 32, kernel_size=4, stride=2, padding=1)
        self.enc_conv2 = nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1)
        self.enc_conv3 = nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1)
        self.enc_fc = nn.Linear(128 * 4 * 4 * 4 + param_dim, latent_dim * 2)  # For mu and logvar

        # Decoder
        self.dec_fc = nn.Linear(latent_dim + param_dim, 128 * 4 * 4 * 4)
        self.dec_conv1 = nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1)
        self.dec_conv2 = nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1)
        self.dec_conv3 = nn.ConvTranspose3d(32, 1, kernel_size=4, stride=2, padding=1)

    def encode(self, x, params):
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = F.relu(self.enc_conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.cat([x, params], dim=1)
        h = self.enc_fc(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, params):
        z = torch.cat([z, params], dim=1)
        x = F.relu(self.dec_fc(z))
        x = x.view(-1, 128, 4, 4, 4)
        x = F.relu(self.dec_conv1(x))
        x = F.relu(self.dec_conv2(x))
        x = torch.sigmoid(self.dec_conv3(x))
        return x

    def forward(self, x, params):
        mu, logvar = self.encode(x, params)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, params)
        return x_recon, mu, logvar

def loss_function(recon_x, x, mu, logvar):
    """
    Compute the VAE loss function as the sum of reconstruction loss and KL divergence.
    """
    # Reconstruction loss (binary cross-entropy)
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    # KL divergence loss
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld_loss

def train_vae(model, dataloader, num_epochs=20, learning_rate=1e-3, device=torch.device('cpu')):
    """
    Train the VAE model with the provided data loader.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(num_epochs):
        train_loss = 0
        for batch_idx, (data_batch, params_batch) in enumerate(dataloader):
            optimizer.zero_grad()
            # Move data to the device (GPU or CPU)
            data_batch = data_batch.to(device)
            params_batch = params_batch.to(device)
            recon_batch, mu, logvar = model(data_batch, params_batch)
            loss = loss_function(recon_batch, data_batch, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        avg_loss = train_loss / len(dataloader.dataset)
        print(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}')

if __name__ == "__main__":
    # Step 1: Generate the dataset
    data, params = generate_dataset(num_samples=500, grid_size=32)

    # Step 2: Convert data to tensors
    tensor_x = torch.Tensor(data).unsqueeze(1)  # Shape: [N, 1, 32, 32, 32]
    tensor_params = torch.Tensor(params)  # Shape: [N, 4]

    # Step 3: Normalize parameters
    params_mean = tensor_params.mean(0, keepdim=True)
    params_std = tensor_params.std(0, unbiased=False, keepdim=True)
    tensor_params = (tensor_params - params_mean) / params_std

    # Step 4: Create dataset and dataloader
    dataset = TensorDataset(tensor_x, tensor_params)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Step 5: Check for GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Step 6: Initialize the model and move it to the device
    model = ConditionalVoxelVAE(latent_dim=128, param_dim=4)
    model.to(device)

    # Step 7: Train the model
    train_vae(model, dataloader, num_epochs=20, learning_rate=1e-3, device=device)

    # Step 8: Generate a new shape based on input parameters
    model.eval()

    # Example input parameters (normalized)
    input_params = torch.Tensor([[12, 6, 2, 1]])
    input_params = (input_params - params_mean) / params_std
    input_params = input_params.to(device)

    # Generate latent vector and decode
    with torch.no_grad():
        z = torch.randn(1, model.latent_dim).to(device)
        generated_shape = model.decode(z, input_params).cpu().squeeze().numpy()

    # Step 9: Visualize the generated shape
    visualize_voxel_grid(generated_shape)

    # Step 10: Save the voxel grid as a NumPy array
    np.save('generated_shape.npy', generated_shape)
    print("Voxel grid saved as 'generated_shape.npy'")

    # Save the generated shape as an STL file
    save_voxel_grid_as_stl(generated_shape)

    
