import scanpy as sc
import tensornetwork as tn
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from torch.utils.data import TensorDataset, DataLoader

#need to install: install scanpy tensornetwork torch numpy scanpy leidenalg

# --- CONFIGURATION FOR RTX 3050 ---
# We limit the chain length to 40 nodes to ensure fast training on your laptop.
# If you move to the V100 later, you can increase PCA_COMPONENTS to 100 or 200.
CONFIG = {
    'pca_components': 40,      # Number of "Genes" in our MPS chain (Nodes)
    'bond_dim': 10,            # The "Memory" (Chi) - keeps VRAM usage low
    'batch_size': 32,          # Process 32 cells at a time
    'epochs': 5,
    'learning_rate': 0.01,
    'data_path': './data/10x_dataset' # POINT THIS TO YOUR 10x FOLDER
}

# Set TensorNetwork to use PyTorch
tn.set_default_backend("pytorch")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# ==========================================
# 1. PREPROCESSING (The Biology Part)
# ==========================================
def load_and_process_data(data_path, n_components):
    print("--- 1. Loading 10x Genomics Data ---")
    # Reads standard 10x output (matrix.mtx, features.tsv, barcodes.tsv)
    adata = sc.read_10x_mtx(data_path, var_names='gene_symbols', cache=True)
    
    print(f"Original shape: {adata.shape}")
    
    # Standard scRNA-seq normalization
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    print("--- 2. Reducing Dimensionality (PCA) ---")
    # We MUST reduce dimensions. An MPS of length 30,000 is impossible on a 3050.
    # We take the top 'n_components' features which contain the most info.
    sc.pp.pca(adata, n_comps=n_components)
    
    # Extract the PCA data (The input for our MPS)
    X_pca = adata.obsm['X_pca']
    
    # Normalize data to range [0, 1] for the Quantum Feature Map
    # This prepares it to be turned into angles.
    X_min = X_pca.min()
    X_max = X_pca.max()
    X_norm = (X_pca - X_min) / (X_max - X_min)
    
    return torch.tensor(X_norm, dtype=torch.float32)

# ==========================================
# 2. THE MPS MODEL (The Physics Part)
# ==========================================
class MPSModel(nn.Module):
    def __init__(self, n_features, bond_dim):
        super().__init__()
        self.n_features = n_features
        self.bond_dim = bond_dim
        
        # We create the MPS cores as PyTorch Parameters so we can train them.
        # Core Shapes:
        # Left: (1, 2, D)  -> Dummy left leg, Physical leg (qubit), Right Bond
        # Middle: (D, 2, D) -> Left Bond, Physical leg, Right Bond
        # Right: (D, 2, 1) -> Left Bond, Physical leg, Dummy right leg
        
        # Initialize with random small noise
        self.cores = nn.ParameterList()
        
        # Node 1
        self.cores.append(nn.Parameter(torch.randn(1, 2, bond_dim) * 0.01))
        
        # Middle Nodes
        for _ in range(n_features - 2):
            self.cores.append(nn.Parameter(torch.randn(bond_dim, 2, bond_dim) * 0.01))
            
        # Last Node
        self.cores.append(nn.Parameter(torch.randn(bond_dim, 2, 1) * 0.01))

    def feature_map(self, x):
        """
        Transforms scalar data into a Qubit state vector.
        Input x: [0, 1]
        Output: [cos(theta), sin(theta)]
        """
        # Map 0..1 to 0..PI/2
        theta = x * (np.pi / 2)
        # Stack cos and sin to get a 2-dimensional vector (The "Qubit")
        return torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)

    def forward(self, x_batch):
        """
        Calculates the contraction <MPS | Data>.
        This is effectively the 'probability' that the data belongs to the trained state.
        """
        batch_scores = []
        
        # Map the whole batch to quantum states: Shape (Batch, 2, Features)
        data_qubits = self.feature_map(x_batch)
        
        # NOTE: TensorNetwork is tricky with batching. 
        # For clarity and safety on the 3050, we loop through the batch.
        # (In a production HPC version, we would vectorize this manually)
        for i in range(x_batch.shape[0]):
            
            # 1. Create Nodes for the MPS Cores
            nodes = [tn.Node(core, backend="pytorch") for core in self.cores]
            
            # 2. Connect the cores (The Bond Dimensions)
            # Connect Right of Node[j] to Left of Node[j+1]
            for j in range(self.n_features - 1):
                nodes[j][2] ^ nodes[j+1][0] # '^' connects edges in TensorNetwork
            
            # 3. Connect Data to the MPS (The Physical Dimensions)
            # We "project" the data onto the MPS.
            sample_qubits = data_qubits[i] # Shape (2, Features)
            
            for j in range(self.n_features):
                # Create a node for the data point (Single qubit state)
                data_node = tn.Node(sample_qubits[:, j], backend="pytorch")
                
                # Connect Data Node to MPS Node's physical leg (Index 1)
                # MPS Node legs are: (Left, Physical, Right)
                nodes[j][1] ^ data_node[0]
            
            # 4. Contract the entire network
            # tensornetwork finds the optimal path to multiply these matrices
            result = tn.contractors.auto(nodes + [tn.Node(sample_qubits[:, j], backend="pytorch") for j in range(self.n_features)])
            
            # The result is a scalar (The amplitude)
            batch_scores.append(result.tensor)
            
        return torch.stack(batch_scores).squeeze()

# ==========================================
# 3. TRAINING LOOP
# ==========================================
def main():
    # Load Data
    try:
        data = load_and_process_data(CONFIG['data_path'], CONFIG['pca_components'])
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please ensure 'data_path' points to a folder containing matrix.mtx.gz")
        return

    # Create DataLoader
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    
    # Initialize Model
    model = MPSModel(n_features=CONFIG['pca_components'], bond_dim=CONFIG['bond_dim'])
    model.to(device)
    
    # Optimizer (Adam works well for TNs)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    print("\n--- Starting Training ---")
    print(f"Model Structure: MPS Chain of length {CONFIG['pca_components']}")
    print(f"Bond Dimension: {CONFIG['bond_dim']}")
    
    for epoch in range(CONFIG['epochs']):
        total_loss = 0
        start_time = time.time()
        
        for batch_idx, (batch_data,) in enumerate(loader):
            batch_data = batch_data.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass: Get the overlap (amplitude)
            # A high overlap means the MPS recognizes the data as "Healthy"
            overlaps = model(batch_data)
            
            # Loss Function: Negative Log Likelihood
            # We want to maximize Overlap^2 (Probability).
            # So we minimize -log(Overlap^2)
            # We add 1e-10 to avoid log(0)
            probability = overlaps ** 2
            loss = -torch.mean(torch.log(probability + 1e-10))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1} [{batch_idx}/{len(loader)}] Loss: {loss.item():.4f}")
        
        print(f"Epoch {epoch+1} Complete. Avg Loss: {total_loss / len(loader):.4f}. Time: {time.time()-start_time:.1f}s")
    
    print("\n--- Training Complete ---")
    print("MPS Model is now trained to recognize the 'Healthy' baseline.")
    
    # --- SAVE THE MODEL ---
    torch.save(model.state_dict(), "mps_biodefence_model.pt")
    print("Model saved to mps_biodefence_model.pt")

if __name__ == "__main__":
    main()