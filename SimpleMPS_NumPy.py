import numpy as np

# --- CONFIGURATION ---
num_genes = 5        # Number of features (genes)
physical_dim = 2     # Gene states: 0 (Off) or 1 (On)
bond_dim = 4         # The "Memory" (How much info passes between genes)

# --- 1. INITIALIZATION (Building the Train) ---
# An MPS is just a list of tensors (arrays)
mps_tensors = []

# First Tensor (Gene 1): Left leg is '1' (start of chain)
# Shape: (Left_Bond, Physical, Right_Bond)
mps_tensors.append(np.random.rand(1, physical_dim, bond_dim))

# Middle Tensors (Gene 2 to 4)
for i in range(num_genes - 2):
    mps_tensors.append(np.random.rand(bond_dim, physical_dim, bond_dim))

# Last Tensor (Gene 5): Right leg is '1' (end of chain)
mps_tensors.append(np.random.rand(bond_dim, physical_dim, 1))

print("MPS Created!")
print(f"Gene 1 Shape: {mps_tensors[0].shape} (Start)")
print(f"Gene 3 Shape: {mps_tensors[2].shape} (Middle)")

# --- 2. CONTRACTION (The "Calculation") ---
# To check if this state is valid, we calculate its Norm <Psi|Psi>
# This means we zip the chain up with itself.

# Let's verify the "Bond" sizes match
left_neighbor = mps_tensors[0]  # Shape (1, 2, 4)
right_neighbor = mps_tensors[1] # Shape (4, 2, 4)

# The '4' on the right of Gene 1 MUST match the '4' on the left of Gene 2
print(f"Bond Check: {left_neighbor.shape[2]} connects to {right_neighbor.shape[0]}")

# We use Einstein Summation (einsum) for the math.
# 'lpr' = (Left, Physical, Right)
# We contract the Right of A (r) with the Left of B (l)
contraction_result = np.einsum('lpr,rPD->lpPD', left_neighbor, right_neighbor)
print(f"Contracted Shape: {contraction_result.shape}")