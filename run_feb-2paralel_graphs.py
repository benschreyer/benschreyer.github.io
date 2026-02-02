#RUN RESULT
"""
C:\Users\schre\OneDrive\Desktop>py paralel_graphs.py
Loaded 274668 graphs from graph9.g6.txt.
Deploying on 16 cores.

[Phase 1] Parallel Matrix Generation...
Generating: 100%|████████████████████████████████████████████████████████████| 274668/274668 [01:23<00:00, 3303.13it/s]
Data Shape: (274668, 8, 8)
Comparing: 100%|█████████████████████████████████████████████████████████████| 274668/274668 [1:49:41<00:00, 41.74it/s]
------------------------------
Total Matches (< 0.1): 0
Traceback (most recent call last):
  File "C:\Users\schre\OneDrive\Desktop\paralel_graphs.py", line 144, in <module>
    print(f"Minimum Distance: {min_dist}")
                               ^^^^^^^^
NameError: name 'min_dist' is not defined

C:\Users\schre\OneDrive\Desktop>"""
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from networkx.generators.atlas import graph_atlas_g
from tqdm import tqdm
import multiprocessing
from functools import partial

# --- Graph Loading & Matrix Functions ---

def get_graphs_from_g6(file):
    graphs = []
    with open(file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                G = nx.from_graph6_bytes(line.encode('ascii'))
                graphs.append(G)
    print(f"Loaded {len(graphs)} graphs from {file}.")
    return graphs

def graph_to_matrix(G, zd=False):
    # Force integer type immediately
    arr = nx.to_numpy_array(G, nodelist=sorted(G.nodes()), dtype=int)
    final_arr = arr.copy()
    final_arr[arr == 0] = -1
    final_arr[arr == 1] = 1
    if zd:
        final_arr[np.tril_indices_from(final_arr, -1)] = 0
    return final_arr

def form_prop_matrix(G, zd=False):
    G_matrix = graph_to_matrix(G, zd)
    n_nodes = G_matrix.shape[0]
    ones_vector = np.ones((n_nodes, 1), dtype=int)
    vecs = [ones_vector]
    
    for i in range(n_nodes):
        vecs.append(np.matmul(G_matrix, vecs[-1]))
    
    # Calculate dimensions
    matrix_size = len(vecs) - 2 
    prop_matrix = np.zeros((matrix_size, matrix_size), dtype=int)
    
    # Pre-transpose for speed
    vecs_t = [v.T for v in vecs]

    for i in range(1, len(vecs) - 1):
        for j in range(1, len(vecs) - 1):
            # item() is faster than keeping 1x1 array
            prop_matrix[i-1][j-1] = np.dot(vecs_t[i], vecs[j]).item()
            
    return prop_matrix

# --- Helper wrapper for Parallel Phase 1 ---
def compute_signature(G):
    """Worker function to compute both matrices for a single graph."""
    return (form_prop_matrix(G, zd=True))

# --- Global storage for Parallel Phase 2 ---
# We use global variables for the big arrays so child processes 
# can read them without expensive pickling/copying.
global GLOBAL_MATS_STD, GLOBAL_MATS_ZD
GLOBAL_MATS_STD = None
GLOBAL_MATS_ZD = None

# --- 1. Revised Worker Function ---
def compare_row_vectorized(i):
    # Access the globals initialized by the pool
    global GLOBAL_MATS_ZD
    
    mat_i_zd = GLOBAL_MATS_ZD[i]
    targets_zd = GLOBAL_MATS_ZD[i+1:]
    
    if len(targets_zd) == 0:
        return []

    # Vectorized subtraction and norm calculation
    diff_zd = targets_zd - mat_i_zd
    
    # Calculate Frobenius norm for each matrix in the stack
    # dist = sqrt(sum of squares of elements)
    norms_zd = np.linalg.norm(diff_zd, axis=(1, 2))
    
    # Find indices where distance is effectively zero
    match_indices = np.where(norms_zd < 0.1)[0]
    
    results = []
    for match_idx in match_indices:
        real_j = i + 1 + match_idx
        results.append((i, real_j, norms_zd[match_idx]))
        
    return results

# --- 2. Initializer for Windows Compatibility ---
def init_worker(shared_zd):
    global GLOBAL_MATS_ZD
    GLOBAL_MATS_ZD = shared_zd
# --- Main Execution ---

if __name__ == "__main__":
    # 1. Setup
    # Ensure this matches your file path
    FILE_PATH = "graph9.g6.txt" 
    try:
        graphs = get_graphs_from_g6(FILE_PATH)
    except FileNotFoundError:
        print(f"File {FILE_PATH} not found. Using Atlas for testing.")
        graphs = [g for g in graph_atlas_g() if len(g.nodes()) == 6]

    # Determine Cores
    num_cores = multiprocessing.cpu_count()
    print(f"Deploying on {num_cores} cores.")

    # --- STEP 2: PARALLEL MATRIX GENERATION ---
    print(f"\n[Phase 1] Parallel Matrix Generation...")
    signatures = []
    with multiprocessing.Pool(processes=num_cores) as pool:
        for res in tqdm(pool.imap(compute_signature, graphs), total=len(graphs), desc="Generating"):
            signatures.append(res)
            
    # Corrected stacking: signatures is a list of matrices
    all_mats_zd = np.array(signatures)
    
    print(f"Data Shape: {all_mats_zd.shape}")

    # [Phase 2] Comparison with Initializer
    matches_found = []
    # Pass the array to the initializer so every worker has it in global scope
    with multiprocessing.Pool(processes=num_cores, 
                             initializer=init_worker, 
                             initargs=(all_mats_zd,)) as pool:
        
        work_indices = range(len(graphs))
        
        for batch_results in tqdm(pool.imap_unordered(compare_row_vectorized, work_indices), 
                                  total=len(graphs), 
                                  desc="Comparing"):
            if batch_results:
                matches_found.extend(batch_results)
    # --- STEP 4: VISUALIZATION (Main Thread) ---
    print("-" * 30)
    print(f"Total Matches (< 0.1): {len(matches_found)}")
    print(f"Minimum Distance: {min_dist}")
    
    # Sort matches by distance (closest first)
    matches_found.sort(key=lambda x: x[2])
    
    # Visualize top 3 matches
    for idx, (i, j, dist) in enumerate(matches_found):
        if idx >= 3: 
            print(f"... and {len(matches_found) - 3} more matches.")
            break
            
        print(f"Displaying Match: Graph {i} vs Graph {j} (Dist: {dist:.6f})")
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle(f"Match Found (Dist: {dist:.6f})", fontsize=14)
        
        nx.draw(graphs[i], ax=axes[0], with_labels=True, node_color='lightblue', edge_color='gray')
        axes[0].set_title(f"Graph {i}")
        
        nx.draw(graphs[j], ax=axes[1], with_labels=True, node_color='lightgreen', edge_color='gray')
        axes[1].set_title(f"Graph {j}")
        
        plt.show()