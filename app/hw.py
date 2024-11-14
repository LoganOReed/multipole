import numpy as np

X = np.array([0.02, 0.04, 0.06, 0.13, 0.17, 0.23, 0.35, 0.45, 0.58, 0.65, 0.68, 0.73, 0.82, 0.94, 0.96])
q = np.array([21, 30, 47, 23, 84, 19, 23, 17, 23, 44, 31, 92, 43, 18, 90]) * 0.001
y = 0.59

def G(xj, yi):
    """Greens function"""
    return 1 / np.abs(xj - yi)**2

# (a)
u_exact = np.sum(G(X, y) * q)
print("Exact value of u:", u_exact)

J = 3  
p_values = [1, 2]

# (b)
for p in p_values:
    # Step i
    x_min, x_max = 0, 1
    level_width = (x_max - x_min) / (2**J)
    cell_index = int((y - x_min) // level_width)
    
    # Step ii
    near_field_cells = [cell_index - 1, cell_index, cell_index + 1]
    
    # Step iii
    far_field_weights = []
    for cell in range(2**J):
        if cell not in near_field_cells:
            x_star = x_min + (cell + 0.5) * level_width # center
            indices = (X >= x_min + cell * level_width) & (X < x_min + (cell + 1) * level_width)
            weights = []
            for m in range(p + 1):
                weight = np.sum(q[indices] * (X[indices] - x_star)**m)
                weights.append(weight)
            far_field_weights.append((x_star, weights))
    
    # Step iv
    far_field_contribution = 0
    for x_star, weights in far_field_weights:
        S = [1 / np.abs(x_star - y)**(m + 1) for m in range(p + 1)]
        far_field_contribution += sum(w * s for w, s in zip(weights, S))
    
    # Step v
    near_field_indices = np.concatenate([np.where((X >= x_min + cell * level_width) & (X < x_min + (cell + 1) * level_width))[0] for cell in near_field_cells])
    near_field_contribution = np.sum(G(X[near_field_indices], y) * q[near_field_indices])
    u_FMM = near_field_contribution + far_field_contribution
    print(f"Numerical value of u with p={p}:", u_FMM)

    # (c)
    relative_error = np.abs(u_FMM - u_exact) / u_exact
    print(f"Relative error with p={p}:", relative_error)
