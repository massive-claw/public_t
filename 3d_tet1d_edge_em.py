import gmsh
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
import plotly.graph_objects as go
import sys

def run_corrected_iso_fem():
    # ---------------------------------------------------------
    # 1. Mesh generation via Gmsh
    # ---------------------------------------------------------
    MESH_SIZE = 0.15
    print(f"Initializing Gmsh (Size={MESH_SIZE})...")
    
    gmsh.initialize()
    gmsh.model.add("UnitCube")
    gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeMin", MESH_SIZE)
    gmsh.option.setNumber("Mesh.MeshSizeMax", MESH_SIZE)
    gmsh.model.mesh.generate(3)
    
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    nodes = np.array(node_coords).reshape(-1, 3)
    tag2idx = {t: i for i, t in enumerate(node_tags)}
    
    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=3)
    tet_node_tags = None
    for i, t in enumerate(elem_types):
        if t == 4:
            tet_node_tags = elem_node_tags[i]
            break
            
    if tet_node_tags is None:
        print("Error: No tetrahedra found.")
        gmsh.finalize()
        return

    elements_raw = np.array(tet_node_tags, dtype=int).reshape(-1, 4)
    elems = np.vectorize(tag2idx.get)(elements_raw)
    
    gmsh.finalize()
    
    num_nodes = len(nodes)
    num_elems = len(elems)
    print(f"Mesh: {num_nodes} nodes, {num_elems} elements")

    # ---------------------------------------------------------
    # 2. Building Connectivity and Signs
    # ---------------------------------------------------------
    print("Building Connectivity and Signs...")
    
    local_edge_pairs = np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])
    
    edge_dict = {}     
    id_to_edge = []    
    next_edge_id = 0
    
    elem_edge_conn = np.zeros((num_elems, 6), dtype=int)
    elem_edge_signs = np.zeros((num_elems, 6), dtype=int)
    
    for e in range(num_elems):
        e_nodes = elems[e]
        for i in range(6):
            u, v = e_nodes[local_edge_pairs[i]]
            
            key = (min(u, v), max(u, v))
            
            if key not in edge_dict:
                edge_dict[key] = next_edge_id
                id_to_edge.append(key)
                next_edge_id += 1
            
            elem_edge_conn[e, i] = edge_dict[key]
            
            # Define signs (orientation)
            elem_edge_signs[e, i] = 1 if u < v else -1

    num_edges = next_edge_id
    print(f"Total Unique Edges: {num_edges}")

    # ---------------------------------------------------------
    # 3. Boundary Conditions (PEC)
    # ---------------------------------------------------------
    boundary_edges = set()
    tol = 1e-6
    for (n1, n2), eid in edge_dict.items():
        p1, p2 = nodes[n1], nodes[n2]
        is_boundary = False
        for d in range(3):
            if (abs(p1[d]) < tol and abs(p2[d]) < tol) or \
               (abs(p1[d]-1) < tol and abs(p2[d]-1) < tol):
                is_boundary = True
                break
        if is_boundary:
            boundary_edges.add(eid)
            
    free_dofs = np.array([i for i in range(num_edges) if i not in boundary_edges])
    print(f"Free DOFs: {len(free_dofs)}")

    # ---------------------------------------------------------
    # 4. Isoparametric Gradient Calculation (Corrected)
    # ---------------------------------------------------------
    print("Computing Gradients via Isoparametric Mapping (Corrected)...")
    
    # Gradient of the reference element (row vector format)
    # N0=1-x-y-z, N1=x, N2=y, N3=z
    grad_ref = np.array([
        [-1, -1, -1],
        [ 1,  0,  0],
        [ 0,  1,  0],
        [ 0,  0,  1]
    ]) # (4, 3)

    grads_physical = np.zeros((num_elems, 4, 3))
    volumes = np.zeros(num_elems)
    # Save the affine inverse matrix for barycentric coordinate calculation for visualization
    inv_mats = np.zeros((num_elems, 4, 4))
    
    for e in range(num_elems):
        X = nodes[elems[e]] # (4, 3)
        
        # --- Jacobian Calculation ---
        # J_code = grad_ref.T @ X
        # The matrix obtained here is the *transpose* of the mathematical Jacobian matrix J = d(x)/d(xi).
        # J_code[i, j] = dx_j / dxi_i
        J_code = grad_ref.T @ X
        
        detJ = np.linalg.det(J_code)
        # Gmsh uses even permutations so detJ should be positive, but taking absolute value just in case
        volumes[e] = abs(detJ) / 6.0
        
        # Inverse matrix
        J_inv_code = np.linalg.inv(J_code)
        
        # --- [Correction Point] ---
        # Physical Gradient = Reference Gradient @ (Mathematical J)^-1
        # Since Mathematical J = J_code.T, then (Mathematical J)^-1 = (J_code.T)^-1 = (J_code^-1).T
        # Therefore, we need to multiply by the transpose of J_inv_code.
        grads_physical[e] = grad_ref @ J_inv_code.T
        
        # Use a robust 4x4 inverse matrix for barycentric coordinate calculation for visualization
        # (keeping this here for convenience)
        mat_aff = np.c_[np.ones(4), X]
        inv_mats[e] = np.linalg.inv(mat_aff)

    # ---------------------------------------------------------
    # 5. Matrix Assembly
    # ---------------------------------------------------------
    print("Assembling Global Matrices...")
    
    rows_S, cols_S, data_S = [], [], []
    rows_M, cols_M, data_M = [], [], []
    
    base_int_table = np.full((4, 4), 1/20.0)
    np.fill_diagonal(base_int_table, 1/10.0)
    
    print_interval = max(1, num_elems // 10)
    
    for e in range(num_elems):
        if e % print_interval == 0:
            sys.stdout.write(f"\r{e}/{num_elems}")
            sys.stdout.flush()
            
        grads = grads_physical[e] # (4, 3)
        vol = volumes[e]
        e_edges = elem_edge_conn[e]
        e_signs = elem_edge_signs[e]
        
        for i in range(6):
            na, nb = local_edge_pairs[i]
            edge_i = e_edges[i]
            sign_i = e_signs[i]
            
            # Curl Ni = 2 * (grad a x grad b)
            curl_Ni = 2.0 * np.cross(grads[na], grads[nb])
            
            for j in range(i, 6): # Upper triangle
                nc, nd = local_edge_pairs[j]
                edge_j = e_edges[j]
                sign_j = e_signs[j]
                
                curl_Nj = 2.0 * np.cross(grads[nc], grads[nd])
                
                # S Matrix
                val_S = np.dot(curl_Ni, curl_Nj) * vol * sign_i * sign_j
                rows_S.append(edge_i); cols_S.append(edge_j); data_S.append(val_S)
                if i != j:
                    rows_S.append(edge_j); cols_S.append(edge_i); data_S.append(val_S)
                
                # M Matrix
                dot_bd = np.dot(grads[nb], grads[nd])
                dot_bc = np.dot(grads[nb], grads[nc])
                dot_ad = np.dot(grads[na], grads[nd])
                dot_ac = np.dot(grads[na], grads[nc])
                
                term_M = dot_bd * base_int_table[na, nc] \
                       - dot_bc * base_int_table[na, nd] \
                       - dot_ad * base_int_table[nb, nc] \
                       + dot_ac * base_int_table[nb, nd]
                
                val_M = term_M * vol * sign_i * sign_j
                rows_M.append(edge_i); cols_M.append(edge_j); data_M.append(val_M)
                if i != j:
                    rows_M.append(edge_j); cols_M.append(edge_i); data_M.append(val_M)

    S_mat = sp.coo_matrix((data_S, (rows_S, cols_S)), shape=(num_edges, num_edges)).tocsr()
    M_mat = sp.coo_matrix((data_M, (rows_M, cols_M)), shape=(num_edges, num_edges)).tocsr()
    print("\nMatrices assembled.")

    # ---------------------------------------------------------
    # 6. Eigenvalue Analysis
    # ---------------------------------------------------------
    print("Solving Eigenvalue Problem...")
    
    S_red = S_mat[free_dofs, :][:, free_dofs]
    M_red = M_mat[free_dofs, :][:, free_dofs]
    
    try:
        print("Using Sparse Solver with shift-invert (sigma=15.0)...")
        vals, vecs = sp.linalg.eigsh(S_red, M=M_red, k=20, sigma=15.0, which='LM')
    except Exception as e:
        print(f"Sparse solver failed: {e}. Falling back to Dense solver.")
        vals, vecs = scipy.linalg.eigh(S_red.toarray(), M_red.toarray())
    
    idx = np.argsort(vals)
    vals = vals[idx]
    vecs = vecs[:, idx]

    print("\n--- Results (Corrected Isoparametric FEM) ---")
    print(f"Theoretical TE101: {2 * np.pi**2:.4f}")
    
    physical_modes = []
    for i, val in enumerate(vals):
        if val > 15.0:
            full_vec = np.zeros(num_edges)
            full_vec[free_dofs] = vecs[:, i]
            physical_modes.append({"id": i+1, "k2": val, "vec": full_vec})
            print(f"Mode (shifted index {i}): k² = {val:.4f} [Physical]")
        else:
            print(f"Mode (shifted index {i}): k² = {val:.4f} [Spurious/Gradient]")
        
        if len(physical_modes) >= 6: break

    # ---------------------------------------------------------
    # 7. Visualization
    # ---------------------------------------------------------
    print("Generating Visualization...")
    
    N_SAMPLE = 10
    g = np.linspace(0.1, 0.9, N_SAMPLE)
    gx, gy, gz = np.meshgrid(g, g, g)
    grid_pts = np.vstack([gx.ravel(), gy.ravel(), gz.ravel()]).T
    
    vis_data = []
    
    elem_bboxes = []
    for e in range(num_elems):
        X = nodes[elems[e]]
        elem_bboxes.append((np.min(X, axis=0), np.max(X, axis=0)))

    for pt in grid_pts:
        pt_h = np.array([1, pt[0], pt[1], pt[2]])
        for e in range(num_elems):
            min_b, max_b = elem_bboxes[e]
            if np.any(pt < min_b) or np.any(pt > max_b): continue
            
            lam = pt_h @ inv_mats[e]
            if np.all(lam >= -1e-6):
                vis_data.append({"e": e, "lam": lam, "pos": pt})
                break
                
    vis_pos = np.array([d["pos"] for d in vis_data])
    
    fig = go.Figure()
    
    ex, ey, ez = [], [], []
    for (n1, n2) in id_to_edge:
        p1, p2 = nodes[n1], nodes[n2]
        ex.extend([p1[0], p2[0], None])
        ey.extend([p1[1], p2[1], None])
        ez.extend([p1[2], p2[2], None])

    fig.add_trace(go.Scatter3d(
        x=ex, y=ey, z=ez, mode='lines',
        line=dict(color='rgba(0,0,0,0.05)', width=1),
        name='Mesh', hoverinfo='none'
    ))
    
    steps = []
    for i, mode in enumerate(physical_modes):
        vec_glob = mode["vec"]
        vectors = []
        for d in vis_data:
            e = d["e"]
            lam = d["lam"]
            grads = grads_physical[e]
            edges = elem_edge_conn[e]
            signs = elem_edge_signs[e]
            
            field = np.zeros(3)
            for k in range(6):
                na, nb = local_edge_pairs[k]
                gid = edges[k]
                sign = signs[k]
                coef = vec_glob[gid]
                
                # Basis reconstruction
                basis = lam[na] * grads[nb] - lam[nb] * grads[na]
                field += coef * sign * basis
            vectors.append(field)
            
        vectors = np.array(vectors)
        norms = np.linalg.norm(vectors, axis=1)
        if len(norms) > 0: vectors /= np.max(norms)
        
        visible = (i==0)
        trace = go.Cone(
            x=vis_pos[:,0], y=vis_pos[:,1], z=vis_pos[:,2],
            u=vectors[:,0], v=vectors[:,1], w=vectors[:,2],
            sizemode="scaled", sizeref=4.0, colorscale='Jet',
            visible=visible, name=f"Mode {mode['id']}", showscale=True,
            colorbar=dict(title='Field')
        )
        fig.add_trace(trace)
        
        vis_list = [True] + [False] * len(physical_modes)
        vis_list[i+1] = True
        steps.append(dict(method="update", args=[{"visible": vis_list}, {"title": f"k²={mode['k2']:.4f}"}], label=f"{i+1}"))
    
    title_txt = f"Iso FEM: k²={physical_modes[0]['k2']:.4f}" if physical_modes else "No Modes"
    fig.update_layout(
        title=title_txt,
        scene=dict(aspectmode='data', camera=dict(eye=dict(x=1.3, y=1.3, z=1.3)),
                   xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
        updatemenus=[dict(active=0, buttons=steps, x=0, y=1)]
    )
    
    fig.write_html("corrected_iso_fem_result.html")
    print("\n[Info] Saved corrected_iso_fem_result.html")
    fig.show()

if __name__ == "__main__":
    run_corrected_iso_fem()
