import gmsh
import numpy as np
import scipy.linalg
import plotly.graph_objects as go
import sys

def run_gmsh_fem_corrected():
    # ---------------------------------------------------------
    # 1. Mesh generation with Gmsh
    # ---------------------------------------------------------
    # Setting balancing accuracy and computation time
    MESH_SIZE = 0.15
    
    print(f"Initializing Gmsh with Mesh Size = {MESH_SIZE}...")
    
    gmsh.initialize()
    gmsh.model.add("UnitCube")
    gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()
    
    gmsh.option.setNumber("Mesh.MeshSizeMin", MESH_SIZE)
    gmsh.option.setNumber("Mesh.MeshSizeMax", MESH_SIZE)
    
    gmsh.model.mesh.generate(3)
    
    # Retrieve data
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    nodes = np.array(node_coords).reshape(-1, 3)
    tag_to_idx = {tag: i for i, tag in enumerate(node_tags)}
    
    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=3)
    
    # Extract tetrahedra (Type 4)
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
    elements = np.zeros_like(elements_raw)
    
    for r in range(elements.shape[0]):
        for c in range(4):
            elements[r, c] = tag_to_idx[elements_raw[r, c]]
            
    gmsh.finalize()
    
    num_nodes = len(nodes)
    num_elems = len(elements)
    print(f"Mesh Generated: {num_nodes} nodes, {num_elems} elements.")

    # ---------------------------------------------------------
    # 2. Edge construction
    # ---------------------------------------------------------
    edge_to_id = {}
    id_to_edge = []
    element_edges = []
    local_pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]

    for elem in elements:
        cur_elem_edges = []
        for (a, b) in local_pairs:
            n1, n2 = elem[a], elem[b]
            key = (min(n1, n2), max(n1, n2))
            if key not in edge_to_id:
                edge_to_id[key] = len(id_to_edge)
                id_to_edge.append(key)
            cur_elem_edges.append(edge_to_id[key])
        element_edges.append(cur_elem_edges)

    num_edges = len(id_to_edge)

    # ---------------------------------------------------------
    # 3. PEC boundary conditions
    # ---------------------------------------------------------
    boundary_indices = set()
    tol = 1e-6
    for idx, (n1, n2) in enumerate(id_to_edge):
        p1, p2 = nodes[n1], nodes[n2]
        is_boundary = False
        for dim in range(3):
            if (abs(p1[dim]) < tol and abs(p2[dim]) < tol) or \
               (abs(p1[dim]-1.0) < tol and abs(p2[dim]-1.0) < tol):
                is_boundary = True
                break
        if is_boundary:
            boundary_indices.add(idx)

    free_dofs = [i for i in range(num_edges) if i not in boundary_indices]
    print(f"Total Edges: {num_edges}, Free DOFs: {len(free_dofs)}")

    # ---------------------------------------------------------
    # 4. Matrix assembly (Corrected version)
    # ---------------------------------------------------------
    S = np.zeros((num_edges, num_edges))
    M = np.zeros((num_edges, num_edges))
    
    print("Pre-calculating geometry...")
    elem_props = []
    
    for e_idx, elem in enumerate(elements):
        coords = nodes[elem]
        mat = np.c_[np.ones(4), coords]
        det_val = np.linalg.det(mat)
        vol = abs(det_val) / 6.0
        
        inv_mat = np.linalg.inv(mat)
        
        # --- [Important Correction] ---
        # Shape function N_i = a_i + b_i x + ...
        # Coefficient matrix C = (X^-1)^T
        # Gradient vector grad(N_i) = [b_i, c_i, d_i]
        # Components 1, 2, 3 of column vector i in X^-1 correspond to b_i, c_i, d_i respectively.
        # Previous code `inv_mat[:, 1:4]` was incorrectly fetching rows.
        # The correct way is `inv_mat[1:4, :].T`.
        grads = inv_mat[1:4, :].T 
        
        bbox_min = np.min(coords, axis=0)
        bbox_max = np.max(coords, axis=0)
        
        elem_props.append({"grads": grads, "vol": vol, "inv_mat": inv_mat, "min": bbox_min, "max": bbox_max})

    print("Assembling global matrices...")
    print_interval = max(1, num_elems // 10)

    for e_idx, elem in enumerate(elements):
        if e_idx % print_interval == 0:
            sys.stdout.write(f"\r{e_idx}/{num_elems}")
            sys.stdout.flush()
            
        props = elem_props[e_idx]
        grads = props["grads"]
        vol = props["vol"]
        g_ids = element_edges[e_idx]
        
        for i in range(6):
            na, nb = local_pairs[i]
            edge_i = g_ids[i]
            sign_i = 1.0 if elem[na] < elem[nb] else -1.0
            
            # Basis: lambda_a grad_b - lambda_b grad_a
            curl_Ni = 2.0 * np.cross(grads[na], grads[nb]) * sign_i
            
            for j in range(i, 6):
                nc, nd = local_pairs[j]
                edge_j = g_ids[j]
                sign_j = 1.0 if elem[nc] < elem[nd] else -1.0
                
                curl_Nj = 2.0 * np.cross(grads[nc], grads[nd]) * sign_j
                
                # S matrix
                val_S = np.dot(curl_Ni, curl_Nj) * vol
                S[edge_i, edge_j] += val_S
                if i != j: S[edge_j, edge_i] += val_S
                
                # M matrix
                def integ_L(p, q): return vol/10.0 if p==q else vol/20.0
                
                term_M = np.dot(grads[nb], grads[nd]) * integ_L(na, nc) \
                       - np.dot(grads[nb], grads[nc]) * integ_L(na, nd) \
                       - np.dot(grads[na], grads[nd]) * integ_L(nb, nc) \
                       + np.dot(grads[na], grads[nc]) * integ_L(nb, nd)
                
                val_M = term_M * sign_i * sign_j
                M[edge_i, edge_j] += val_M
                if i != j: M[edge_j, edge_i] += val_M

    print("\nSolving Eigenvalue Problem...")
    S_red = S[np.ix_(free_dofs, free_dofs)]
    M_red = M[np.ix_(free_dofs, free_dofs)]
    
    vals, vecs = scipy.linalg.eigh(S_red, M_red)
    
    print("\n--- Corrected Results ---")
    print(f"Theoretical TE101: {2 * np.pi**2:.4f}")
    
    physical_modes = []
    
    # If gradient calculation is correct, gradient modes (spurious) should be 0 (or around 1e-10).
    # Physical modes should start around 19.74.
    for i, val in enumerate(vals):
        if val > 15.0: # Physical mode
            full_vec = np.zeros(num_edges)
            full_vec[free_dofs] = vecs[:, i]
            physical_modes.append({"id": i+1, "k2": val, "vec": full_vec})
            print(f"Mode {i+1}: k² = {val:.4f} [Physical]")
        elif val > 0: # Might not be exactly 0 due to numerical error, but should be distinctly small
             print(f"Mode {i+1}: k² = {val:.6e} [Gradient/Spurious]")
        
        if len(physical_modes) >= 6: break

    # ---------------------------------------------------------
    # 5. Visualization
    # ---------------------------------------------------------
    print("Generating visualization...")
    
    # Grid sampling
    N_SAMPLE = 12
    g = np.linspace(0.05, 0.95, N_SAMPLE)
    gx, gy, gz = np.meshgrid(g, g, g)
    grid_pts = np.vstack([gx.ravel(), gy.ravel(), gz.ravel()]).T
    
    vis_samples = []
    
    for pt in grid_pts:
        pt_h = np.array([1, pt[0], pt[1], pt[2]])
        for e_idx, props in enumerate(elem_props):
            if np.any(pt < props["min"] - 1e-5) or np.any(pt > props["max"] + 1e-5): continue
            
            lam = pt_h @ props["inv_mat"]
            if np.all(lam >= -1e-6):
                vis_samples.append({"elem": e_idx, "lam": lam, "pos": pt})
                break
                
    vis_pos = np.array([s["pos"] for s in vis_samples])
    
    fig = go.Figure()
    
    # Mesh (Thin)
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
        vec_global = mode["vec"]
        vectors = []
        for s in vis_samples:
            e_idx = s["elem"]
            lam = s["lam"]
            props = elem_props[e_idx]
            grads = props["grads"]
            elem = elements[e_idx]
            g_ids = element_edges[e_idx]
            
            field = np.zeros(3)
            for k in range(6):
                na, nb = local_pairs[k]
                coef = vec_global[g_ids[k]]
                sign = 1.0 if elem[na] < elem[nb] else -1.0
                basis = lam[na] * grads[nb] - lam[nb] * grads[na]
                field += coef * sign * basis
            vectors.append(field)
            
        vectors = np.array(vectors)
        norms = np.linalg.norm(vectors, axis=1)
        max_v = np.max(norms) if len(norms) > 0 else 1.0
        if max_v > 1e-12: vectors /= max_v
        
        visible = (i == 0)
        trace = go.Cone(
            x=vis_pos[:,0], y=vis_pos[:,1], z=vis_pos[:,2],
            u=vectors[:,0], v=vectors[:,1], w=vectors[:,2],
            sizemode="scaled", sizeref=4.0,
            anchor="center", colorscale='Jet',
            name=f"Mode {mode['id']}", visible=visible,
            showscale=True, colorbar=dict(title='Field')
        )
        fig.add_trace(trace)
        
        vis_arr = [True] + [False] * len(physical_modes)
        vis_arr[i+1] = True
        
        steps.append(dict(
            method="update",
            args=[{"visible": vis_arr}, {"title": f"Mode {mode['id']}: k² = {mode['k2']:.4f}"}],
            label=f"{mode['id']}"
        ))

    if not physical_modes:
        print("No physical modes found.")
        return

    fig.update_layout(
        title=f"Corrected FEM: Mode {physical_modes[0]['id']} (k²={physical_modes[0]['k2']:.4f})",
        scene=dict(aspectmode='data', camera=dict(eye=dict(x=1.3, y=1.3, z=1.3)),
                   xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
        updatemenus=[dict(active=0, buttons=steps, x=0, y=1)]
    )
    
    out_file = "gmsh_fem_final.html"
    fig.write_html(out_file)
    print(f"\n[Info] Saved to {out_file}")
    fig.show()

if __name__ == "__main__":
    run_gmsh_fem_corrected()
