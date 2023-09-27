import numpy as np
import torch
import igl
import scipy as sp

def compute_edges(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    # NOTE modified from pytorch3d.structures.meshes.Meshes._compute_packed
    """
    Computes edges in packed form from the packed version of faces and verts.
    """
    F = faces.shape[0]
    v0, v1, v2 = faces.chunk(3, dim=1)
    e01 = torch.cat([v0, v1], dim=1)  # (sum(F_n), 2)
    e12 = torch.cat([v1, v2], dim=1)  # (sum(F_n), 2)
    e20 = torch.cat([v2, v0], dim=1)  # (sum(F_n), 2)

    edges = torch.cat([e12, e20, e01], dim=0)  # (sum(F_n)*3, 2)
    edges, _ = edges.sort(dim=1)

    V = verts.shape[0]
    edges_hash = V * edges[:, 0] + edges[:, 1]
    u, inverse_idxs = torch.unique(edges_hash, return_inverse=True)

    edges_packed = torch.stack([u // V, u % V], dim=1)
    return edges_packed


def compute_eigenfunctions(verts, faces, k):
    L = -igl.cotmatrix(verts, faces)
    M = igl.massmatrix(verts, faces, igl.MASSMATRIX_TYPE_VORONOI)

    try:
        eigenvalues, eigenfunctions = sp.sparse.linalg.eigsh(L, k, M, sigma=0, which="LM")
    except RuntimeError as e:
        # singular coefficient matrix
        c = 1e-10
        L = L + c * sp.sparse.eye(L.shape[0])
        eigenvalues, eigenfunctions = sp.sparse.linalg.eigsh(L, k, M, sigma=0, which="LM")

    assert np.all(np.max(eigenfunctions, axis=0) != np.min(eigenfunctions, axis=0))

    return eigenfunctions, eigenvalues



def laplacian(verts: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    # NOTE taken from pytorch3d.ops.laplacian
    """
    Computes the laplacian matrix.
    The definition of the laplacian is
    L[i, j] =    -1       , if i == j
    L[i, j] = 1 / deg(i)  , if (i, j) is an edge
    L[i, j] =    0        , otherwise
    where deg(i) is the degree of the i-th vertex in the graph.

    Args:
        verts: tensor of shape (V, 3) containing the vertices of the graph
        edges: tensor of shape (E, 2) containing the vertex indices of each edge
    Returns:
        L: Sparse FloatTensor of shape (V, V)
    """
    V = verts.shape[0]

    e0, e1 = edges.unbind(1)

    idx01 = torch.stack([e0, e1], dim=1)  # (E, 2)
    idx10 = torch.stack([e1, e0], dim=1)  # (E, 2)
    idx = torch.cat([idx01, idx10], dim=0).t()  # (2, 2*E)

    # First, we construct the adjacency matrix,
    # i.e. A[i, j] = 1 if (i,j) is an edge, or
    # A[e0, e1] = 1 &  A[e1, e0] = 1
    ones = torch.ones(idx.shape[1], dtype=torch.float32, device=verts.device)
    A = torch.sparse.FloatTensor(idx, ones, (V, V))

    # the sum of i-th row of A gives the degree of the i-th vertex
    deg = torch.sparse.sum(A, dim=1).to_dense()

    # We construct the Laplacian matrix by adding the non diagonal values
    # i.e. L[i, j] = 1 ./ deg(i) if (i, j) is an edge
    deg0 = deg[e0]
    deg0 = torch.where(deg0 > 0.0, 1.0 / deg0, deg0)
    deg1 = deg[e1]
    deg1 = torch.where(deg1 > 0.0, 1.0 / deg1, deg1)
    val = torch.cat([deg0, deg1])
    L = torch.sparse.FloatTensor(idx, val, (V, V))

    # Then we add the diagonal values L[i, i] = -1.
    idx = torch.arange(V, device=verts.device)
    idx = torch.stack([idx, idx], dim=0)
    ones = torch.ones(idx.shape[1], dtype=torch.float32, device=verts.device)
    L -= torch.sparse.FloatTensor(idx, ones, (V, V))

    return L

def export_to_ply(filename, verts, faces, quality):
    """
    verts: [..., 3]
    quality: [...]
    """
    verts = verts.reshape(-1, 3)
    faces = faces.reshape(-1, 3)
    quality = quality.reshape(-1)
    n_pts = verts.shape[0]
    n_faces = faces.shape[0]
    
    with open(filename, 'w') as plyfile:
        plyfile.write('ply\n')
        plyfile.write('format ascii 1.0\n')
        plyfile.write('comment trisst custom\n')
        plyfile.write(f'element vertex {n_pts}\n')
        plyfile.write('property float x\n')
        plyfile.write('property float y\n')
        plyfile.write('property float z\n')
        plyfile.write('property float quality\n')
        plyfile.write(f'element face {n_faces}\n')
        plyfile.write('property list uchar int vertex_indices\n')
        plyfile.write('end_header\n')
        for i in range(n_pts):
            pt = verts[i]
            q = quality[i]
            plyfile.write(f'{pt[0]} {pt[1]} {pt[2]} {q}\n')

        for i in range(n_faces):
            f = faces[i]
            plyfile.write(f'3 {f[0]} {f[1]} {f[2]}\n')

