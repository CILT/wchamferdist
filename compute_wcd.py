import json, meshio
import torch
from localchamferdist.wchamfer import WeightedChamferDistance
from localchamferdist import ChamferDistance


category_to_weight = {
    "corner": 2.0,
    "edge": 1.5,
    "flat_face": 0.5,
    "sculpted_face": 3.0
}

def load_mesh_and_weights(mesh_path, json_path=None, device="cpu"):
    # Load mesh
    # Only pointcloud matters for Chamfer distance
    # If json_path is None, no weights are loaded
    mesh = meshio.read(mesh_path)
    verts = torch.tensor(mesh.points, dtype=torch.float32, device=device)[None, :, :]  # Shape (1, P, 3)

    if json_path is None:
        return verts, None
    
    # Load weights
    with open(json_path) as f:
        vertex_categories = json.load(f)  # dict: {"0": "corner", ...}

    weights = torch.tensor([
        category_to_weight.get(vertex_categories.get(str(i), "default"), 1.0)
        for i in range(verts.shape[1])
    ], dtype=torch.float32, device=device)[None, :]  # Shape (1, N)

    return verts, weights


if __name__ == "__main__":

    root = "/home/cllull/Desktop/blender-4.4.3-linux-x64_anakena"
    source_path = f"{root}/cube.ply"
    categories_path = f"{root}/cube.ply.json"
    target_path = f"/media/cllull/Arxius/Meus_Documents/PhD/Investigacion/data/primitives/cube.obj"
    target_path = f"/media/cllull/Arxius/Meus_Documents/PhD/Investigacion/data/primitives/cube_flat.ply"
    target_path = f"/media/cllull/Arxius/Meus_Documents/PhD/Investigacion/data/primitives/cube_2.obj"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    source_verts, weights = load_mesh_and_weights(source_path, categories_path, device=device)
    target_verts, _ = load_mesh_and_weights(target_path, None, device=device)

    print(f"Loaded source vertices shape: {source_verts.shape}")
    print(f"\t with weights shape: {weights.shape}")
    print(f"Loaded target vertices shape: {target_verts.shape}")

    wcd = WeightedChamferDistance()
    dist = 0.5 * wcd(
        source_verts,
        target_verts,
        # weights_source=weights,
        reverse=False,
        bidirectional=True
    )

    print("Weighted Chamfer distance:", dist.item())
    dist_self = wcd(source_verts, source_verts, weights_source=weights)
    print("Chamfer distance (self):", dist_self.detach().cpu().item())
    dist_self = wcd(target_verts, target_verts, weights_source=weights)
    print("Chamfer distance (self):", dist_self.detach().cpu().item())

    cd = ChamferDistance()
    dist_unweighted = 0.5 * cd(
        source_verts,
        target_verts,
        bidirectional=True
    )
    print("Unweighted Chamfer distance:", dist_unweighted.item())
    # As a sanity check, chamfer distance between a pointcloud and itself must be
    # zero.
    dist_self = cd(source_verts, source_verts)
    print("Chamfer distance (self):", dist_self.detach().cpu().item())
    dist_self = cd(target_verts, target_verts)
    print("Chamfer distance (self):", dist_self.detach().cpu().item())
