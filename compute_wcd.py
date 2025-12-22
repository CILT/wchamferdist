import json, meshio
import torch
from localchamferdist.wchamfer import WeightedChamferDistance
from localchamferdist import ChamferDistance


category_to_weight = {
    "corner": 2.0,
    "edge": 1.5,
    "flat_face": 0.5,
    "sculpted_face": 3.0,
    "sculpted_flat": 0.5,
}

def load_mesh_and_weights(mesh_path, json_path=None, device="cpu"):
    # Load mesh
    # Only pointcloud matters for Chamfer distance
    # If json_path is None, no weights are loaded
    if mesh_path.lower().endswith(".obj"):
        verts_list = []
        with open(mesh_path, "r") as f:
            for line in f:
                if line.startswith("v "):   # vertex position only
                    _, x, y, z = line.strip().split()[:4]
                    verts_list.append([float(x), float(y), float(z)])

        verts = torch.tensor(verts_list, dtype=torch.float32, device=device)[None, :, :]
    else:
        # PLY and others are safe
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
    ], dtype=torch.float32, device=device)[None, :]  # Shape (1, P)

    return verts, weights


if __name__ == "__main__":

    # root = "/home/cllullt/blender-4.4.3-linux-x64_anakena"
    # # root = "/home/cllullt/blender-4.4.3-linux-x64"
    # source_path = f"{root}/cube.ply"
    # # source_path = f"/media/cllullt/Arxius/Meus_Documents/PhD/Investigacion/data/primitives/cube_2.obj"
    # categories_path = f"{root}/cube.ply.json"
    # target_path = f"/media/cllullt/Arxius/Meus_Documents/PhD/Investigacion/data/primitives/cube.obj"
    # #target_path = f"/media/cllull/Arxius/Meus_Documents/PhD/Investigacion/data/primitives/cube_flat.ply"
    # #target_path = f"/media/cllullt/Arxius/Meus_Documents/PhD/Investigacion/data/primitives/cube_simple.obj"

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # source_verts, weights = load_mesh_and_weights(source_path, categories_path, device=device)
    # target_verts, _ = load_mesh_and_weights(target_path, None, device=device)

    # print(f"Loaded source vertices shape: {source_verts.shape}")
    # print(f"\t with weights shape: {weights.shape}")
    # print(f"Loaded target vertices shape: {target_verts.shape}")

    # wcd = WeightedChamferDistance()
    # print("Computing weighted Chamfer distance...")
    # dist = 0.5 * wcd(
    #     source_verts,
    #     target_verts,
    #     weights_source=weights,
    #     reverse=False,
    #     bidirectional=True,
    #     point_reduction="mean"
    # )
    # print("Weighted Chamfer distance:", dist.item())
    
    # dist_self = wcd(source_verts, source_verts, weights_source=weights)
    # print("Weighted Chamfer distance (self):", dist_self.detach().cpu().item())
    # dist_self = wcd(target_verts, target_verts, weights_source=weights[0][:4])
    # print("Weighted Chamfer distance (self):", dist_self.detach().cpu().item())

    # cd = ChamferDistance()
    # dist_unweighted = 0.5 * cd(
    #     source_verts,
    #     target_verts,
    #     bidirectional=True,
    #     point_reduction="mean"
    # )
    # print("Unweighted Chamfer distance:", dist_unweighted.item())
    # # As a sanity check, chamfer distance between a pointcloud and itself must be
    # # zero.
    # dist_self = cd(source_verts, source_verts)
    # print("Chamfer distance (self):", dist_self.detach().cpu().item())
    # dist_self = cd(target_verts, target_verts)
    # print("Chamfer distance (self):", dist_self.detach().cpu().item())


    import os
    import csv
    import torch

    # ------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------
    ROOT_DIR = "/media/cllullt/Arxius/Meus_Documents/PhD/Investigacion/data/reconstructions/300x300_st_0_02"
    OUTPUT_CSV = "evaluation_results.csv"

    TARGETS = {
        "neuralangelo": "neuralangelo.ply",
        "neus": "neus.ply",
        "vggt": os.path.join("sparse", "vggt.ply"),
    }

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(DEVICE)

    tracker = ResourceTracker(DEVICE)

    # ------------------------------------------------------------
    # Initialize distances
    # ------------------------------------------------------------
    cd = ChamferDistance()
    wcd = WeightedChamferDistance()

    # ------------------------------------------------------------
    # CSV setup
    # ------------------------------------------------------------
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "experiment_id",
            "method",
            "cd",
            "wcd",
            "num_source_pts",
            "num_target_pts"
        ])

        # --------------------------------------------------------
        # Traverse experiments
        # --------------------------------------------------------
        for exp_name in sorted(os.listdir(ROOT_DIR)):
            exp_dir = os.path.join(ROOT_DIR, exp_name)
            if not os.path.isdir(exp_dir):
                continue

            print(f"\n=== Processing {exp_name} ===")

            gt_obj = os.path.join(exp_dir, "gt.obj")
            gt_json = os.path.join(exp_dir, "gt.json")

            if not (os.path.exists(gt_obj) and os.path.exists(gt_json)):
                print("  Skipping: missing gt.obj or gt.json")
                continue

            # ----------------------------------------------------
            # Load GT once
            # ----------------------------------------------------
            source_verts, source_weights = load_mesh_and_weights(
                gt_obj,
                gt_json,
                device=DEVICE
            )

            num_source_pts = source_verts.shape[1]

            # ----------------------------------------------------
            # Compare with each reconstruction
            # ----------------------------------------------------
            for method, rel_path in TARGETS.items():
                target_path = os.path.join(exp_dir, rel_path)
                if not os.path.exists(target_path):
                    print(f"  [{method}] missing → skipped")
                    continue

                print(f"  [{method}] evaluating...")

                target_verts, _ = load_mesh_and_weights(
                    target_path,
                    None,
                    device=DEVICE
                )

                num_target_pts = target_verts.shape[1]

                # ------------------------------
                # Unweighted CD
                # ------------------------------
                cd_val = 0.5 * cd(
                    source_verts,
                    target_verts,
                    bidirectional=True,
                    point_reduction="mean"
                )

                # ------------------------------
                # Weighted CD
                # ------------------------------
                wcd_val = 0.5 * wcd(
                    source_verts,
                    target_verts,
                    weights_source=source_weights,
                    reverse=False,
                    bidirectional=True,
                    point_reduction="mean"
                )

                # ------------------------------
                # Write CSV row
                # ------------------------------
                writer.writerow([
                    exp_name,
                    method,
                    cd_val.item(),
                    wcd_val.item(),
                    num_source_pts,
                    num_target_pts
                ])

                print(f"    CD = {cd_val.item():.6f}, WCD = {wcd_val.item():.6f}")


