import json, meshio
import torch
import os
import csv
import torch
from localchamferdist.wchamfer import WeightedChamferDistance
from localchamferdist import ChamferDistance
from ResourceTracker import ResourceTracker


category_to_weight = {
    "corner": 100.0,
    "edge": 50.0,
    "flat_face": 1.0,
    "sculpted_face": 300.0,
    "sculpted_flat": 1.0,
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
    # ------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------
    # ROOT_DIR = "/media/cllullt/Arxius/Meus_Documents/PhD/Investigacion/data/reconstructions/300x300_st_0_02"
    ROOT_DIR = "/media/cllull/Arxius/Meus_Documents/PhD/Congresos_etc/2026_SHREC/Reconstructions/gathered_objects"
    OUTPUT_CSV = "evaluation_results_SHREC_3.csv"

    SOURCES = {
        "colmap": "colmap.ply",
        "neuralangelo": "neuralangelo.ply",
        "neus": "neus.ply",
        "vggt": os.path.join("sparse", "vggt.ply"),
        "vggt_corrected": os.path.join("sparse", "vggt_corrected.ply"),
    }
    SOURCES = {
        "colmap": "colmap.ply",
        "neus": "NeuS_00140000.ply",
        "sam_sugar": "SAMplify_SuGaR.obj",
        "nerf_instantngp": "NeRF_InstantNGP.ply",
        "vggt_100000": "vggt_sparse_100000_aligned.ply",
        "vggt": "vggt_sparse_aligned.ply",
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
            "hcd",
            "hcd_2",
            "wcd",
            "hwcd",
            "hwcd_2",
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
            # Load GT once (ground truth = Target points)
            # ----------------------------------------------------
            target_verts, target_weights = load_mesh_and_weights(
                gt_obj,
                gt_json,
                device=DEVICE
            )

            num_target_pts = target_verts.shape[1]

            # ----------------------------------------------------
            # Compare with each reconstruction (sources)
            # ----------------------------------------------------
            for method, rel_path in SOURCES.items():
                source_path = os.path.join(exp_dir, rel_path)
                if not os.path.exists(source_path):
                    print(f"  [{method}] missing → skipped")
                    continue

                print(f"  [{method}] evaluating...")

                # Reconstructions are the Source points
                source_verts, _ = load_mesh_and_weights(
                    source_path,
                    None,
                    device=DEVICE
                )

                num_source_pts = source_verts.shape[1]

                # Forward time:
                tracker.start()
                forward = cd(
                    target_verts,
                    source_verts,
                    reverse=True,
                    bidirectional=False,
                    point_reduction="mean"
                )
                backward = cd(
                    target_verts,
                    source_verts,
                    reverse=False,
                    bidirectional=False,
                    point_reduction="mean"
                )
                cd_stats = tracker.stop()
                print(f"    Time taken cd: {cd_stats['wall_time_sec']:.2f}")

                tracker.start()
                w_forward = wcd(
                    target_verts,
                    source_verts,
                    weights_source=target_weights,
                    reverse=True,
                    bidirectional=False,
                    point_reduction="mean"
                )
                w_backward = wcd(
                    target_verts,
                    source_verts,
                    weights_source=target_weights,
                    reverse=False,
                    bidirectional=False,
                    point_reduction="mean"
                )
                wcd_stats = tracker.stop()
                print(f"    Time taken wcd: {wcd_stats['wall_time_sec']:.2f}")

                # ------------------------------
                # Unweighted CD
                # ------------------------------
                tracker.start()
                # cd_val = 0.5 * cd(
                #     source_verts,
                #     target_verts,
                #     bidirectional=True,
                #     point_reduction="mean"
                # )
                cd_val = 0.5 * (forward + backward)
                cd_stats = tracker.stop()
                print(f"    Time taken: {cd_stats['wall_time_sec']:.2f}")

                # ------------------------------
                # Harmonic CD
                # ------------------------------
                tracker.start()
                # Using (target, source) ordering; `reverse` controls
                # the direction: `reverse=True` computes source->target
                hcd = 2.0 * forward * backward / (forward + backward + 1e-8)
                hcd_stats = tracker.stop()
                print(f"    Time taken: {hcd_stats['wall_time_sec']:.2f}")

                hcd_2 = 2.0 * (1 + forward) * (1 + backward) / ((1 + forward) + (1 + backward)) - 1.0
                print(f"hcd_2 check: {hcd.item():.6f} vs {hcd_2.item():.6f}")
                
                # ------------------------------
                # Weighted CD
                # ------------------------------
                tracker.start()
                # For weighted CD, weights belong to the ground-truth (Target)
                # so pass them as weights_target to match the target argument.
                # wcd_val = 0.5 * wcd(
                #     target_verts,
                #     source_verts,
                #     weights_source=target_weights,
                #     reverse=True,
                #     bidirectional=True,
                #     point_reduction="mean"
                # )
                wcd_stats = tracker.stop()
                print(f"    Time taken: {wcd_stats['wall_time_sec']:.2f}")

                wcd_val = 0.5 * (w_forward + w_backward)
                # ------------------------------
                # Harmonic Weighted CD
                # ------------------------------
                tracker.start()
                # `reverse=True` for forward (source->target)

                hwcd = 2.0 * forward * backward / (forward + backward + 1e-8)
                hwcd_stats = tracker.stop()
                print(f"    Time taken: {hwcd_stats['wall_time_sec']:.2f}")

                hwcd_2 = 2.0 * (1 + w_forward) * (1 + w_backward) / ((1 + w_forward) + (1 + w_backward)) - 1.0
                print(f"hwcd_2 check: {hwcd.item():.6f} vs {hwcd_2.item():.6f}")

                # ------------------------------
                # Write CSV row
                # ------------------------------
                writer.writerow([
                    exp_name,
                    method,
                    cd_val.item(),
                    hcd.item(),
                    hcd_2.item(),

                    wcd_val.item(),
                    hwcd.item(),
                    hwcd_2.item(),
                    num_source_pts,
                    num_target_pts
                ])

                print(f"    CD  = {cd_val.item():.6f}, CD_h  = {hcd.item():.6f},\n    WCD = {wcd_val.item():.6f}, WCD_h = {hwcd.item():.6f}")


