import torch
from os import path
from config import get_config
from datasets import get_dataloader
from model import MipNeRF
from pose_utils import to8b, normalize, generate_spherical_cam_to_world
import numpy as np
from ray_utils import Rays, namedtuple_map
import matplotlib.pyplot as plt
import mcubes
import open3d as o3d
from plyfile import PlyData, PlyElement
from tqdm import tqdm


def extract_mesh(config):
    model = MipNeRF(
        use_viewdirs=config.use_viewdirs,
        randomized=config.randomized,
        ray_shape=config.ray_shape,
        white_bkgd=config.white_bkgd,
        num_levels=config.num_levels,
        num_samples=config.num_samples,
        hidden=config.hidden,
        density_noise=config.density_noise,
        density_bias=config.density_bias,
        rgb_padding=config.rgb_padding,
        resample_padding=config.resample_padding,
        min_deg=config.min_deg,
        max_deg=config.max_deg,
        viewdirs_min_deg=config.viewdirs_min_deg,
        viewdirs_max_deg=config.viewdirs_max_deg,
        device=config.device,
        return_raw=True
    )
    model.load_state_dict(torch.load(config.model_weight_path))
    model.eval()
    model = model.to(config.device)

    near = 2.0
    far = 6.0
    if config.dataset_name == "llff":
        near = 0.0
        far = 1.0

    xmin, xmax = config.x_range
    ymin, ymax = config.y_range
    zmin, zmax = config.z_range
    x = np.linspace(xmin, xmax, config.grid_size)
    y = np.linspace(ymin, ymax, config.grid_size)
    z = np.linspace(zmin, zmax, config.grid_size)
    origins = torch.FloatTensor(np.stack(np.meshgrid(x, y, z), -1).reshape(-1, 3))
    directions = torch.zeros_like(origins)
    viewdirs = torch.zeros_like(origins)
    radii = torch.ones_like(origins[..., :1]) * 0.0005
    ones = torch.ones_like(origins[..., :1])
    rays = Rays(
            origins=origins,
            directions=directions,
            viewdirs=viewdirs,
            radii=radii,
            lossmult=ones,
            near=ones * near,
            far=ones * far)

    print("Predicting occupancy")
    raws = []
    with torch.no_grad():
        for i in tqdm(range(0, rays[0].shape[0], config.chunks)):
            chunk_rays = namedtuple_map(lambda r: r[i:i + config.chunks].float().to(model.device), rays)
            img, dist, acc, raw = model(chunk_rays)
            raws.append(torch.mean(raw, dim=1).cpu())
    sigma = torch.cat(raws, dim=0)
    sigma = np.maximum(sigma[:, -1].cpu().numpy(), 0)
    sigma = sigma.reshape(config.grid_size, config.grid_size, config.grid_size)
    print("Extracting mesh")
    print("fraction occupied", np.mean(np.array(sigma > config.sigma_threshold), dtype=np.float32))
    vertices, triangles = mcubes.marching_cubes(sigma, config.sigma_threshold)
    vertices_ = (vertices / config.sigma_threshold).astype(np.float32)
    x_ = (ymax - ymin) * vertices_[:, 1] + ymin
    y_ = (xmax - xmin) * vertices_[:, 0] + xmin
    vertices_[:, 0] = x_
    vertices_[:, 1] = y_
    vertices_[:, 2] = (zmax - zmin) * vertices_[:, 2] + zmin
    vertices_.dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    face = np.empty(len(triangles), dtype=[('vertex_indices', 'i4', (3,))])
    face["vertex_indices"] = triangles
    mesh_path = path.join(config.log_dir, "mesh.ply")
    PlyData([PlyElement.describe(vertices_[:, 0], "vertex"), PlyElement.describe(face, "face")]).write(mesh_path)
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    print(f"Mesh has {len(mesh.vertices) / 1e6:.2f} M vertices and {len(mesh.triangles) / 1e6:.2f} M faces.")


if __name__ == "__main__":
    config = get_config()
    extract_mesh(config)
