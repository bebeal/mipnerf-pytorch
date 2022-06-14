import torch
from os import path
from config import get_config
from model import MipNeRF
import imageio
from datasets import get_dataloader
from tqdm import tqdm
from pose_utils import visualize_depth, visualize_normals, to8b


def visualize(config):
    data = get_dataloader(config.dataset_name, config.base_dir, split="render", factor=config.factor, shuffle=False)

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
    )
    model.load_state_dict(torch.load(config.model_weight_path))
    model.eval()

    print("Generating Video using", len(data), "different view points")
    rgb_frames = []
    if config.visualize_depth:
        depth_frames = []
    if config.visualize_normals:
        normal_frames = []
    for ray in tqdm(data):
        img, dist, acc = model.render_image(ray, data.h, data.w, chunks=config.chunks)
        rgb_frames.append(img)
        if config.visualize_depth:
            depth_frames.append(to8b(visualize_depth(dist, acc, data.near, data.far)))
        if config.visualize_normals:
            normal_frames.append(to8b(visualize_normals(dist, acc)))

    imageio.mimwrite(path.join(config.log_dir, "video.mp4"), rgb_frames, fps=30, quality=10, codecs="hvec")
    if config.visualize_depth:
        imageio.mimwrite(path.join(config.log_dir, "depth.mp4"), depth_frames, fps=30, quality=10, codecs="hvec")
    if config.visualize_normals:
        imageio.mimwrite(path.join(config.log_dir, "normals.mp4"), normal_frames, fps=30, quality=10, codecs="hvec")


if __name__ == "__main__":
    config = get_config()
    visualize(config)
