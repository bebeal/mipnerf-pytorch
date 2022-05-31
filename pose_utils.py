import numpy as np
import matplotlib.cm as cm
from scipy import signal


def generate_spiral_cam_to_world(radii, focus_depth, n_poses=120):
    """
    Generate a spiral path for rendering
    ref: https://github.com/kwea123/nerf_pl/blob/master/datasets/llff.py
    Computes poses that follow a spiral path for rendering purpose.
    See https://github.com/Fyusion/LLFF/issues/19
    In particular, the path looks like:
    https://tinyurl.com/ppn7ddat
    Inputs:
        radii: (3) radii of the spiral for each axis
        focus_depth: float, the depth that the spiral poses look at
        n_poses: int, number of poses to create along the path
    Outputs:
        poses_spiral: (n_poses, 3, 4) the cam to world transformation matrix of a spiral path
    """

    spiral_cams = []
    for t in np.linspace(0, 4 * np.pi, n_poses + 1)[:-1]:  # rotate 4pi (2 rounds)
        # the parametric function of the spiral (see the interactive web)
        center = np.array([(np.cos(t) * 0.5) - 2, -np.sin(t) - 0.5, -np.sin(0.5 * t) * 0.75]) * radii
        # the viewing z axis is the vector pointing from the focus_depth plane to center
        z = normalize(center - np.array([0, 0, -focus_depth]))
        # compute other axes as in average_poses
        x = normalize(np.cross(np.array([0, 1, 0]), z))
        y = np.cross(z, x)
        spiral_cams += [np.stack([y, z, x, center], 1)]
    return np.stack(spiral_cams, 0)


def generate_spherical_cam_to_world(radius, n_poses=120):
    """
    Generate a 360 degree spherical path for rendering
    ref: https://github.com/kwea123/nerf_pl/blob/master/datasets/llff.py
    ref: https://github.com/yenchenlin/nerf-pytorch/blob/master/load_blender.py
    Create circular poses around z axis.
    Inputs:
        radius: the (negative) height and the radius of the circle.
    Outputs:
        spheric_cams: (n_poses, 3, 4) the cam to world transformation matrix of a circular path
    """

    def spheric_pose(theta, phi, radius):
        trans_t = lambda t: np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, t],
            [0, 0, 0, 1],
        ], dtype=np.float)

        rotation_phi = lambda phi: np.array([
            [1, 0, 0, 0],
            [0, np.cos(phi), -np.sin(phi), 0],
            [0, np.sin(phi), np.cos(phi), 0],
            [0, 0, 0, 1],
        ], dtype=np.float)

        rotation_theta = lambda th: np.array([
            [np.cos(th), 0, -np.sin(th), 0],
            [0, 1, 0, 0],
            [np.sin(th), 0, np.cos(th), 0],
            [0, 0, 0, 1],
        ], dtype=np.float)
        cam_to_world = trans_t(radius)
        cam_to_world = rotation_phi(phi / 180. * np.pi) @ cam_to_world
        cam_to_world = rotation_theta(theta) @ cam_to_world
        cam_to_world = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
                                dtype=np.float) @ cam_to_world
        return cam_to_world

    spheric_cams = []
    for th in np.linspace(0, 2 * np.pi, n_poses + 1)[:-1]:
        spheric_cams += [spheric_pose(th, -30, radius)]
    return np.stack(spheric_cams, 0)


def recenter_poses(poses):
    """Recenter poses according to the original NeRF code."""
    poses_ = poses.copy()
    bottom = np.reshape([0, 0, 0, 1.], [1, 4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)
    poses = np.linalg.inv(c2w) @ poses
    poses_[:, :3, :4] = poses[:, :3, :4]
    poses = poses_
    return poses


def poses_avg(poses):
    """Average poses according to the original NeRF code."""
    hwf = poses[0, :3, -1:]
    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([look_at(vec2, up, center), hwf], 1)
    return c2w


def look_at(z, up, pos):
    """Construct look at view matrix
    https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/lookat-function
    """
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def flatten(x):
    # Always flatten out the height x width dimensions
    x = [y.reshape([-1, y.shape[-1]]) for y in x]
    # concatenate all data into one list
    x = np.concatenate(x, axis=0)
    return x


def normalize(x):
    """Normalization helper function."""
    return x / np.linalg.norm(x)


def convolve2d(z, f):
  return signal.convolve2d(z, f, mode='same')


def depth_to_normals(depth):
    """Assuming `depth` is orthographic, linearize it to a set of normals."""
    f_blur = np.array([1, 2, 1]) / 4
    f_edge = np.array([-1, 0, 1]) / 2
    dy = convolve2d(depth, f_blur[None, :] * f_edge[:, None])
    dx = convolve2d(depth, f_blur[:, None] * f_edge[None, :])
    inv_denom = 1 / np.sqrt(1 + dx**2 + dy**2)
    normals = np.stack([dx * inv_denom, dy * inv_denom, inv_denom], -1)
    return normals


def sinebow(h):
    """A cyclic and uniform colormap, see http://basecase.org/env/on-rainbows."""
    f = lambda x: np.sin(np.pi * x)**2
    return np.stack([f(3 / 6 - h), f(5 / 6 - h), f(7 / 6 - h)], -1)


def visualize_normals(depth, acc, scaling=None):
    """Visualize fake normals of `depth` (optionally scaled to be isotropic)."""
    if scaling is None:
        mask = ~np.isnan(depth)
        x, y = np.meshgrid(
            np.arange(depth.shape[1]), np.arange(depth.shape[0]), indexing='xy')
        xy_var = (np.var(x[mask]) + np.var(y[mask])) / 2
        z_var = np.var(depth[mask])
        scaling = np.sqrt(xy_var / z_var)

        scaled_depth = scaling * depth
        normals = depth_to_normals(scaled_depth)
        vis = np.isnan(normals) + np.nan_to_num((normals + 1) / 2, 0)

        # Set non-accumulated pixels to white.
        if acc is not None:
            vis = vis * acc[:, :, None] + (1 - acc)[:, :, None]

        return vis


def visualize_depth(depth,
                    acc=None,
                    near=None,
                    far=None,
                    ignore_frac=0,
                    curve_fn=lambda x: -np.log(x + np.finfo(np.float32).eps),
                    modulus=0,
                    colormap=None):
    """Visualize a depth map.
    Args:
      depth: A depth map.
      acc: An accumulation map, in [0, 1].
      near: The depth of the near plane, if None then just use the min().
      far: The depth of the far plane, if None then just use the max().
      ignore_frac: What fraction of the depth map to ignore when automatically
        generating `near` and `far`. Depends on `acc` as well as `depth'.
      curve_fn: A curve function that gets applied to `depth`, `near`, and `far`
        before the rest of visualization. Good choices: x, 1/(x+eps), log(x+eps).
          Note that the default choice will flip the sign of depths, so that the
          default colormap (turbo) renders "near" as red and "far" as blue.
      modulus: If > 0, mod the normalized depth by `modulus`. Use (0, 1].
      colormap: A colormap function. If None (default), will be set to
        matplotlib's turbo if modulus==0, sinebow otherwise.
    Returns:
      An RGB visualization of `depth`.
    """
    if acc is None:
        acc = np.ones_like(depth)
    acc = np.where(np.isnan(depth), np.zeros_like(acc), acc)

    # Sort `depth` and `acc` according to `depth`, then identify the depth values
    # that span the middle of `acc`, ignoring `ignore_frac` fraction of `acc`.
    sortidx = np.argsort(depth.reshape([-1]))
    depth_sorted = depth.reshape([-1])[sortidx]
    acc_sorted = acc.reshape([-1])[sortidx]
    cum_acc_sorted = np.cumsum(acc_sorted)
    mask = ((cum_acc_sorted >= cum_acc_sorted[-1] * ignore_frac) &
            (cum_acc_sorted <= cum_acc_sorted[-1] * (1 - ignore_frac)))
    depth_keep = depth_sorted[mask]

    # If `near` or `far` are None, use the highest and lowest non-NaN values in
    # `depth_keep` as automatic near/far planes.
    eps = np.finfo(np.float32).eps
    near = near or depth_keep[0] - eps
    far = far or depth_keep[-1] + eps

    # Curve all values.
    depth, near, far = [curve_fn(x) for x in [depth, near, far]]

    # Wrap the values around if requested.
    if modulus > 0:
        value = np.mod(depth, modulus) / modulus
        colormap = colormap or sinebow
    else:
        # Scale to [0, 1].
        value = np.nan_to_num(
            np.clip((depth - np.minimum(near, far)) / np.abs(far - near), 0, 1))
        colormap = colormap or cm.get_cmap('turbo')

    vis = colormap(value)[:, :, :3]

    # Set non-accumulated pixels to white.
    vis = vis * acc[:, :, None] + (1 - acc)[:, :, None]

    return vis


def to8b(img):
    if len(img.shape) >= 3:
        return np.array([to8b(i) for i in img])
    else:
        return (255 * np.clip(np.nan_to_num(img), 0, 1)).astype(np.uint8)


def to_float(img):
    if len(img.shape) >= 3:
        return np.array([to_float(i) for i in img])
    else:
        return (img / 255.).astype(np.float32)
