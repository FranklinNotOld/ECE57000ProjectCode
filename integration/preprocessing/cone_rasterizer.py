"""
Rasterize traffic cones into a Gaussian-blob image channel for Mode B-A.

Each cone is rendered as a 2D Gaussian with sigma = 1m (= pixels_per_meter px).
Output is a uint8 [W_px, H_px] array matching the map_mask grid.

Usage (standalone test):
    python integration/preprocessing/cone_rasterizer.py \
        --cones_csv dataset_collector/output/4L_close_2_3/cones.csv \
        --W_px 5001 --H_px 76 --pixels_per_meter 5.0
"""
import numpy as np
import pandas as pd


def rasterize_cones(cones_csv_or_array, W_px, H_px, pixels_per_meter,
                    sigma_m=1.0):
    """
    Create a single-channel uint8 raster of Gaussian cone blobs.

    Parameters
    ----------
    cones_csv_or_array : str or np.ndarray
        Path to cones.csv (columns: cone_id, x, y) or Nx2 array of (x, y).
    W_px, H_px : int
        Raster dimensions matching map_mask (x_dim, y_dim).
    pixels_per_meter : float
        Scale factor (typically 5.0).
    sigma_m : float
        Gaussian standard deviation in meters (default 1.0).

    Returns
    -------
    channel : np.ndarray, shape [W_px, H_px], dtype uint8
    """
    # Load cone positions
    if isinstance(cones_csv_or_array, str):
        df = pd.read_csv(cones_csv_or_array)
        cones_xy = df[['x', 'y']].values
    else:
        cones_xy = np.asarray(cones_csv_or_array)

    if len(cones_xy) == 0:
        return np.zeros((W_px, H_px), dtype=np.uint8)

    sigma_px = sigma_m * pixels_per_meter
    # Truncate Gaussian at 3*sigma for efficiency
    radius = int(np.ceil(3.0 * sigma_px))

    # Pre-compute 1D Gaussian kernel
    k = np.arange(-radius, radius + 1, dtype=np.float64)
    gauss_1d = np.exp(-0.5 * (k / sigma_px) ** 2)

    # Accumulate in float
    canvas = np.zeros((W_px, H_px), dtype=np.float64)

    for cx, cy in cones_xy:
        # Convert scene coords to pixel coords: px = coord * pixels_per_meter
        px_x = cx * pixels_per_meter
        px_y = cy * pixels_per_meter

        ix = int(round(px_x))
        iy = int(round(px_y))

        # Compute bounds with clipping
        x_lo = max(ix - radius, 0)
        x_hi = min(ix + radius, W_px - 1)
        y_lo = max(iy - radius, 0)
        y_hi = min(iy + radius, H_px - 1)

        if x_lo > x_hi or y_lo > y_hi:
            continue

        # Slice into the 1D kernel
        kx = gauss_1d[(x_lo - ix + radius):(x_hi - ix + radius + 1)]
        ky = gauss_1d[(y_lo - iy + radius):(y_hi - iy + radius + 1)]

        # Outer product -> 2D Gaussian patch
        canvas[x_lo:x_hi + 1, y_lo:y_hi + 1] += np.outer(kx, ky)

    # Normalize to [0, 255]
    max_val = canvas.max()
    if max_val > 0:
        canvas = canvas / max_val * 255.0

    return canvas.astype(np.uint8)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Cone rasterizer test')
    parser.add_argument('--cones_csv', type=str, required=True)
    parser.add_argument('--W_px', type=int, default=5001)
    parser.add_argument('--H_px', type=int, default=76)
    parser.add_argument('--pixels_per_meter', type=float, default=5.0)
    parser.add_argument('--sigma_m', type=float, default=1.0)
    args = parser.parse_args()

    channel = rasterize_cones(args.cones_csv, args.W_px, args.H_px,
                              args.pixels_per_meter, args.sigma_m)
    print(f"Output shape: {channel.shape}, dtype: {channel.dtype}")
    print(f"  min={channel.min()}, max={channel.max()}, "
          f"nonzero={np.count_nonzero(channel)}")
