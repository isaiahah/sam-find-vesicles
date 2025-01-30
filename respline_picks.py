import numpy as np
from argparse import ArgumentParser
from scipy.interpolate import splprep, splev
import os
from pathlib import Path

# Parse command line arguments
parser = ArgumentParser(
    prog="respline_picks.py",
    description="Fit a spline to coordinates and return the number of specified points on the spline"
)
parser.add_argument(
    "--input_dir",
    type=str,
    default=".",
    help="Directory containing np arrays of edge coordinates"
)
parser.add_argument(
    "--spline_density",
    type=int,
    default=20000,
    help="Number of points to pick from each spline fit to a vesicle"
)
parser.add_argument(
    "--spline_dir",
    type=str,
    default=None,
    help="Path to save np array of final membrane spline coordinates"
)

args = parser.parse_args()
input_dir = args.input_dir
spline_density = args.spline_density
spline_dir = args.spline_dir

input_dir_files = [entry for entry in os.scandir(input_dir) if entry.is_file()]
for input_file in input_dir_files:
    # Skip non .npy files
    if not input_file.name.endswith(".npy"):
        continue
    # Load points from file
    points = np.load(input_file.path)
    # Sort points by angle
    points_angles = np.arctan2(points[:, 1] - np.mean(points[:, 1]), 
                               points[:, 0] - np.mean(points[:, 0]))
    points = points[np.argsort(points_angles), :]
    # Generate a spline through the points
    try:
        tck, u = splprep([points[:, 0], points[:, 1]], k=3)
    except Exception as e:
        print(f"File {input_file.name} has error {e}")
    
    spline = splev(np.linspace(0, 1.0, spline_density), tck)
    spline = np.unique(np.round(spline).astype(int).T, axis=0)
    
    # Save spline points to file
    spline_dir = Path(spline_dir)
    np.save(spline_dir / input_file.name, spline)

