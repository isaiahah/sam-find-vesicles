# IMPORTANT INFO
# Tested in the vesicle-picker-samhq conda env, but should run in any env with 
# vesicle-picker and all Python dependencies.
# Run this on the picks generated by pick_membrane.py


# Imports
from vesicle_picker import (
    postprocess,
    helpers,
    external_import,
    external_export
)
from cryosparc.tools import Dataset
import numpy as np
import cv2
from scipy.signal import find_peaks
from tqdm import tqdm
from argparse import ArgumentParser
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
from pathlib import Path
from math import sqrt
import sys
import os
import time

t_pixels_in_rectangle = 0
t_bin_rectangle = 0
t_find_bilayers = 0
t_fit_splines = 0

# Initialize helper functions
def downsample_contour(particles, dist, psize):
    # Select a sparse set of particles from a contour such that each particle
    # is at least dist nm from its previous neighbor
    
    # NOTE: If the particles were generated by some method which does not return
    # them in clockwise order, apply this code. However, it may cause poor
    # ordering on vesicles with many folds
    # Sort particles by angle
    particles_angles = np.arctan2(particles[:, 1] - np.mean(particles[:, 1]), 
                                  particles[:, 0] - np.mean(particles[:, 0]))
    particles = particles[np.argsort(particles_angles), :]
    
    particles_downsampled = [particles[0]]
    for particle in particles:
        if np.linalg.norm(particle - particles_downsampled[-1]) * psize >= dist:
            particles_downsampled.append(particle)
    return particles_downsampled

def sign(x):
    # Return the sign of x, or 0 for 0
    if x == 0:
        return 0
    return int(abs(x) / x)

def is_pixel_in_square(corners, row, col):
    # Precondition: corners given in traversible order (clockwise or counterclockwise)
    # Compute determinant of edge to point transformation: positive if clockwise, negative otherwise
    D1 = (corners[1][0] - corners[0][0]) * (col - corners[0][1]) - (corners[1][1] - corners[0][1]) * (row - corners[0][0])
    D2 = (corners[2][0] - corners[1][0]) * (col - corners[1][1]) - (corners[2][1] - corners[1][1]) * (row - corners[1][0])
    D3 = (corners[3][0] - corners[2][0]) * (col - corners[2][1]) - (corners[3][1] - corners[2][1]) * (row - corners[2][0])
    D4 = (corners[0][0] - corners[3][0]) * (col - corners[3][1]) - (corners[0][1] - corners[3][1]) * (row - corners[3][0])
    # If the point is on different sides of both pairs of opposite lines, it is within the square
    # Negative sign reverses orientation of opposite line, use not equal to permit points on one line (D=0) 
    return (sign(D1) != sign(-D3) and sign(D2) != sign(-D4))

def pixels_in_square(p1, p2):
    # Find the pixels in a square whose upper corners are p1, p2
    # Calculate corners of the square
    d1 = p1[0] - p2[0]
    d2 = p1[1] - p2[1]
    if d2 > 0:
        s1, s2 = d2, -d1
    else: # d2 < 0
        s1, s2 = -d2, d1
    corners = (p1, p2, (p2[0] + s1, p2[1] + s2), (p1[0] + s1, p1[1] + s2))
    # Check each pixel in the bounding box of the square.
    in_square = set()
    for row in range(min(c[0] for c in corners), max(c[0] for c in corners)):
        for col in range(min(c[1] for c in corners), max(c[1] for c in corners)):
            if is_pixel_in_square(corners, row, col):
                in_square.add((row, col))
    return in_square

def pixels_in_rectangle(p1, p2):
    # Find the pixels in the 4:1 rectangle centered around the two membrane points given.
    # Calculate rectangle as union of four 1:1 squares shifted from the given points.
    in_rectangle = set()
    # Calculate offset of one square
    d1 = p1[0] - p2[0]
    d2 = p1[1] - p2[1]
    if d2 > 0:
        s1, s2 = d2, -d1
    else: # d2 < 0
        s1, s2 = -d2, d1
    for i in range(-2, 2):
        in_rectangle.update(pixels_in_square((p1[0] + i * s1, p1[1] + i * s2), (p2[0] + i * s1, p2[1] + i * s2)))
    return in_rectangle

def proj_dist(p, d):
    # Compute length of projection of p onto d
    return (p[0] * d[0] + p[1] * d[1]) / (sqrt(d[0] ** 2 + d[1] ** 2))

def bin_rectangle(img, p1, p2, rectangle, psize, hist_offset):
    # Bin intensities of points in the rectangle based on their distance from p1 and p2
    # Calculate vector pointing out of the vesicle, given points proceed clockwise
    # TODO: This is the slowest component of the code. Start time optimization here
    d1 = - (p2[1] - p1[1])
    d2 = (p2[0] - p1[0])
    bins = {i: [] for i in range(-hist_offset - 5, hist_offset + 5)}
    for p in rectangle:
        if 0 <= p[1] < img.shape[0] and 0 <= p[0] < img.shape[1]:
            p_dist = int(proj_dist((p[0] - p1[0], p[1] - p1[1]), (d1, d2)) * psize)
            if abs(p_dist) < hist_offset + 3:
                bins[p_dist].append(img[p[1]][p[0]])
    return bins

def find_bilayers(intensities, offset):
    # Identify the membrane bilayers as a pair of positive peaks surrounding a negative peak with 25 to 45 A of separation between them
    intensities_range = np.max(intensities) - np.min(intensities)
    pos_peaks = find_peaks(intensities, prominence=0.1 * intensities_range)[0]
    neg_peaks = find_peaks([-1 * intensity for intensity in intensities], prominence=0.1 * intensities_range)[0]
    candidates = []
    if len(neg_peaks) < 2 or len(pos_peaks) < 1: # Insufficient peaks for bilayer
        return candidates
    pos_considering = 0 # Index of positive peak under consideration
    while pos_peaks[pos_considering] < neg_peaks[0]: # Update to first positive peak after at least one negative peak
        pos_considering += 1
        if pos_considering == len(pos_peaks): # No positive peak between two negative peaks
            return candidates
    neg_considering = 0 # Index of negative peak before pos peak considered
    while pos_considering < len(pos_peaks):
        # Update closest negative peak 
        while neg_considering + 1 < len(neg_peaks) and neg_peaks[neg_considering + 1] < pos_peaks[pos_considering]:
            neg_considering += 1
        if neg_considering == len(neg_peaks) - 1: # No subsequent negative peak
            return candidates
        if 25 <= (neg_peaks[neg_considering + 1] - neg_peaks[neg_considering]) <= 45:
            candidates.append((neg_peaks[neg_considering] - offset, pos_peaks[pos_considering] - offset, neg_peaks[neg_considering + 1] - offset))
        pos_considering += 1
    return candidates

def update_pick(p1, p2, bilayer, psize):
    # Generate an updated membrane pick by shifting the midpoint of p1 and p2 by the distances in bilayer corresponding to inner membrane, intermembrane space, and outer membrane
    # Calculate midpoint from p1 to p2
    m1 = (p1[0] + p2[0]) / 2
    m2 = (p1[1] + p2[1]) / 2
    # Calculate vector pointing out of the vesicle, given points proceed clockwise
    d1 = - (p2[1] - p1[1])
    d2 = (p2[0] - p1[0])
    # Normalize vector out of vesicle (now 1 pixel = 0.819A)
    d_norm = sqrt(d1 ** 2 + d2 ** 2)
    d1 = d1 / d_norm
    d2 = d2 / d_norm
    return ((round(m1 + bilayer[0] * d1 / psize), round(m2 + bilayer[0] * d2 / psize)), 
            (round(m1 + bilayer[1] * d1 / psize), round(m2 + bilayer[1] * d2 / psize)), 
            (round(m1 + bilayer[2] * d1 / psize), round(m2 + bilayer[2] * d2 / psize)))

def bilayer_intensity(intensities, bilayer, offset):
    # Return the average intensity of a pair of bilayer positions
    return (intensities[bilayer[0] + offset] + intensities[bilayer[2] + offset])

def clean_edges(edges, cutoff, psize):
    # Clean a proposed set of membrane positions by removing positions which are further than the given cutoff distance in A from the line between their neighbours
    updated_edges = []
    for edge in edges:
        if len(edge) < 3:
            updated_edges.append([])
            continue
        # Uses inner membrane (im)
        im_edge = np.array([bilayer[0] for bilayer in edge] + [edge[0][0], edge[1][0]])
        im_edge_vec1 = im_edge[1:-1] - im_edge[:-2] # Vector from neighbour to point
        im_edge_vec2 = im_edge[2:] - im_edge[:-2] # Vector from neighbor to neighbour
        # Distance of self from line between neighbours
        im_deviance = np.array([psize * proj_dist(im_edge_vec1[i], (im_edge_vec2[i][1], -im_edge_vec2[i][0])) for i in range(len(im_edge_vec1))])
        updated_edges.append([edge[i] for i in range(len(edge)) if i not in (np.where(np.abs(im_deviance) > cutoff)[0] + 1)])
    return updated_edges

# Parse command line arguments
parser = ArgumentParser(
    prog="repick_membrane.py",
    description="Re-pick membrane coordinates in micrographs"
)
parser.add_argument(
    "parameters", 
    type=str,
    help="Path to .ini file containing parameters for vesicle picking"
)
parser.add_argument(
    "--input_dir",
    type=str,
    default=".",
    help="Directory containing np arrays of edge coordinates"
)
parser.add_argument(
    "--contour_spacing",
    type=float,
    default=50,
    help="Separation in A between sample points on vesicle contours"
)
parser.add_argument(
    "--hist_endpoints",
    type=int,
    default=45,
    help="Distance in A for histogram to extend from the membrane"
)
parser.add_argument(
    "--first_clean_cutoff",
    type=float,
    default=20.0,
    help="Maximum distance in A to permit picks to deviate from their neighbours in first cleaning step"
)
parser.add_argument(
    "--second_clean_cutoff",
    type=float,
    default=50.0,
    help="Maximum distance in A to permit picks to deviate from their neighbours in second cleaning step"
)
parser.add_argument(
    "--spline_density",
    type=int,
    default=20000,
    help="Number of points to pick from each spline fit to a vesicle"
)
parser.add_argument(
    "--picks_dir",
    type=str,
    default=None,
    help="Path to save image of initial membrane picks"
)
parser.add_argument(
    "--cleaned_picks_dir",
    type=str,
    default=None,
    help="Path to save image of only cleaned membrane picks"
)
parser.add_argument(
    "--support_separation",
    type=float,
    default=200,
    help="Maximum distance in A between adjacent points in a spline's supporting arc. If -1, full spline returned"
)
parser.add_argument(
    "--spline_dir",
    type=str,
    default=None,
    help="Path to save np array of final membrane spline coordinates"
)


args = parser.parse_args()
parameters_filepath = args.parameters
parameters = helpers.read_config(parameters_filepath)
input_dir = args.input_dir
contour_spacing = args.contour_spacing
hist_offset = args.hist_endpoints
first_cutoff = args.first_clean_cutoff
second_cutoff = args.second_clean_cutoff
spline_density = args.spline_density
picks_dir = args.picks_dir
cleaned_picks_dir = args.cleaned_picks_dir
support_separation = args.support_separation
spline_dir = args.spline_dir

# Load in commonly used parameters
downsample = int(parameters.get('general', 'downsample'))
psize = float(parameters.get('general', 'psize'))

# Initialize a cryosparc session, open a project, and pull in the micrographs
cs = external_import.load_cryosparc(parameters.get('csparc_input', 'login'))
project = cs.find_project(parameters.get("csparc_input", "PID"))
micrographs = external_import.micrographs_from_csparc(
    cs=cs,
    project_id=parameters.get('csparc_input', 'PID'),
    job_id=parameters.get('csparc_input', 'JID'),
    job_type=parameters.get('csparc_input', 'type')
)

# Initialize the final Dataset
vesicle_picks = Dataset()
vesicle_picks.add_fields(
    ['location/micrograph_uid',
     'location/exp_group_id',
     'location/micrograph_path',
     'location/micrograph_shape',
     'location/center_x_frac',
     'location/center_y_frac',
     'location/micrograph_psize_A'],
    ["<u8", "<u4", "str", "<u4", "<f4", "<f4", "<f4"])

input_dir_files = [entry for entry in os.scandir(input_dir) if entry.is_file()]

# Loop over all micrographs
for micrograph in tqdm(micrographs[0:]):

    # Extract the micrograph UID
    uid = micrograph['uid']

    # Construct the filename of the file to import
    masks_filename = (
        f"{parameters.get('input', 'directory')}{uid}_vesicles_filtered.pkl"
    )
    
    micrograph_files = sorted([entry for entry in input_dir_files if entry.name.startswith(str(uid)) and entry.name.endswith("intermembrane.npy")], key=lambda entry: entry.name)
    # If there is no file of edge coordinates in the input directory
    # then go to the next micrograph
    if len(micrograph_files) == 0:
        print(f"Missing files for {uid}", file=sys.stderr)
        continue
    # Load the contours from that all files for this micrograph
    masks_edges = [np.load(entry.path) for entry in micrograph_files]
    
    # Extract the image
    header, image_fullres = project.download_mrc(
        micrograph["micrograph_blob/path"]
    )
    image_fullres = image_fullres[0]
    image_blurred = cv2.GaussianBlur(image_fullres, (29, 29), 5, 5)

    # Downsample vesicle edges
    masks_edges_downsampled = [downsample_contour(edges, contour_spacing, psize)
                               for edges in masks_edges]

    # Update vesicle edges to detected membrane                    
    all_updated_edges = []
    # Iterate over vesicle masks
    for edges in masks_edges_downsampled:
        updated_edges = []
        # Iterate over pairs of adjacent sample points within the vesicle mask
        for i in range(0, len(edges) - 1):
            # For runtime reasons, skip very far particle pairs
            if np.linalg.norm(edges[i] - edges[i + 1]) * psize > contour_spacing * 1.5:
                continue
            t = time.time()
            edge_rectangle = pixels_in_rectangle(edges[i], edges[i + 1])
            t_pixels_in_rectangle += (time.time() - t)
            t = time.time()
            bins = bin_rectangle(image_blurred, edges[i], edges[i + 1], edge_rectangle, psize, hist_offset)
            t_bin_rectangle += (time.time() - t)
            intensities = []
            for j in range(-hist_offset, hist_offset + 1):
                if len(bins[j]) > 0:
                    intensities.append(np.mean(bins[j]))
                else:
                    intensities.append(0.0)
            t = time.time()
            bilayers = find_bilayers(intensities, hist_offset)
            t_find_bilayers += (time.time() - t)
            # If only one bilayer candidate is detected, record it
            if len(bilayers) == 1:
                updated_edges.append(update_pick(edges[i], edges[i + 1], bilayers[0], psize))
            # If multiple are detected but the candidate with largest intensity is significantly larger than the second largest, return that candidate
            elif len(bilayers) > 1:
                bilayers = sorted(bilayers, key=lambda bilayer: bilayer_intensity(intensities, bilayer, hist_offset))
                intensities_range = np.max(intensities) - np.min(intensities)
                if bilayer_intensity(intensities, bilayers[1], hist_offset) - bilayer_intensity(intensities, bilayers[0], hist_offset) > 0.25 * intensities_range:
                    updated_edges.append(update_pick(edges[i], edges[i + 1], bilayers[0], psize))
        all_updated_edges.append(updated_edges)
        
    # Save particle pick images
    if picks_dir is not None:
        picks_dir = Path(picks_dir)
        image_out = np.copy(image_blurred)
        image_out_max = np.max(image_out)
        for edge in all_updated_edges:
            for particle_trio in edge:
                for particle in particle_trio:
                    for i in range(-4, 5):
                        for j in range(-4, 5):
                            if 0 <= particle[1] + i < image_out.shape[0] and 0 <= particle[0] + j < image_out.shape[1]:
                                image_out[particle[1] + i, particle[0] + j] = image_out_max
        plt.imsave(picks_dir / f"{uid}.png", image_out, cmap="gray")
        
    # Clean the refined vesicle edge picks to remove outliers
    all_updated_edges_cleaned = clean_edges(all_updated_edges, first_cutoff, psize)
    # Repeat with a second cutoff until all points fit, to catch remaining outliers
    while any(len(all_updated_edges[i]) != len(all_updated_edges_cleaned[i]) for i in range(len(all_updated_edges))):
        all_updated_edges = all_updated_edges_cleaned
        all_updated_edges_cleaned = clean_edges(all_updated_edges, second_cutoff, psize)
        
    # Save cleaned particle pick images
    if cleaned_picks_dir is not None:
        cleaned_picks_dir = Path(cleaned_picks_dir)
        image_out = np.copy(image_blurred)
        image_out_max = np.max(image_out)
        for edge in all_updated_edges_cleaned:
            for particle_trio in edge:
                for particle in particle_trio:
                    for i in range(-4, 5):
                        for j in range(-4, 5):
                            if 0 <= particle[1] + i < image_out.shape[0] and 0 <= particle[0] + j < image_out.shape[1]:
                                image_out[particle[1] + i, particle[0] + j] = image_out_max
        plt.imsave(cleaned_picks_dir / f"{uid}_cleaned.png", image_out, cmap="gray")
        
    # Generate splines through the updated points
    splines = []
    t = time.time()
    for edge in all_updated_edges_cleaned:
        if len(edge) > 3:
            # Determine regions with points (support) to include spline
            if support_separation != -1:
                # Determine supports based on outer membrane
                p_x = np.array([points[2][0] for points in edge] + [edge[0][2][0]])
                p_y = np.array([points[2][1] for points in edge] + [edge[0][2][1]])
                supports = []
                curr_start = 0
                curr_end = 0
                for j in range(1, len(p_x)):
                    if psize * ((p_x[j] - p_x[curr_end]) ** 2 + (p_y[j] - p_y[curr_end]) ** 2) < (support_separation) ** 2:
                        curr_end = j
                    else:
                        if curr_start != curr_end:
                            supports.append((curr_start, curr_end))
                        curr_start = j
                        curr_end = j
                if curr_start != curr_end:
                    supports.append((curr_start, curr_end))
            
            for i in range(3):
                p_x = np.array([points[i][0] for points in edge] + [edge[0][i][0]])
                p_y = np.array([points[i][1] for points in edge] + [edge[0][i][1]])
                
                # Skip if data points are insufficient
                if len(p_x) < 4 or len(p_y) < 4:
                    print(f"Skipping spline generation for edge with insufficient data points: {len(p_x)}")
                    continue
                if np.any(np.isnan(p_x)) or np.any(np.isnan(p_y)):
                    print(f"Skipping spline generation due to NaN values in p_x or p_y")
                    continue
                try:
                    tck, u = splprep([p_x, p_y], k=3)
                    spline = splev(np.linspace(0, 1.0, spline_density), tck)
                except ValueError as e:
                    print(f"Skipping spline generation for edge due to error: {e}")
                    continue

                if support_separation != -1:
                    spline_supported = []
                    for arc in supports:
                        idx1 = np.argmin((spline[:, 0] - p_x[arc[0]]) ** 2 + (spline[:, 1] - p_y[arc[0]]) ** 2)
                        idx2 = np.argmin((spline[:, 0] - p_x[arc[1]]) ** 2 + (spline[:, 1] - p_y[arc[1]]) ** 2)
                        # PRECONDITION: Points returned counterclockwise, so should have idx1 < idx2
                        if idx1 == spline.shape[0] - 1:
                            idx1 = 0
                        if idx2 == 0:
                            idx2 = spline.shape[0] - 1
                        for j in range(min(idx1, idx2), max(idx1, idx2) + 1):
                            spline_supported.append(spline[j])
                    spline_supported = np.array(spline_supported)
                else: # Include full spline
                    spline_supported = spline
                splines.append(spline_supported)
    t_fit_splines += (time.time() - t)

    # Save final pick locations as arrays
    if spline_dir is not None:
        spline_dir = Path(spline_dir)
        for i in range(len(splines)//3):
            np.save(spline_dir / f"{uid}_vesicle_{i}_inner.npy", splines[3 * i + 0])
            np.save(spline_dir / f"{uid}_vesicle_{i}_intermembrane.npy", splines[3 * i + 1])
            np.save(spline_dir / f"{uid}_vesicle_{i}_outer.npy", splines[3 * i + 2])

    # Record final pick indices
    pick_indices = (np.array([particle[1] for spline in splines for particle in spline]),
                    np.array([particle[0] for spline in splines for particle in spline]))
    pick_dataset = external_export.construct_csparc_dataset(micrograph, pick_indices)
    vesicle_picks = vesicle_picks.append(pick_dataset)
    

# Push vesicle_picks to cryosparc
# Initialize project and job
project = cs.find_project(parameters.get('csparc_input', 'PID'))
job = project.create_external_job(
    parameters.get('csparc_input', 'WID'),
    title="Vesicle Picks"
)

# Tell the job what kind of output to expect
job.add_output("particle", "vesicle_picks", slots=["location"])

# Start the job, push the output to cryosparc, stop the job
job.start()
job.save_output("vesicle_picks", vesicle_picks)
job.stop()

print(f"Time in pixels_in_rectangle: {t_pixels_in_rectangle}s")
print(f"Time in bin_rectangle: {t_bin_rectangle}s")
print(f"Time in find_bilayers: {t_find_bilayers}s")
print(f"Time fiting splines: {t_fit_splines}s")

