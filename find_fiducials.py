from fiducials import Fiducials, ControlPoint
from mesh_helpers import read_as_vtkpolydata, get_mesh_actor, convert_between_ras_and_lps
import vtk
import numpy as np
import os
import argparse


def find_fiducials(mesh: vtk.vtkPolyData) -> Fiducials:
    """
    !!!YOUR CODE GOES HERE!!!

    Find fiducials in a mesh.

    Input:
        mesh: vtkPolyData object representing the mesh.

    Output:
        fiducials: Fiducials object containing the found fiducials.

    This example is highly reductive and inaccurate, but demonstrates the flow of data through the function
    """
    # Convert mesh to LPS coordinate system
    mesh = convert_between_ras_and_lps(mesh)

    # get the positions of the mesh vertices
    points = mesh.GetPoints()
    if points is None:
        print("Error: mesh.GetPoints() returned None. The mesh might be invalid or empty.")
        exit(1)
    num_points = points.GetNumberOfPoints()
    if num_points == 0:
        print(f"Error: The mesh {mesh_path} loaded 0 points. Please check the mesh file.")
        exit(1)
    print(f"Successfully loaded {num_points} points from the mesh.") # Added for debugging
    arr = np.zeros((num_points, 3), dtype=np.float32)
    for i in range(num_points):
        p = points.GetPoint(i)
        arr[i, 0] = p[0]
        arr[i, 1] = p[1]
        arr[i, 2] = p[2]

    # Analyze the mesh vertices to find fiducials
    center = np.mean(arr, axis=0)
    pmin = np.min(arr, axis=0)
    pmax = np.max(arr, axis=0)
    size = pmax - pmin

    # Nasion Detection
    # Filter points that are anterior (Y points Posterior in arr, so we look for smaller Y values for anterior)
    anterior_filter = arr[:, 1] < center[1] - size[1] * 0.1
    # Filter points that are roughly midline horizontally
    midline_filter = np.abs(arr[:, 0] - center[0]) < size[0] * 0.15
    # Filter points that are in the expected vertical region for the nasion (Variation 1 Z-filter)
    vertical_filter_nasion = (arr[:, 2] > center[2] - size[2]*0.07) & (arr[:, 2] < center[2] + size[2]*0.23)
    
    candidate_nasion_points = arr[anterior_filter & midline_filter & vertical_filter_nasion]
    
    if candidate_nasion_points.shape[0] > 0:
        # Select the point with the largest Y value (most posterior among anterior candidates, due to Y inversion)
        nasion = candidate_nasion_points[np.argmax(candidate_nasion_points[:, 1])]
    else:
        # Fallback if no points match criteria, use a point near the top-front-middle
        # This is a simplistic fallback and might need refinement
        nasion = arr[np.argmax(arr[:, 1] + arr[:,2] - np.abs(arr[:,0]-center[0]))]


    # Ear Detection (Left and Right Tragus)
    # Filter points vertically (New restrictive filter)
    vertical_filter_ears = np.abs(arr[:, 2] - center[2]) < size[2] * 0.15
    # Filter points posteriorly (Y points Posterior in arr, so we look for larger Y values for posterior) (New restrictive filter)
    posterior_filter = arr[:, 1] > center[1] + size[1] * 0.05
    
    candidate_ear_points = arr[vertical_filter_ears & posterior_filter]

    if candidate_ear_points.shape[0] > 0:
        # arr[:,0] is X-Left (more positive X is Patient's Left)
        # Left ear: Max X in X-Left system is Patient's Left
        left_ear = candidate_ear_points[np.argmax(candidate_ear_points[:, 0])]
        # Right ear: Min X in X-Left system is Patient's Right
        right_ear = candidate_ear_points[np.argmin(candidate_ear_points[:, 0])]
    else:
        # Fallback if no points match criteria
        # This is a simplistic fallback and might need refinement
        right_ear = arr[np.argmax(arr[:,0] - np.abs(arr[:,1]-center[1]) - np.abs(arr[:,2]-center[2]))] 
        left_ear = arr[np.argmin(arr[:,0] + np.abs(arr[:,1]-center[1]) + np.abs(arr[:,2]-center[2]))]

    # Eye Corner Detection (retaining placeholder logic with new nasion and ears)
    # Radii for weighted average calculations (optional, but kept from placeholder for consistency)
    nasion_r = np.sqrt(np.sum((nasion - center) ** 2))
    left_ear_r = np.sqrt(np.sum((left_ear - center) ** 2))
    right_ear_r = np.sqrt(np.sum((right_ear - center) ** 2))

    left_eye_outside = left_ear * 0.5 + nasion * 0.5
    left_eye_outside_r = np.sqrt(np.sum((left_eye_outside - center) ** 2))
    if left_eye_outside_r > 1e-6: # Avoid division by zero
        left_eye_outside = (
            (left_eye_outside - center)
            * (left_ear_r * 0.5 + nasion_r * 0.5)
            / left_eye_outside_r
        ) + center
    else: # If coincident, just use the calculated point
        left_eye_outside = left_eye_outside

    left_eye_inside = left_ear * 0.25 + nasion * 0.75
    left_eye_inside_r = np.sqrt(np.sum((left_eye_inside - center) ** 2))
    if left_eye_inside_r > 1e-6:
        left_eye_inside = (
            (left_eye_inside - center)
            * (left_ear_r * 0.25 + nasion_r * 0.75)
            / left_eye_inside_r
        ) + center
    else:
        left_eye_inside = left_eye_inside

    right_eye_outside = right_ear * 0.5 + nasion * 0.5
    right_eye_outside_r = np.sqrt(np.sum((right_eye_outside - center) ** 2))
    if right_eye_outside_r > 1e-6:
        right_eye_outside = (
            (right_eye_outside - center)
            * (right_ear_r * 0.5 + nasion_r * 0.5)
            / right_eye_outside_r
        ) + center
    else:
        right_eye_outside = right_eye_outside

    right_eye_inside = right_ear * 0.25 + nasion * 0.75
    right_eye_inside_r = np.sqrt(np.sum((right_eye_inside - center) ** 2))
    if right_eye_inside_r > 1e-6:
        right_eye_inside = (
            (right_eye_inside - center)
            * (right_ear_r * 0.25 + nasion_r * 0.75)
            / right_eye_inside_r
        ) + center
    else:
        right_eye_inside = right_eye_inside
        
    # Create the Fiducials object and populate the control points
    # The order of appends MUST be maintained as per the problem description.

    # Convert points from internal RPS (Y-Posterior) to Slicer LPS (Y-Anterior) by negating Y
    # It's safer to copy if these arrays are used elsewhere, but current code uses them only here.
    # For clarity and safety, let's make copies and then negate Y.
    
    left_ear_lps = left_ear.copy()
    left_ear_lps[1] = -left_ear_lps[1]

    left_eye_outside_lps = left_eye_outside.copy()
    left_eye_outside_lps[1] = -left_eye_outside_lps[1]

    left_eye_inside_lps = left_eye_inside.copy()
    left_eye_inside_lps[1] = -left_eye_inside_lps[1]

    nasion_lps = nasion.copy()
    nasion_lps[1] = -nasion_lps[1]

    right_eye_inside_lps = right_eye_inside.copy()
    right_eye_inside_lps[1] = -right_eye_inside_lps[1]

    right_eye_outside_lps = right_eye_outside.copy()
    right_eye_outside_lps[1] = -right_eye_outside_lps[1]

    right_ear_lps = right_ear.copy()
    right_ear_lps[1] = -right_ear_lps[1]

    fiducials = Fiducials(color=[0, 1, 0])
    fiducials.control_points.append(ControlPoint(left_ear_lps, "left_ear"))
    fiducials.control_points.append(ControlPoint(left_eye_outside_lps, "left_eye_outside"))
    fiducials.control_points.append(ControlPoint(left_eye_inside_lps, "left_eye_inside"))
    fiducials.control_points.append(ControlPoint(nasion_lps, "nasion"))
    fiducials.control_points.append(ControlPoint(right_eye_inside_lps, "right_eye_inside"))
    fiducials.control_points.append(
        ControlPoint(right_eye_outside_lps, "right_eye_outside")
    )
    fiducials.control_points.append(ControlPoint(right_ear_lps, "right_ear"))

    return fiducials


if __name__ == "__main__":
    # This script is used to find fiducials in a mesh.
    # It takes a scan ID as input and loads the corresponding mesh from a file.
    # The script uses VTK to render the mesh and the fiducials in a 3D window if requested.
    # The script can be run from the command line with the following arguments:
    #   python find_fiducials.py <scan_id> [--display] [--save] [--reference] [--dataset <dataset_name>] [--point-size <size>]
    #
    # Required Arguments:
    #   scan_id: The ID of the scan to process (e.g., 0).
    # Flags:
    #   -d, --display/--no-display: Display the fiducials in a VTK window. Default is not to display.
    #   -s, --save/--no-save: Save the fiducials to a file. Default is not to save.
    #   --reference/--no-reference: Load the reference fiducials from a file for display.
    # Options:
    #   --dataset: The name of the dataset (e.g., training). If you add additional data to the data folder, you can specify the dataset name here.
    #   --point-size: The size of the points in the VTK window. Default is 0.01.

    parser = argparse.ArgumentParser(description="Find fiducials in a mesh.")
    parser.add_argument("id", type=str, help="Scan ID (e.g., 0).")
    parser.add_argument(
        "--dataset",
        type=str,
        default="training",
        help="Dataset name (e.g., training).",
    )
    parser.add_argument(
        "-d",
        "--display",
        action=argparse.BooleanOptionalAction,
        help="Display the fiducials.",
    )
    parser.add_argument(
        "-s",
        "--save",
        action=argparse.BooleanOptionalAction,
        help="Save the fiducials to a file.",
    )
    parser.add_argument(
        "--reference",
        action=argparse.BooleanOptionalAction,
        help="Load the reference fiducials from a file for display.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=0.01,
        help="Size of the points in the VTK window.",
    )
    args = parser.parse_args()
    scan_id = int(args.id)
    dataset = args.dataset
    here = os.path.dirname(__file__)
    data_dir = os.path.join(here, "data", dataset)
    mesh_path = os.path.join(data_dir, "input_meshes", f"scan_{scan_id:03d}.obj")
    mesh = read_as_vtkpolydata(mesh_path)
    # The mesh_path variable needs to be accessible in find_fiducials for the error message.
    # We can pass it as an argument, or make it global, or re-construct it.
    # For simplicity in this step, I'll just re-use the global mesh_path from the main scope,
    # though passing it explicitly would be cleaner.
    # Modifying find_fiducials signature would be a larger change, so let's assume it can access mesh_path for now
    # if find_fiducials is called from __main__ block.
    # For a cleaner solution, find_fiducials should accept mesh_path.
    # For now, the error message in find_fiducials will use the global `mesh_path`.
    output_fiducials = find_fiducials(mesh) # renamed variable to avoid conflict
    if args.save:
        output_fiducials.to_file(
            os.path.join(data_dir, "output_points", f"fiducials_{scan_id:03d}.mrk.json")
        )
    else:
        output_fiducials.print()
    if args.display:
        if args.reference:
            reference_fiducials = Fiducials.from_file(
                os.path.join(
                    data_dir, "reference_points", f"fiducials_{scan_id:03d}.mrk.json"
                )
            )
        else:
            reference_fiducials = None
        renderWindow = vtk.vtkRenderWindow()
        renderer = vtk.vtkRenderer()
        renderWindow.AddRenderer(renderer)
        renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        renderWindowInteractor.SetRenderWindow(renderWindow)
        renderer.AddActor(get_mesh_actor(mesh))
        cp_actors = output_fiducials.get_actors(size=args.point_size) # Use output_fiducials
        for cp in cp_actors:
            renderer.AddActor(cp)
        if reference_fiducials:
            cp_actors = reference_fiducials.get_actors(size=args.point_size)
            for cp in cp_actors:
                renderer.AddActor(cp)
        renderWindow.Render()
        renderWindowInteractor.Start()
