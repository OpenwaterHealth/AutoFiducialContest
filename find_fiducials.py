from fiducials import Fiducials, ControlPoint
from mesh_helpers import read_as_vtkpolydata, get_mesh_actor
import vtk
import numpy as np
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scipy import stats
import argparse
import warnings


def find_fiducials(mesh: vtk.vtkPolyData) -> Fiducials:
    """
    Find fiducials in a mesh using the full pipeline (MediaPipe + VTK rendering + 3D picking).

    NOTE: For full functionality (textured rendering), the mesh must be loaded from a .obj file
    and the corresponding .mtl and texture (.png) files must be present in the same directory.
    The function attempts to infer the file path by searching for a matching mesh in data/training/input_meshes.
    If this fails, the pipeline will not work as intended and a clear error will be raised.

    Input:
        mesh: vtkPolyData object representing the mesh (should be loaded from .obj file).
    Output:
        fiducials: Fiducials object containing the found fiducials.
    """
    # --- Parameters ---
    num_views = 16
    image_size = 800
    sweep_size = 5
    filter_thresh = 1.0
    model_path = os.path.join("mediapipe", "face_landmarker.task")
    
    """
    MediaPipe landmark index and fiducial ID mapping
    First number: MediaPipe face landmark index for detection
    Second number: Fiducial ID (999 is placeholder, can be changed if needed)
    NOTE: 999 used due to discrepancies between ground truth and template JSON IDs
    """
    landmark_map = {
        "left_ear": (234, 999),
        "left_eye_outside": (130, 999),
        "left_eye_inside": (133, 999),
        "nasion": (168, 999),
        "right_eye_inside": (362, 999),
        "right_eye_outside": (359, 999),
        "right_ear": (454, 999),
    }

    # --- Helper: Guess mesh file path by matching points ---
    def guess_mesh_path_from_points(mesh):
        data_dir = os.path.join("data", "training", "input_meshes")
        if not os.path.isdir(data_dir):
            return None
        mesh_points = mesh.GetPoints()
        num_points = mesh_points.GetNumberOfPoints()
        mesh_arr = np.array([mesh_points.GetPoint(i) for i in range(num_points)])
        for fname in os.listdir(data_dir):
            if not fname.endswith(".obj"):
                continue
            candidate_path = os.path.join(data_dir, fname)
            try:
                candidate_mesh = read_as_vtkpolydata(candidate_path)
                c_points = candidate_mesh.GetPoints()
                if c_points.GetNumberOfPoints() != num_points:
                    continue
                c_arr = np.array([c_points.GetPoint(i) for i in range(num_points)])
                # Use a quick hash or mean check for speed
                if np.allclose(np.mean(mesh_arr, axis=0), np.mean(c_arr, axis=0), atol=1e-5):
                    if np.allclose(mesh_arr, c_arr, atol=1e-5):
                        return candidate_path
            except Exception:
                continue
        return None

    # --- Infer mesh file path (required for .mtl and texture) ---
    mesh_path = guess_mesh_path_from_points(mesh)
    if mesh_path is None or not os.path.isfile(mesh_path):
        raise RuntimeError("Could not infer mesh file path from input mesh. Please ensure the mesh is loaded from a .obj file in data/training/input_meshes and is unmodified. Textured rendering and picking require the original file. If you are running on a different dataset, update the search path in find_fiducials.")
    mesh_base = os.path.splitext(os.path.basename(mesh_path))[0]
    mtl_path = os.path.join(os.path.dirname(mesh_path), mesh_base + ".mtl")
    texture_path = os.path.join(os.path.dirname(mesh_path))

    # --- Setup MediaPipe detector ---
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
        min_face_detection_confidence=0.7,
        min_face_presence_confidence=0.7,
        min_tracking_confidence=0.7
    )
    detector = vision.FaceLandmarker.create_from_options(options)
    mp_face_detection = mp.solutions.face_detection
    testdetector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.95)

    # --- Find viable Z rotations ---
    def find_viable_z_rots(mesh_path, detector, stepZ=sweep_size, verbose=False, image_size=image_size):
        window = vtk.vtkRenderWindow()
        window.SetOffScreenRendering(1)
        window.SetSize(image_size, image_size)
        renderer = vtk.vtkRenderer()
        window.AddRenderer(renderer)
        lights = [
            {'pos': (0, 0, 1), 'color': (1, 1, 1)},
            {'pos': (0, 1, 0), 'color': (0.5, 0.5, 0.5)},
            {'pos': (1, 0, 0), 'color': (0.3, 0.3, 0.3)},
            {'pos': (-1, 0, 0), 'color': (0.3, 0.3, 0.3)}
        ]
        for light_cfg in lights:
            light = vtk.vtkLight()
            light.SetPosition(*light_cfg['pos'])
            light.SetFocalPoint(0, 0, 0)
            light.SetColor(*light_cfg['color'])
            light.SetIntensity(1.2)
            renderer.AddLight(light)
        importer = vtk.vtkOBJImporter()
        importer.SetFileName(mesh_path)
        importer.SetFileNameMTL(mtl_path)
        importer.SetTexturePath(texture_path)
        importer.SetRenderWindow(window)
        importer.Update()
        renderer.SetBackground(1, 1, 1)
        prevDetect = False
        prevprevDetect = False
        minZrot = 0
        maxZrot = 0
        currZ = 0
        numsteps = round(360/stepZ)
        for z in range(numsteps):
            renderer.ResetCamera()
            renderer.GetActiveCamera().Zoom(1.0)
            renderer.GetActiveCamera().OrthogonalizeViewUp()
            renderer.GetActiveCamera().Azimuth(stepZ)
            currZ = currZ + stepZ
            renderer.ResetCameraClippingRange()
            window.Render()
            windowToImageFilter = vtk.vtkWindowToImageFilter()
            windowToImageFilter.SetInput(window)
            windowToImageFilter.Update()
            vtk_image = windowToImageFilter.GetOutput()
            scalars = vtk_image.GetPointData().GetScalars()
            image = np.frombuffer(scalars, dtype=np.uint8)
            image = image.reshape(image_size, image_size, 3)
            image = np.flipud(image)
            # Patch: ensure correct format for MediaPipe
            if image.shape[-1] > 3:
                image = image[..., :3]
            image = np.ascontiguousarray(image, dtype=np.uint8)
            rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            detection_result = detector.detect(rgb_frame)
            currDetect = bool(detection_result.face_landmarks)
            if prevprevDetect and (not prevDetect) and (not currDetect):
                maxZrot = currZ-2*stepZ
                break
            prevprevDetect = prevDetect
            prevDetect = currDetect
        prevDetect = False
        prevprevDetect = False
        currZ = maxZrot
        for z in range(numsteps):
            renderer.ResetCamera()
            renderer.GetActiveCamera().Zoom(1.0)
            renderer.GetActiveCamera().OrthogonalizeViewUp()
            renderer.GetActiveCamera().Azimuth(-stepZ)
            currZ = currZ - stepZ
            renderer.ResetCameraClippingRange()
            window.Render()
            windowToImageFilter = vtk.vtkWindowToImageFilter()
            windowToImageFilter.SetInput(window)
            windowToImageFilter.Update()
            vtk_image = windowToImageFilter.GetOutput()
            scalars = vtk_image.GetPointData().GetScalars()
            image = np.frombuffer(scalars, dtype=np.uint8)
            image = image.reshape(image_size, image_size, 3)
            image = np.flipud(image)
            # Patch: ensure correct format for MediaPipe
            if image.shape[-1] > 3:
                image = image[..., :3]
            image = np.ascontiguousarray(image, dtype=np.uint8)
            rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            detection_result = detector.detect(rgb_frame)
            currDetect = bool(detection_result.face_landmarks)
            if prevprevDetect and (not prevDetect) and (not currDetect):
                minZrot = currZ+2*stepZ
                break
            prevprevDetect = prevDetect
            prevDetect = currDetect
        return maxZrot, minZrot

    maxZrot, minZrot = find_viable_z_rots(mesh_path, detector, sweep_size, False, image_size)
    z_rotations = np.linspace(minZrot, maxZrot, num_views)

    # --- Render all views ---
    def render_images_all(mesh_path, z_rotations, image_size=image_size):
        images = []
        stepZ = z_rotations[1] - z_rotations[0]
        window = vtk.vtkRenderWindow()
        window.SetOffScreenRendering(1)
        window.SetSize(image_size, image_size)
        renderer = vtk.vtkRenderer()
        window.AddRenderer(renderer)
        lights = [
            {'pos': (0, 0, 1), 'color': (1, 1, 1)},
            {'pos': (0, 1, 0), 'color': (0.5, 0.5, 0.5)},
            {'pos': (1, 0, 0), 'color': (0.3, 0.3, 0.3)},
            {'pos': (-1, 0, 0), 'color': (0.3, 0.3, 0.3)}
        ]
        for light_cfg in lights:
            light = vtk.vtkLight()
            light.SetPosition(*light_cfg['pos'])
            light.SetFocalPoint(0, 0, 0)
            light.SetColor(*light_cfg['color'])
            light.SetIntensity(1.2)
            renderer.AddLight(light)
        importer = vtk.vtkOBJImporter()
        importer.SetFileName(mesh_path)
        importer.SetFileNameMTL(mtl_path)
        importer.SetTexturePath(texture_path)
        importer.SetRenderWindow(window)
        importer.Update()
        renderer.SetBackground(1, 1, 1)
        renderer.ResetCamera()
        renderer.GetActiveCamera().Zoom(1.0)
        renderer.GetActiveCamera().OrthogonalizeViewUp()
        renderer.GetActiveCamera().Azimuth(z_rotations[0] - stepZ)
        for z in z_rotations:
            renderer.ResetCamera()
            renderer.GetActiveCamera().Zoom(1.0)
            renderer.GetActiveCamera().OrthogonalizeViewUp()
            renderer.GetActiveCamera().Azimuth(stepZ)
            renderer.ResetCameraClippingRange()
            window.Render()
            windowToImageFilter = vtk.vtkWindowToImageFilter()
            windowToImageFilter.SetInput(window)
            windowToImageFilter.Update()
            vtk_image = windowToImageFilter.GetOutput()
            scalars = vtk_image.GetPointData().GetScalars()
            image = np.frombuffer(scalars, dtype=np.uint8)
            image = image.reshape(image_size, image_size, 3)
            image = np.flipud(image)
            # Patch: ensure correct format for MediaPipe
            if image.shape[-1] > 3:
                image = image[..., :3]
            image = np.ascontiguousarray(image, dtype=np.uint8)
            images.append(image)
        return np.stack(images, axis=0)

    images = render_images_all(mesh_path, z_rotations, image_size)

    # --- Extract fiducials ---
    fiducial_points = {label: [] for label in landmark_map}
    picker = vtk.vtkCellPicker()
    picker.SetTolerance(0.005)
    stepZ = z_rotations[1] - z_rotations[0]
    window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.SetSize(image_size, image_size)
    renderer = vtk.vtkRenderer()
    window.AddRenderer(renderer)
    importer = vtk.vtkOBJImporter()
    importer.SetFileName(mesh_path)
    importer.SetFileNameMTL(mtl_path)
    importer.SetTexturePath(texture_path)
    importer.SetRenderWindow(window)
    importer.Update()
    renderer.SetBackground(1, 1, 1)
    renderer.ResetCamera()
    renderer.GetActiveCamera().Zoom(1.0)
    renderer.GetActiveCamera().OrthogonalizeViewUp()
    renderer.GetActiveCamera().Azimuth(z_rotations[0] - stepZ)
    renderer.ResetCameraClippingRange()
    window.Render()
    currZ = z_rotations[0]
    middleZ = (z_rotations[-1] + z_rotations[0]) / 2
    for view_id in range(num_views):
        image = images[view_id]
        # Patch: ensure correct format for MediaPipe
        if image.shape[-1] > 3:
            image = image[..., :3]
        image = np.ascontiguousarray(image, dtype=np.uint8)
        rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        detection_result = detector.detect(rgb_frame)
        testdetector_results = testdetector.process(image)
        renderer.ResetCamera()
        renderer.GetActiveCamera().Zoom(1.0)
        renderer.GetActiveCamera().OrthogonalizeViewUp()
        renderer.GetActiveCamera().Azimuth(stepZ)
        renderer.ResetCameraClippingRange()
        window.Render()
        currZ = currZ + stepZ
        if not detection_result.face_landmarks:
            continue
        face_landmarks = detection_result.face_landmarks[0]
        h, w, _ = image.shape
        for label, (mp_idx, fid_id) in landmark_map.items():
            lm = face_landmarks[mp_idx]
            image_x, image_y = float(lm.x * w), float(lm.y * h)
            y_vtk = image_size - image_y
            # Special handling for tragion
            if label == "left_ear":
                if not testdetector_results.detections:
                    continue
                if currZ < middleZ - 15:
                    continue
                for detection in testdetector_results.detections:
                    Earpoint = mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.LEFT_EAR_TRAGION)
                image_x, image_y = float(Earpoint.x * w), float(Earpoint.y * h)
                y_vtk = image_size - image_y
            elif label == "right_ear":
                if not testdetector_results.detections:
                    continue
                if currZ > middleZ + 15:
                    continue
                for detection in testdetector_results.detections:
                    Earpoint = mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.RIGHT_EAR_TRAGION)
                image_x, image_y = float(Earpoint.x * w), float(Earpoint.y * h)
                y_vtk = image_size - image_y
            picker.Pick(image_x, y_vtk, 0, renderer)
            point_3d = picker.GetPickPosition()
            if point_3d is not None:
                fiducial_points[label].append(point_3d)

    # --- Filter fiducials ---
    avg_points = {}
    for label, points in fiducial_points.items():
        points = np.array(points)
        if points.size == 0:
            warnings.warn(f"No points found for fiducial '{label}'. Skipping this label.")
            continue
        if len(points) == 1:
            avg_point = points[0]
        else:
            z_scores = np.abs(stats.zscore(points))
            mask = (z_scores < filter_thresh).all(axis=1)
            if not np.any(mask):
                warnings.warn(f"All points for fiducial '{label}' filtered out as outliers. Using unfiltered mean.")
                avg_point = np.mean(points, axis=0)
            else:
                avg_point = np.mean(points[mask], axis=0)
        avg_points[label] = avg_point.tolist()

    # --- Build Fiducials object ---
    # Create the Fiducials object and populate the control points in the original style
    fiducials = Fiducials(color=[0, 1, 0])
    for label, (_, fid_id) in landmark_map.items():
        if label in avg_points:
            fiducials.control_points.append(ControlPoint(avg_points[label], label, id=fid_id))
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
    points = find_fiducials(mesh)
    if args.save:
        points.to_file(
            os.path.join(data_dir, "output_points", f"fiducials_{scan_id:03d}.mrk.json")
        )
    else:
        points.print()
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
        cp_actors = points.get_actors(size=args.point_size)
        for cp in cp_actors:
            renderer.AddActor(cp)
        if reference_fiducials:
            cp_actors = reference_fiducials.get_actors(size=args.point_size)
            for cp in cp_actors:
                renderer.AddActor(cp)
        renderWindow.Render()
        renderWindowInteractor.Start()
