For contest instructions, see the [Instructions](INSTRUCTIONS.md)

# Auto Fiducial Detection
## Author: [Pin-Hao Cheng](https://github.com/pinhaocheng/), [Michael Feldkamp](https://github.com/mkfeldkamp), [Adam Imdieke](https://github.com/AdamImd)

## Our solution
This project implements an automated pipeline for detecting anatomical fiducial points on 3D facial meshes. The solution leverages MediaPipe's face landmark detection, VTK for 3D mesh rendering and picking, and a multi-view approach to robustly localize key facial landmarks. The pipeline:
- Loads a 3D mesh (OBJ format with texture).
- Renders the mesh from multiple viewpoints.
- Uses MediaPipe to detect 2D facial landmarks in each rendered view.
- Projects detected 2D landmarks back to 3D using VTK picking.
- Aggregates and filters the 3D points to produce robust fiducial locations.
- ***Note that all fiducials label id is being set to a value of 999, since there is discrepencies between the provided ground truth and template json files.***


 | Name         | Default Value | Description                                                                                      |
|--------------|--------------|--------------------------------------------------------------------------------------------------|
| `num_views`  | 16           | The number of camera angles used to render and analyze the face. Increasing this value can provide more detailed coverage but may increase computation time.        |
| `image_size` | 800          | The resolution (in pixels) of the images rendered and analyzed for each view. A higher value results in higher-resolution images, which may improve accuracy but also increases memory and processing requirements.                     |
| `sweep_size` | 5            |  The size in degrees of steps used when searching for a frontal view of the face. A larger step size may result in a wider front view search space, but could be less precise.              |
| `filter_thresh` | 1.0       | The threshold value used to discard predicted points based on the z-score. Only points with a value below this threshold are kept for further processing.            |
| `model_path` | `mediapipe/face_landmarker.task` | The file path to the MediaPipe pre-trained model used for face landmark detection.        |


## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/<yourusername>/AutoFiducialContest.git
   cd AutoFiducialContest
   ```
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
   If you encounter issues, try:
   ```bash
   pip install vtk mediapipe scipy numpy
   ```

3. **Prepare input data:**
   - Place your input meshes (`.obj` files), along with their corresponding `.mtl` and texture (`.png`) files, in the directory:
     ```
     data/training/input_meshes/
     ```
     The mesh filenames should follow the format: `scan_XXX.obj` (e.g., `scan_000.obj`).

4. **MediaPipe model file (if not present):**
   - Download the MediaPipe face landmark detection task file (`face_landmarker.task`) from the official [MediaPipe Models repository](https://developers.google.com/mediapipe/solutions/vision/face_landmarker#models).
   - Place the downloaded `face_landmarker.task` file in the following directory:
     ```
     mediapipe/
     ```
     The full path should be: `mediapipe/face_landmarker.task`

5. (Optional) Place reference fiducial files in `data/training/reference_points/` if you wish to compare your results to ground truth.



## System Specs
Tested on:
- Operating System: MacOS Sequoia 15.4
   - CPU: *M4*
   - RAM: *16 GB*
   - Estimated processing time: *16-22 seconds per scan*

- Operating System: MacOS Sequoia 15.4
   - CPU: *M4*
   - RAM: *32 GB*
   - Estimated processing time: *10-15 seconds per scan*


 - Operating System: Ubuntu 22.04
   - CPU: *Ryzen 9 3950x*
   - RAM: *64 GB*
   - Estimated processing time: *~30 seconds per scan*
   
