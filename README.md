For contest instructions, see the [Instructions](INSTRUCTIONS.md)

# Auto Fiducial Detection
## Author: *your name*

## My solution
The solution implements a geometric heuristic-based approach to identify seven facial fiducial points (Nasion, Left/Right Ears, and Inner/Outer Left/Right Eyes) from a 3D facial mesh.

**Coordinate System Handling:**
1.  The input mesh (assumed to be in a RAS-like system, specifically X-Left, Y-Anterior, Z-Superior) is first converted to an internal working coordinate system of RPS (X-Right, Y-Posterior, Z-Superior) using a standard (-X, -Y, Z) transformation. This is done via the `convert_between_ras_and_lps` function.
2.  All geometric heuristics operate in this internal RPS system.
3.  The final identified fiducial coordinates are then converted to the target Slicer LPS system (X-Right, Y-Anterior, Z-Superior) by negating their Y-coordinates before they are stored in the output `Fiducials` object.

**Fiducial Detection Heuristics (operating in internal RPS: X-Right, Y-Posterior, Z-Superior):**

1.  **Nasion:**
    *   Candidates are filtered to a region:
        *   Near the horizontal midline of the face.
        *   In the anterior part of the face (more negative Y in RPS).
        *   Within a specific vertical band refined to target the typical nasion height: `center[2] - size[2]*0.05` to `center[2] + size[2]*0.25`.
    *   The nasion is chosen as the most posterior point (largest Y value in RPS) among these candidates, aiming for the deepest point of concavity at the nasal bridge.

2.  **Ear Points (Left Tragus, Right Tragus):**
    *   Candidates are filtered to a region:
        *   Around the vertical midpoint of the head.
        *   In the posterior part of the head (more positive Y in RPS).
    *   The right ear is the point with the maximum X value (most to the patient's right) among these candidates.
    *   The left ear is the point with the minimum X value (most to the patient's left) among these candidates.
    *   (Note: The left ear heuristic showed lower accuracy on the example scan compared to the right ear).

3.  **Eye Corners (Inner/Outer, Left/Right):**
    *   These are derived geometrically as weighted averages of the detected nasion and respective ear points, similar to the original placeholder logic. Their accuracy is thus dependent on the nasion and ear point accuracy.

**General Notes:**
*   The approach relies on the mesh being reasonably centered and oriented, although the initial coordinate transformation handles common RAS-to-LPS differences.
*   The heuristics use global mesh properties like center and size to define search regions.
*   Fallback mechanisms (using the mesh center) are in place if no candidate points are found by the heuristics, though these are rudimentary.

## Installation
*if there are additional instructions required for installation, describe them here. Installation must be easy to follow*

## System Specs
- Operating System: *your os*
- CPU: *CPU*
- GPU: *GPU*
- RAM: *RAM*
- Estimated processing time: *tell us approximately how long it takes to run*
