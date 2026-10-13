# =============================================================================
#
#   CRISIS — Parameter File Creation Script
#
#   Builds the CRISIS-Landlab parameter dictionary (project details, input
#   file paths, triggering/seismic/hydrological settings, shear strength,
#   etc.) and optionally writes it out as a JSON parameter file.
#
#   Usage
#   -----
#   As a standalone script — writes Outputs/params.json by default:
#
#       python Scripts/create_parameter_file.py
#       python Scripts/create_parameter_file.py -o Outputs/my_params.json
#
#   As an importable module — get the dict directly, with any values
#   overridden, and pass it straight to CrisisComponent:
#
#       from landlab.components.crisis import build_parameters
#       params = build_parameters(Ru=0.5, time_points=100)
#       crisis = CrisisComponent(grid, params)
#
#   The resulting JSON file can also be run standalone through
#   _crisis_core.py (HPC / batch use):
#
#       python Scripts/_crisis_core.py Outputs/params.json
#
# =============================================================================

import json
from pathlib import Path

# ── Project directories ──────────────────────────────────────────────────────
# This file lives in Scripts/, so the repo root is one level up.
ROOT_DIR    = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT_DIR / "Scripts"
INPUTS_DIR  = ROOT_DIR / "Inputs"
OUTPUTS_DIR = ROOT_DIR / "Outputs"


def build_parameters(**overrides):
    """
    Build the CRISIS-Landlab parameter dictionary.

    Starts from the current default values (matching the example run) and
    applies any keyword overrides on top, e.g.:

        params = build_parameters(Ru=0.5, Triggering_Indicator=2)

    Returns
    -------
    dict
    """

    params = {

        # ── Project details ───────────────────────────────────────────────────
        "project_title": "Example_landlab_standalone",  # name for this run — output folder will be <title>_Outputs/
        "analyst_name":  "Yuval Shmilovitz",             # optional
        "project_date":  "15/03/2026",                   # optional
        "site_location": "Utuado, Puerto Rico",          # optional
        "notes":         "",                             # optional free-form notes

        # ── Output ────────────────────────────────────────────────────────────
        "output_dir":                  str(OUTPUTS_DIR),  # folder to write outputs into
        "input_dir": str(INPUTS_DIR),  # folder to write outputs into

        "Coordinate_Reference_System": "EPSG:6566",        # EPSG code for shapefiles

        # ── Raster input files ───────────────────────────────────────────────
        # Slope is always loaded from Slope.h5 (ArcGIS export, already in degrees).
        # No Landlab fallback — this file must exist.

        "slope_file": str(INPUTS_DIR / "Slope.h5"),

        # Flow direction and flow accumulation — pre-computed ArcGIS HDF5 rasters.
        # Leave these as "" if you'd rather have Landlab compute routing from the
        # DEM instead (FlowDirectorD8 / FlowAccumulator).

        "flow_direction_file":    str(INPUTS_DIR / "Flow_Direction.h5"),
        "flow_accumulation_file": str(INPUTS_DIR / "Flow_Accumulation.h5"),

        # Coordinate files are always required — used to place landslide shapefiles
        # in the correct real-world UTM location on the map.

        "x_coordinates_file": str(INPUTS_DIR / "x_coordinates.h5"),  # easting of each DEM column (m)
        "y_coordinates_file": str(INPUTS_DIR / "y_coordinates.h5"),  # northing of each DEM row (m)

        # ── Triggering event ─────────────────────────────────────────────────
        # Triggering_Indicator controls what drives slope failure:
        #   1 = Rainfall only — pore pressure drives failure (Factor of Safety < 1)
        #   2 = Coupled Rainfall + Earthquake — both pore pressure and seismic
        #       displacement are evaluated; requires PGA raster and seismic params below

        "Triggering_Indicator": 1,

        # Seismic_displacement_model selects which empirical Newmark model to use
        # (only relevant when Triggering_Indicator = 2):
        #   1 = Jibson (2007)
        #   2 = Rathje Rigid (Rathje et al., 2014)
        #   3 = Rathje Flexible (Rathje & Antonakos, 2011)
        #   4 = Bray & Macedo (2019)

        "Seismic_displacement_model": None,   # not used — Triggering_Indicator = 1

        # Rathje_method selects the ground motion intensity measure for Rathje models:
        #   1 = PGAM (peak ground acceleration, magnitude-corrected)
        #   2 = PGAV (requires PGV raster)

        "Rathje_method": None,   # not used — Triggering_Indicator = 1

        # ── Seismic raster files ─────────────────────────────────────────────
        # Required only when Triggering_Indicator = 2.

        "pga_file": "",   # HDF5 file — peak ground acceleration raster (g)
        "pgv_file": "",   # HDF5 file — peak ground velocity raster (cm/s);
                          # required only for Rathje PGAV and Bray & Macedo models

        # ── Fault plane corners ──────────────────────────────────────────────
        # Required only for Rathje Flexible and Bray & Macedo displacement models.
        # Define the fault rupture plane using four 3D corner points in the same
        # projected CRS as the DEM. X and Y in meters (easting/northing); Z in
        # meters depth, positive downward (surface = 0, 15 km deep = 15000).
        # Corner order: top-left → top-right → bottom-right → bottom-left
        # (top = shallow edge, bottom = deep edge).

        "fault_X1": None, "fault_Y1": None, "fault_Z1": None,   # top-left corner
        "fault_X2": None, "fault_Y2": None, "fault_Z2": None,   # top-right corner
        "fault_X3": None, "fault_Y3": None, "fault_Z3": None,   # bottom-right corner
        "fault_X4": None, "fault_Y4": None, "fault_Z4": None,   # bottom-left corner

        # ── Scalar inputs ────────────────────────────────────────────────────
        "Cell_size":    1.0,    # DEM grid cell size (meters)
        "Gamma":        22.0,   # total unit weight of soil (kN/m³)
        "z_min":        0.125,  # depth at center of shallowest layer (meters)
        "z_max":        1.875,  # depth at center of deepest layer (meters)
        "Depth_points": 8,      # number of subsurface depth layers to evaluate FS at
        "time_points":  73,     # total number of time steps in the simulation
        "Min_LD_area":  12.0,   # minimum landslide area to include in output (m²)

        # top_or_bottom_indicator: when multiple depth layers fail at the same cell,
        # which depth is used to define the triggering surface for the 3D geometry?
        #   0 = shallowest failing depth (use for shallow translational failures)
        #   1 = deepest failing depth

        "top_or_bottom_indicator": 0,

        # ── Seismic scalar inputs ────────────────────────────────────────────
        # Required only when Triggering_Indicator = 2.

        # Earthquake_time_point: which time step (0-based index) the earthquake
        # occurs during the storm sequence. E.g. 36 means the earthquake strikes
        # at time step 36 out of 73. Must be between 0 and time_points - 1.

        "Earthquake_time_point": None,

        # Seismic_displacement_threshold: Newmark displacement above which a cell
        # is classified as seismically unstable (cm). Typically 5 cm.

        "Seismic_displacement_threshold": None,

        "M":  None,   # earthquake moment magnitude Mw
        "Ts": None,   # fundamental site period of the sliding mass (s);
                      # required for Rathje Flexible and Bray & Macedo models;
                      # set to 0 for rigid block models (Jibson, Rathje Rigid)

        # ── Hydrological formulation ─────────────────────────────────────────
        # Hydrological_model_indicator controls how pore pressure is provided:
        #   0 = constant Ru — a single uniform pore pressure coefficient applied
        #       everywhere and at all times (no files needed)
        #   1 = variable pore pressure — spatially and temporally varying pressure
        #       head files from an external groundwater model (ParFlow, TRIGRS, etc.)
        #       Files must be named: Pressure_head_T{t}Z{d}.h5
        #       e.g. Pressure_head_T0Z1.h5 = time step 0, depth layer 1

        "Hydrological_model_indicator": 0,

        "hydro_files_dir": None,   # folder containing Pressure_head_T{t}Z{d}.h5 files
        "Ru": 0.35,                # pore pressure ratio [0–1] (constant Ru option only); Ru = u / (gamma * z) = (psi * gamma_w) / (gamma * z)

        # ── Volumetric water content ─────────────────────────────────────────
        # Strength_variation_with_theta controls Bishop's effective stress coefficient:
        #   0 = fully saturated assumption — chi = 1 everywhere (no theta files needed)
        #   1 = chi varies with degree of saturation, interpolated from water content
        #       files named Theta_T{t}Z{d}.h5 (e.g. Theta_T0Z1.h5)
        #       chi = (theta - theta_r) / (theta_s - theta_r)

        "Strength_variation_with_theta": 0,

        "theta_files_dir": "",   # folder containing Theta_T{t}Z{d}.h5 files
        "Theta_s": 0.45,         # saturated volumetric water content
        "Theta_r": 0.1,          # residual volumetric water content

        # ── Shear strength ───────────────────────────────────────────────────
        # Strength_status controls whether shear strength is uniform or spatially variable:
        #   0 = spatially uniform — single c and Phi apply to all cells and depths
        #   1 = spatially variable 3D arrays — different c and Phi at every cell
        #       and depth layer, loaded from HDF5 files of shape (rows x cols x Depth_points)

        "Strength_status": 0,

        "c":   20.0,   # cohesion (kPa) — uniform strength only
        "Phi": 37.0,   # friction angle (degrees) — uniform strength only
        "cohesion_3d_file": "",   # HDF5 file — 3D cohesion array (kPa) — spatially variable only
        "phi_3d_file":      "",   # HDF5 file — 3D friction angle array (degrees) — spatially variable only
    }

    params.update(overrides)
    return params


def save_parameters(params, path):
    """
    Write a parameter dict out as a JSON file (any Path objects are
    stringified so the file is valid JSON).

    Returns the resolved Path that was written to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    serializable = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in params.items()
    }

    with open(path, "w") as fh:
        json.dump(serializable, fh, indent=2)

    return path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create a CRISIS-Landlab parameter JSON file from the default parameters."
    )
    parser.add_argument(
        "-o", "--output",
        default=str(OUTPUTS_DIR / "params.json"),
        help="Where to write the parameter JSON file (default: Outputs/params.json)",
    )
    args = parser.parse_args()

    params = build_parameters()
    out_path = save_parameters(params, args.output)
    print(f"Parameter file written to {out_path}")
