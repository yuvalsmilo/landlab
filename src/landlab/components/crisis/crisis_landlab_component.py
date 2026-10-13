# =============================================================================
#
#   CRISIS — Coupled Regional Rainfall-Induced and Seismic Slope Instability
#            Simulations
#
#   Module  : crisis_landlab_component.py
#   Purpose : Landlab Component implementation of the CRISIS forward analysis model.
#             Rainfall only, earthquake only, or coupled (rainfall + seismic) triggers
#
#   Authors : Mirna Kassem
#             University of California, Berkeley
#             kassem_mirna@berkeley.edu
#
#             Prof. Dimitrios Zekkos
#             University of California, Berkeley
#             zekkos@berkeley.edu
#   
#   Code written and reviewed also by Yuval Shmilovitz (yuval.shmilovich@mail.huji.ac.il)
        
# =============================================================================


# =============================================================================
# Imports
# =============================================================================
# All CRISIS physics functions are imported from the private _crisis_core.py module;
# this file contains interface logic only.

from __future__ import annotations
from pathlib import Path
import h5py
import numpy as np
from landlab import Component, RasterModelGrid

Gamma_W = 9.81

# pull in the actual physics/math functions from the core CRISIS module
from ._crisis_core import (
    _build_upslope_map,
    _find_triggering_cells_vectorized,
    Determine_Pseudo_3D_landslides,
    Modeled_landslides_shapefiles,
    _save_individual_excel,
    _save_summary_excel,
    load_h5,
    Rupture_Dist,
)


# =============================================================================
# Module-level helper
# =============================================================================
# Landlab stores node data as flat 1D arrays; CRISIS physics functions expect
# 2D (nrows, ncols) arrays — this helper reshapes them on the fly.

# small helper: turn a flat Landlab field back into a 2D grid shape
def _node_to_2d(grid, field_name):
    """Return a node field reshaped to (nrows, ncols).

    Examples
    --------
    >>> import numpy as np
    >>> from landlab import RasterModelGrid
    >>> grid = RasterModelGrid((2, 3))
    >>> _ = grid.add_field("topographic__elevation", np.arange(6.0), at="node")
    >>> _node_to_2d(grid, "topographic__elevation")
    array([[ 0.,  1.,  2.],
           [ 3.,  4.,  5.]])
    """
    # grab the named field and reshape it to match the grid's row/col layout
    return grid.at_node[field_name].reshape(grid.shape)


# =============================================================================
# CrisisComponent — main class
# =============================================================================
# this is the main class — it implements CRISIS as a native Landlab component

class CrisisComponent(Component):
    """Coupled rainfall-induced and coseismic landslide hazard model (CRISIS).

    Implements a pseudo-3-D slope stabilty analysis to predict landslides polygons
    (with individual location, area, volume) over a Landlab ``RasterModelGrid``.
    Time-varying pore-pressure and moisture fields (both optional)can be supplied
    either via HDF5 file directories (same convention as the standalone CRISIS model)
    or by updating Landlab node fields before each call to ``run_one_step``.

    The component delegates all physics to ``_crisis_core.py`` and conforms
    natively to the Landlab component API.

    Parameters
    ----------
    grid : RasterModelGrid
        Landlab raster grid.  Must carry the required input node fields listed
        in ``_info`` before the component is instantiated.
    params : dict
        CRISIS model parameters.  All keys and their meanings are documented
        in ``crisis_params.py``.  The most commonly used keys are listed below
        under *Key Parameters*.
    log_callback : callable, optional
        Receives progress strings during the simulation.  Defaults to
        ``print``.  The GUI passes its text-widget updater here.

    Key Parameters
    ----------------
    The ``params`` dict must contain at minimum:

    Cell_size : float
        Grid cell width (m).  Must match ``grid.dx``.
    Gamma : float
        Unit weight of soil (kN/m³).
    z_min : float
        Shallowest depth layer (m).
    z_max : float
        Deepest depth layer (m).
    Depth_points : int
        Number of depth discretisation layers.
    time_points : int
        Total number of time steps to be run.
    Min_LD_area : float
        Minimum accepted landslide planar area (m²).
    Triggering_Indicator : {1, 2}
        1 = rainfall only; 2 = rainfall + seismic.
    Hydrological_model_indicator : {0, 1}
        0 = constant Ru pore-pressure ratio; 1 = time-varying pressure files.
    Strength_status : {0, 1}
        0 = uniform cohesion/friction; 1 = spatially variable.
    Strength_variation_with_theta : {0, 1}
        0 = fully saturated (Bishop χ = 1); 1 = Bishop χ scaled by θ.
    top_or_bottom_indicator : {0, 1}
        0 = report shallowest failure depth; 1 = deepest.
    Theta_s : float
        Saturated volumetric water content (−).
    Theta_r : float
        Residual volumetric water content (−).
    Coordinate_Reference_System : str
        EPSG string for shapefile export (e.g. ``"EPSG:32610"``).
    output_dir : str or Path
        Root directory for all output files.
    project_title : str, optional
        Label prepended to the output folder name.

    Attributes
    ----------
    landslide_fid : ndarray, shape (nrows, ncols)
        Cumulative landslide-ID raster.  NaN where no failure has occurred.
    failed_cells : ndarray of bool, shape (nrows, ncols)
        True at every cell that has failed in any time step.
    state : dict
        Full output accumulator (same structure as ``forward_analysis.run_model``).
    current_time_index : int
        Index of the *next* time step that will be run by ``run_one_step``.

    """

    # ── Component identity ────────────────────────────────────────────────────

    # Landlab uses _name to identify the component in error messages and docs
    # the name Landlab shows for this component
    _name = "CrisisComponent"

    # _unit_agnostic = False — inputs and outputs must be in the SI units declared in _info
    # tells Landlab our units matter and must be respected
    _unit_agnostic = False

    # =============================================================================
    # Field declarations
    # =============================================================================
    # _info declares every grid field this component reads ("intent": "in") or
    # writes ("intent": "out"). Landlab validates required fields at construction
    # time and uses this dict to auto-generate documentation.

    # dictionary listing every input/output field this component needs or makes
    _info = {

        # ── Required inputs (DEM-derived) ─────────────────────────────────────
        # Must exist on the grid before CrisisComponent is instantiated.
        # The first two come from your DEM rasters; the last two are produced
        # by FlowDirectorD8 and FlowAccumulator.

        # ground surface height at each grid point, from the DEM
        "topographic__elevation": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Surface elevation (DEM).  No-data cells should be NaN.",
        },

        # which way water flows from each cell (D8 direction)
        "flow__receiver_node": {
            "dtype": int,
            "intent": "in",
            "optional": True,
            "units": "-",
            "mapping": "node",
            "doc": (
                "Receiver-node index for each node (D8 routing). Produced by FlowDirectorD8. "
                "Optional when flow_direction_file is supplied in params — in that case the "
                "ArcGIS D8 HDF5 file is loaded directly and this field is not used."
            ),
        },

        "drainage_area": {
            "dtype": float,
            "intent": "in",
            "optional": True,
            "units": "m^2",
            "mapping": "node",
            "doc": (
                "Upstream contributing area in m². Produced by FlowAccumulator (already in m²). "
                "Optional when flow_accumulation_file is supplied in params — in that case the "
                "ArcGIS flow accumulation HDF5 file is loaded directly (cell count × Cell_size² "
                "to convert to m²) and this field is not used."
            ),
        },

        # ── Optional seismic inputs ───────────────────────────────────────────
        # Required only when Triggering_Indicator == 2 (seismic or coupled run).
        # Can be provided as grid fields or via pga_file / pgv_file HDF5 paths.

        # strongest shaking (acceleration) expected at each cell, for earthquakes
        "peak_ground_acceleration": {
            "dtype": float,
            "intent": "in",
            "optional": True,
            "units": "g",
            "mapping": "node",
            "doc": "Peak Ground Acceleration (g). Required when Triggering_Indicator == 2.",
        },

        # strongest shaking (velocity) expected at each cell, for earthquakes
        "peak_ground_velocity": {
            "dtype": float,
            "intent": "in",
            "optional": True,
            "units": "cm/s",
            "mapping": "node",
            "doc": "Peak Ground Velocity (cm/s). Required for Rathje PGAV and Bray & Macedo (2019) models.",
        },

        # ── Optional time-varying hydrology inputs ────────────────────────────
        # Alternative to HDF5 files from hydro_files_dir / theta_files_dir.
        # If another Landlab component produces these fields, CRISIS reads them directly.

        # how much water pressure is pushing up in the soil right now
        "soil__pressure_head": {
            "dtype": float,
            "intent": "in",
            "optional": True,
            "units": "m",
            "mapping": "node",
            "doc": (
                "Pore-pressure head for the current time step (shallowest layer). "
                "Used when Hydrological_model_indicator == 1 and no HDF5 directory is provided."
            ),
        },

        # how wet the soil is right now (fraction of pore space filled with water)
        "soil__volumetric_water_content": {
            "dtype": float,
            "intent": "in",
            "optional": True,
            "units": "-",
            "mapping": "node",
            "doc": (
                "Volumetric water content for the current time step (shallowest layer). "
                "Used when Strength_variation_with_theta == 1 and no HDF5 directory is provided."
            ),
        },

        # ── Outputs ───────────────────────────────────────────────────────────
        # Created on the grid by __init__ and updated after every run_one_step call.

        # ID number telling which landslide (if any) each cell belongs to
        "landslide__failure_id_accumulated": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": (
                "Cumulative landslide-ID raster. Each failed cell carries the integer ID "
                "of the landslide it belongs to. NaN where no failure has occurred."
            ),
        },

        # simple true/false flag: has this cell ever failed
        "landslide__failed_accumulated": {
            "dtype": bool,
            "intent": "out",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "True at nodes that have failed in any time step (cumulative).",
        },
        "landslide__failure_id": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": (
                "Landslide-ID raster. Each failed cell carries the integer ID "
                "of the landslide it belongs to. NaN where no failure has occurred."
            ),
        },

        # simple true/false flag: has this cell ever failed
        "landslide__failed": {
            "dtype": bool,
            "intent": "out",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "True at nodes that have failed in a current time step",
        },
        "failure__depth": {
            "dtype": bool,
            "intent": "out",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "failure depth at node that have failed in a current time step ",
        },
        "triggering__cells": {
            "dtype": bool,
            "intent": "out",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "Landslide triggering cells in a current time step",
        },
    }

    # =============================================================================
    # Constructor  (__init__)
    # =============================================================================
    # this runs once when you create a CrisisComponent — it sets everything up

    def __init__(
            self,
            grid,
            params,
            slope = 'topographic__slope',
            drainage_area = 'drainage_area',
            flow_direction= "flow__receiver_node",
            log_callback=print):
        """Initialise the CRISIS component and pre-load all static data.

        Parameters
        ----------
        grid : RasterModelGrid
            Landlab raster grid carrying the required input fields.
        params : dict
            CRISIS model parameters (see class docstring).
        log_callback : callable, optional
            Progress-message sink.  Defaults to ``print``.

        Raises
        ------
        TypeError
            If *grid* is not a ``RasterModelGrid``.
        FileNotFoundError
            If ``Hydrological_model_indicator == 1`` and neither HDF5 pressure
            files nor a ``soil__pressure_head`` grid field are available.

        Examples
        --------
        Create a 3x100 grid representing a uniformly inclined slope, and
        instantiate CrisisComponent on it (same setup as :meth:`run_one_step`).

        >>> import numpy as np
        >>> from landlab import RasterModelGrid
        >>> from landlab.components import FlowAccumulator
        >>> from landlab.components import CrisisComponent
        >>> from landlab.components.crisis import build_parameters
        >>> nrows, ncols = 3, 100
        >>> dx = 1.0  # grid spacing (m)
        >>> grid = RasterModelGrid((nrows, ncols), xy_spacing=dx)
        >>> slope = 0.9  # rise per unit run
        >>> elevation = slope * grid.x_of_node
        >>> _ = grid.add_field("topographic__elevation", elevation, at="node")
        >>> s = grid.add_zeros("topographic__slope", at="node")
        >>> s[:] = np.degrees(grid.calc_slope_at_node("topographic__elevation"))
        >>> grid.set_closed_boundaries_at_grid_edges(True, True, True, True)
        >>> outlet_id = grid.nodes_at_left_edge[1]
        >>> grid.status_at_node[outlet_id] = grid.BC_NODE_IS_FIXED_VALUE
        >>> fa = FlowAccumulator(grid, runoff_rate=1)
        >>> fa.run_one_step()
        >>> params = build_parameters(Ru=0.8, Hydrological_model_indicator=0)
        >>> crisis = CrisisComponent(grid, params)
        >>> crisis.current_time_index
        0
        """

        # ── Grid type check ───────────────────────────────────────────────────
        # Guard before super().__init__ — Landlab's base class validates required
        # fields immediately, and those only exist on RasterModelGrid

        # stop early with a clear error if someone passes the wrong grid type
        if not isinstance(grid, RasterModelGrid):
            raise TypeError("CrisisComponent requires a RasterModelGrid.")

        # ── Base class initialisation ─────────────────────────────────────────
        # super().__init__ registers the grid and validates all _info "in" fields

        # hand the grid to Landlab's base class so it checks our required fields
        super().__init__(grid)

        # ── Store params and logger ───────────────────────────────────────────

        # remember the parameters dict for later use
        self._params = params
        # remember how to print/report progress messages
        self._log    = log_callback

        # ── Simulation clock and grid shape ───────────────────────────────────

        # Internal counter — tracks which time step run_one_step will process next
        # start the simulation clock at time step 0
        self._current_time_index = 0

        # store the grid's row and column counts for convenience
        self._rows, self._clms = grid.shape

        # ── Elevation ─────────────────────────────────────────────────────────
        # make our own private copy of the elevation grid
        self._Elev_Mat = _node_to_2d(grid, "topographic__elevation").copy().astype(np.float32)
        # turn "no data" elevation placeholders into proper NaN (missing value)
        self._Elev_Mat[self._Elev_Mat < -900] = np.nan

        # ── Slope (degrees) ───────────────────────────────────────────────────
        # Always loaded from the user-supplied Slope.h5 file (already in degrees).
        # The file path must be provided as slope_file in params.

        if grid.has_field(slope, at='node'):
            self._Slope_Mat = grid.at_node[slope]
        else:
            slope_file = params["slope_file"]
            self._Slope_Mat = load_h5(slope_file).astype(np.float32)
        # clean up any ArcGIS no-data sentinels
        self._Slope_Mat[self._Slope_Mat < -900] = np.nan

        # ── Drainage area (m²) ────────────────────────────────────────────────
        # Option A: load from an ArcGIS HDF5 file — ArcGIS stores upstream cell
        #           COUNT, so multiply by Cell_size² to convert to m²
        # Option B: read from the Landlab grid field produced by FlowAccumulator
        #           — Landlab already outputs area in m², no conversion needed

        if grid.has_field(drainage_area, at='node'):
            self._Area_Mat = grid.at_node[drainage_area]

        else:
            fa_file = params.get("flow_accumulation_file", "")
            if fa_file and Path(fa_file).exists():
                # load the ArcGIS flow accumulation grid (cell count) and convert to m²
                self._Area_Mat  = load_h5(fa_file).astype(np.float32)
                self._Area_Mat *= params["Cell_size"] ** 2

        # clean up any bogus negative values
        self._Area_Mat[self._Area_Mat < 0] = np.nan

        # ── D8 flow direction matrix ───────────────────────────────────────────
        # Priority: load ArcGIS D8 HDF5 file directly (same as standalone forward_analysis.py),
        # or fall back to converting Landlab receiver-node indices from the grid field.

        if grid.has_field(flow_direction, at='node'):
            self._FD_Mat = self._build_fd_mat_from_landlab(grid, flow_direction)
        else:
            fd_file = params.get("flow_direction_file", "")
            if fd_file and Path(fd_file).exists():
                # load the flow-direction grid straight from the file
                self._FD_Mat = load_h5(fd_file).astype(np.float32)



        # ── Coordinate arrays for shapefile export ────────────────────────────
        # Easting/northing of each grid column/row; loaded from HDF5 if provided,
        # otherwise derived from Landlab's node coordinate arrays

        # ── Coordinates (easting / northing) ─────────────────────────────────
        # Always loaded from user-supplied HDF5 files to guarantee real-world
        # UTM coordinates. x_coordinates_file and y_coordinates_file must be
        # provided in params.

        self._x = load_h5(params["x_coordinates_file"])
        self._y = load_h5(params["y_coordinates_file"])

        # ── Depth discretisation vector ───────────────────────────────────────
        # FS and Newmark displacement are evaluated at each depth for every cell

        # build the list of soil depths we'll test for failure (shallow to deep)
        self._Depths_vector = np.linspace(
            params["z_min"], params["z_max"],
            params["Depth_points"], dtype=np.float32,
        )

        # ── Static seismic arrays (Triggering_Indicator == 2 only) ───────────
        # PGA and PGV are spatial maps loaded once; rupture distance computed
        # once from the fault plane geometry

        # placeholders for earthquake-related arrays, filled in only if needed
        self._PGA_2D_array     = None
        self._PGV_2D_array     = None
        self._Rupture_2D_array = None

        # only load earthquake data if this run includes seismic triggering
        if params.get("Triggering_Indicator") == 2:

            # load peak ground acceleration map (from grid field or file)
            self._PGA_2D_array = self._load_field_or_h5(
                "peak_ground_acceleration", params.get("pga_file", "")
            )

            # which displacement model to use for slope failure under shaking
            seismic_model = params.get("Seismic_displacement_model", 1)
            # which Rathje calculation method to use
            rathje_method = params.get("Rathje_method", 1)

            # PGV required for Rathje PGAV (models 2/3, method 2) and Bray (model 4)
            # figure out if this model choice needs peak ground velocity too
            pgv_needed = (seismic_model == 4) or (
                seismic_model in (2, 3) and rathje_method == 2
            )
            if pgv_needed:
                # load peak ground velocity map (from grid field or file)
                self._PGV_2D_array = self._load_field_or_h5(
                    "peak_ground_velocity", params.get("pgv_file", "")
                )

            # Rupture distance required for Rathje flexible and Bray models (3, 4)
            # some models also need distance-to-fault-rupture, so compute it
            if seismic_model in (3, 4):
                self._Rupture_2D_array = self._compute_rupture_distances(params)

        # ── Spatially variable strength arrays (Strength_status == 1 only) ───
        # 3D arrays (rows × cols × depth_points) loaded once from HDF5

        # placeholders for soil-strength maps that vary by location and depth
        self._Cohesion_3D_array = None
        self._Phi_3D_array      = None

        # only load these if strength is set to vary spatially
        if params.get("Strength_status") == 1:
            # path to the cohesion (soil stickiness) 3D file
            c3d = params.get("cohesion_3d_file", "")
            # path to the friction angle 3D file
            p3d = params.get("phi_3d_file",      "")
            if c3d:
                # load cohesion values for every cell and depth
                self._Cohesion_3D_array = load_h5(c3d)
            if p3d:
                # load friction angle values for every cell and depth
                self._Phi_3D_array = load_h5(p3d)

        # ── Upslope flow map ──────────────────────────────────────────────────
        # Pre-build {cell: [upslope neighbors]} dict so BFS delineation does not
        # recompute it at every time step (most expensive setup step for large grids)

        # let the user know this slow setup step is starting
        # log_callback("Precomputing upslope flow map...\n")
        # build a lookup of "which cells flow into this cell" once, up front
        self._upslope_map = _build_upslope_map(self._FD_Mat)

        # ── Output directory ──────────────────────────────────────────────────
        # Shapefiles, Excel workbooks, and FID_all_times.h5 go in <title>_Outputs/

        # build a clean project title to use as a folder name
        _title           = (params.get("project_title") or "CRISIS").strip() or "CRISIS"
        # the actual output folder path, named after the project
        self._output_dir = Path(params["output_dir"]) / f"{_title}_Outputs"
        # create that folder now if it doesn't already exist
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # ── State accumulator ─────────────────────────────────────────────────
        # All arrays that change across time steps live here, mirroring the state
        # dict in forward_analysis.run_model so the same writers can be reused.
        #   "all_times" rasters  — grid-shaped, updated in place each step
        #   "with_time" vectors  — one entry appended per step, written to Excel

        # how many time steps this run will simulate
        time_points = params["time_points"]

        # this dictionary holds every result that builds up as the simulation runs
        self._state = {

            # ── Cumulative rasters ────────────────────────────────────────────
            # grid of which cells have ever failed (true/false)
            "Fail_Index_all_times":  np.zeros((self._rows, self._clms), dtype=bool),
            # grid of landslide ID numbers (NaN = no landslide there)
            "FID_all_times":         np.full((self._rows, self._clms), np.nan, dtype=np.float32),

            # ── Per-step totals (current = new this step only) ────────────────
            # total new landslide area at each time step
            "Total_Current_Area_vector_with_time":      np.zeros(time_points, dtype=np.float32),
            # total new landslide volume at each time step
            "Total_Current_Volume_vector_with_time":    np.zeros(time_points, dtype=np.float32),
            # count of new landslides at each time step
            "Total_Current_NL_vector_with_time":        np.zeros(time_points, dtype=np.float32),

            # ── Per-step running totals (cumulative = all steps so far) ───────
            # running total area of all landslides so far, per time step
            "Total_Cumulative_Area_vector_with_time":   np.zeros(time_points, dtype=np.float32),
            # running total volume of all landslides so far, per time step
            "Total_Cumulative_Volume_vector_with_time": np.zeros(time_points, dtype=np.float32),
            # running total count of landslides so far, per time step
            "Total_Cumulative_NL_vector_with_time":     np.zeros(time_points, dtype=np.float32),

            # ── Per-step lists — new (current) landslide properties ───────────
            # list of cell locations that triggered new landslides, one entry per step
            "Individual_Current_triggering_linear_indices_with_time": [],
            # list of failure depths for the new landslides, one entry per step
            "Individual_Current_triggering_depths_with_time":         [],
            # list of what caused the new failures (rain/quake), one entry per step
            "Individual_Current_triggering_sources_with_time":        [],
            # list of new landslide areas, one entry per step
            "Individual_Current_areas_with_time":                     [],
            # list of new landslide volumes, one entry per step
            "Individual_Current_volumes_with_time":                   [],

            # ── Per-step snapshots — cumulative landslide registry ────────────
            # snapshot of all triggering cell locations seen so far, per step
            "Individual_Cumulative_triggering_linear_indices_with_time": [],
            # snapshot of all failure depths seen so far, per step
            "Individual_Cumulative_triggering_depths_with_time":         [],
            # snapshot of all trigger sources seen so far, per step
            "Individual_Cumulative_triggering_sources_with_time":        [],
            # snapshot of all landslide areas seen so far, per step
            "Individual_Cumulative_areas_with_time":                     [],
            # snapshot of all landslide volumes seen so far, per step
            "Individual_Cumulative_volumes_with_time":                   [],

            # ── Running master registry — all unique triggering cells ─────────
            # master list of every unique landslide-triggering cell location
            "Individual_Cumulative_triggering_linear_indices": [],
            # master list of failure depths matching the above cells
            "Individual_Cumulative_triggering_depths":         [],
            # master list of trigger sources matching the above cells
            "Individual_Cumulative_triggering_sources":        [],
            # master list of volumes matching the above cells
            "Individual_Cumulative_volumes":                   [],
            # master list of areas matching the above cells
            "Individual_Cumulative_areas":                     [],
        }

        # ── Initialise output fields on the grid ──────────────────────────────

        # create the output fields on the Landlab grid so other components can read them
        self._init_output_fields()

        # ── Extract constant model parameters ─────────────────────────────────
        # These values never change during the simulation, so extract them once
        # at initialization rather than re-extracting them in every run_one_step call

        # ── General model settings ────────────────────────────────────────
        self._Gamma       = params["Gamma"]                    # soil unit weight (kN/m³)
        self._Gamma_W     = Gamma_W                               # unit weight of water (kN/m³)
        self._Cell_size   = params["Cell_size"]                # grid cell size (m)
        self._Min_LD_area = params["Min_LD_area"]              # minimum landslide area (m²)
        self._top_bot     = params["top_or_bottom_indicator"]  # shallowest (0) or deepest (1) failure

        # ── Triggering and strength settings ─────────────────────────────
        self._Strength_status               = params["Strength_status"]
        self._Triggering_Indicator          = params["Triggering_Indicator"]
        self._Strength_variation_with_theta = params["Strength_variation_with_theta"]
        self._Hydro_indicator               = params["Hydrological_model_indicator"]
        self._CRS                           = params["Coordinate_Reference_System"]

        # ── Soil strength constants (used if uniform) ─────────────────────
        self._c       = params.get("c")        # constant cohesion (kPa)
        self._Phi     = params.get("Phi")      # constant friction angle (degrees)
        self._Ru      = params.get("Ru")       # constant pore-pressure ratio
        self._Theta_s = params["Theta_s"]      # saturated water content
        self._Theta_r = params["Theta_r"]      # residual water content

        # ── Seismic settings ──────────────────────────────────────────────
        self._Earthquake_time_point          = params.get("Earthquake_time_point")
        self._Seismic_displacement_threshold = params.get("Seismic_displacement_threshold")
        self._Seismic_displacement_model     = params.get("Seismic_displacement_model", 1)
        self._Rathje_method                  = params.get("Rathje_method", 1)
        self._Ts                             = params.get("Ts", 0.3)
        self._M                              = params.get("M")

        # ── Ready message ─────────────────────────────────────────────────────

        # # print a friendly summary that setup is done and what's about to run
        # log_callback(
        #     f"CrisisComponent ready: grid {self._rows} × {self._clms}, "
        #     f"{params['Depth_points']} depth layer(s), "
        #     f"{time_points} time step(s).\n"
        # )

    # =============================================================================
    # Public API
    # =============================================================================

    # ─────────────────────────────────────────────────────────────────────────────
    # run_one_step — advance the simulation by one time step
    # ─────────────────────────────────────────────────────────────────────────────

    def run_one_step(self, dt=1.0):
        """Run the CRISIS forward model for a single time step.

        Advances the internal time-step counter by one each call.  The first
        call corresponds to time-step index 0, the second to index 1, and so
        on.  Time-varying HDF5 files (pressure heads, water content) are loaded
        by index.

        Parameters
        ----------
        dt : float, optional
            Duration of this time step (any unit consistent with the model
            clock).  CRISIS does not use *dt* internally — it is accepted for
            compatibility with the standard Landlab component interface and
            ``ModelGrid`` clock integrations.  Defaults to ``1.0``.

        Raises
        ------
        RuntimeError
            If called more times than ``params["time_points"]``.
        FileNotFoundError
            If ``Hydrological_model_indicator == 1`` and pressure-head data
            cannot be found for the current time step.
        Examples
        --------
        Create a 3x100 grid representing a uniformly inclined slope, draining
        to a single outlet on the left edge, and run CRISIS on it.

        >>> import numpy as np
        >>> from landlab import RasterModelGrid
        >>> from landlab.components import FlowAccumulator
        >>> from landlab.components import CrisisComponent
        >>> from landlab.components.crisis import build_parameters
        >>> nrows, ncols = 3, 100
        >>> dx = 1.0  # grid spacing (m)
        >>> grid = RasterModelGrid((nrows, ncols), xy_spacing=dx)
        >>> slope = 0.9  # rise per unit run
        >>> elevation = slope * grid.x_of_node
        >>> _ = grid.add_field("topographic__elevation", elevation, at="node")
        >>> s = grid.add_zeros("topographic__slope", at="node")
        >>> s[:] = np.degrees(grid.calc_slope_at_node("topographic__elevation"))
        >>> grid.set_closed_boundaries_at_grid_edges(True, True, True, True)
        >>> outlet_id = grid.nodes_at_left_edge[1]
        >>> grid.status_at_node[outlet_id] = grid.BC_NODE_IS_FIXED_VALUE
        >>> fa = FlowAccumulator(grid, runoff_rate=1)
        >>> fa.run_one_step()

        Set up and run CRISIS.
        >>> Ru = 0.8
        >>> params = build_parameters(Ru=Ru, Hydrological_model_indicator=0)
        >>> crisis = CrisisComponent(grid, params)
        >>> crisis.run_one_step()
        >>> np.unique(grid.at_node["landslide__failure_id"])
        array([0, 1, 2])
        """

        # Pointers
        self._Elev_Mat = _node_to_2d(self._grid, "topographic__elevation").copy().astype(np.float32)
        self._Elev_Mat[self._Elev_Mat < -900] = np.nan

        self._Slope_Mat = self.get_topographic_slope()
        self._Slope_Mat=self._Slope_Mat[self._grid.nodes]
        self._Area_Mat = self.get_drainage_area()
        self._Area_Mat = self._Area_Mat[self._grid.nodes]

        # ── Time-step counter and bounds check ────────────────────────────────
        # dt is intentionally unused — internal counter drives all file indexing
        # which time step we're about to run
        t = self._current_time_index

        # Reset output
        self._reset_output_fields()

        # refuse to run past the number of time steps the user asked for
        if t >= self._params["time_points"]:
            raise RuntimeError(
                f"run_one_step called {t + 1} times but time_points = "
                f"{self._params['time_points']}."
            )

        # ── Unpack parameters into local variables ────────────────────────────
        # grab shortcuts to params and grid size so the code below is easier to read

        params = self._params
        rows   = self._rows
        clms   = self._clms


        # Set pointers
        Gamma         = self._Gamma
        Gamma_W       = self._Gamma_W
        Cell_size     = self._Cell_size
        Min_LD_area   = self._Min_LD_area
        top_bot       = self._top_bot

        # ── Triggering and strength settings
        Strength_status               = self._Strength_status
        Triggering_Indicator          = self._Triggering_Indicator
        Strength_variation_with_theta = self._Strength_variation_with_theta
        Hydro_indicator               = self._Hydro_indicator
        CRS                           = self._CRS

        # Soil strength constants
        c       = self._c
        Phi     = self._Phi
        Ru      = self._Ru
        Theta_s = self._Theta_s
        Theta_r = self._Theta_r

        # Seismic parameters
        Earthquake_time_point          = self._Earthquake_time_point
        Seismic_displacement_threshold = self._Seismic_displacement_threshold
        Seismic_displacement_model     = self._Seismic_displacement_model
        Rathje_method                  = self._Rathje_method
        Ts                             = self._Ts
        M                              = self._M

        # Load hydrology
        Theta_3D, Pressures_3D = self._calc_hydrology(t, params)

        # ── Identify triggering cells ─────────────────────────────────────────
        # Compute FS and/or Newmark displacement for every cell and depth layer.
        # Returns flat arrays of linear indices, failure depths, and trigger sources.

        # run the core math to find which cells fail this step, and at what depth
        Trig_Cells, Trig_Depths, Trig_sources = _find_triggering_cells_vectorized(
            rows, clms,
            self._Elev_Mat, self._Slope_Mat, self._Depths_vector,
            Hydro_indicator, Pressures_3D, Theta_3D, Ru,
            Triggering_Indicator, Strength_status, c, Phi,
            Theta_s, Theta_r,
            self._Cohesion_3D_array, self._Phi_3D_array,
            Gamma_W, Gamma, t, Earthquake_time_point,
            self._PGA_2D_array,
            Seismic_displacement_threshold, M,
            Strength_variation_with_theta, top_bot,
            Seismic_displacement_model=Seismic_displacement_model,
            PGV_2D_array=self._PGV_2D_array,
            Ts=Ts,
            Rathje_method=Rathje_method,
            Rupture_2D_array=self._Rupture_2D_array,
        )

        # ── Pull cumulative failure arrays from state ─────────────────────────

        # get the running "failed cells so far" grid out of state
        Fail_Index = self._state["Fail_Index_all_times"]
        # get the running landslide-ID grid out of state
        FID        = self._state["FID_all_times"]

        self.LD_Z = []
        self.LD_Index = []
        self.Trig_Index_Accept = []
        # ── Pseudo-3-D BFS landslide delineation ──────────────────────────────
        # Walk upslope from each triggering cell; cells that satisfy the geometric
        # failure criterion form one landslide body each.
        # Walk upslope from each triggering cell; cells that satisfy the geometric
        # failure criterion form one landslide body each.


        if len(Trig_Cells) > 0:
            # walk uphill from each failing cell to build full landslide shapes
            (Vol, Area, NL,
             Trig_Index_Accept, _,
             Fail_Index, Trig_Depth_Accept,
             Trig_source_Accept, FID,
             LD_Z,
             LD_Index,
             FID_Index) = Determine_Pseudo_3D_landslides(
                self._Elev_Mat, Trig_Cells, self._Area_Mat, self._Slope_Mat,
                self._FD_Mat, Cell_size, Trig_Depths, Trig_sources,
                Fail_Index, FID, Min_LD_area,
                upslope_map=self._upslope_map,
            )

            self.LD_Z = LD_Z
            self.LD_Index = LD_Index
            self.Trig_Index_Accept = Trig_Index_Accept
            self._FID_Index = FID_Index
        else:
            # nothing triggered this step — use empty results
            Vol  = np.array([], dtype=np.float32)
            Area = np.array([], dtype=np.float32)
            NL   = 0
            Trig_Index_Accept  = np.array([], dtype=np.int64)
            Trig_Depth_Accept  = []
            Trig_source_Accept = []

        # ── Write updated failure arrays back into state ───────────────────────
        # save the updated failed-cells grid back into state
        self._state["Fail_Index_all_times"] = Fail_Index
        # save the updated landslide-ID grid back into state
        self._state["FID_all_times"]        = FID

        # ── Update cumulative registry ────────────────────────────────────────
        # Merge newly accepted landslides into the running cumulative lists and
        # append per-timestep snapshots for Excel output

        # fold this step's new landslides into the running master list
        self._update_cumulative_registry(
            t, Trig_Index_Accept, Trig_Depth_Accept,
            Trig_source_Accept, Vol, Area, Cell_size,
        )

        # ── Compute summary scalars ───────────────────────────────────────────
        # Aggregate statistics for this time step; stored in pre-allocated vectors

        # if nothing has ever failed, totals are zero
        if np.all(np.isnan(FID)):
            cum_NL, cum_area_total, cum_vol_total = 0, 0.0, 0.0
        else:
            # otherwise compute the running totals: landslide count, area, volume
            cum_NL         = int(np.nanmax(FID))
            cum_area_total = np.nansum(Fail_Index) * Cell_size ** 2
            cum_vol_total  = np.nansum(self._state["Individual_Cumulative_volumes"])

        # store this step's new-landslide area total
        self._state["Total_Current_Area_vector_with_time"][t]      = np.sum(Area)
        # store this step's new-landslide volume total
        self._state["Total_Current_Volume_vector_with_time"][t]    = np.sum(Vol)
        # store this step's new-landslide count
        self._state["Total_Current_NL_vector_with_time"][t]        = NL
        # store the running cumulative area total
        self._state["Total_Cumulative_Area_vector_with_time"][t]   = cum_area_total
        # store the running cumulative landslide count
        self._state["Total_Cumulative_NL_vector_with_time"][t]     = cum_NL
        # store the running cumulative volume total
        self._state["Total_Cumulative_Volume_vector_with_time"][t] = cum_vol_total

        # ── Shapefile export ──────────────────────────────────────────────────
        # One shapefile per time step with all cumulative landslide polygons

        # write out a shapefile of all landslides so far, for this time step
        Modeled_landslides_shapefiles(
            FID, Cell_size, self._x, self._y, t, CRS, str(self._output_dir)
        )

        # ── Update Landlab node fields ────────────────────────────────────────
        # Flatten 2D state arrays back to 1D for the Landlab grid

        # push the updated 2D results back onto the flat Landlab grid fields
        self._update_output_fields()

        # ── Log progress and advance clock ────────────────────────────────────

        # # report progress for this time step
        # self._log(f"  Time step {t}: {NL} new landslides, {cum_NL} cumulative.\n")

        # Internal counter — tracks which time step run_one_step will process next
        # move the internal clock forward by one step
        self._current_time_index += 1

    # =============================================================================
    # Properties
    # =============================================================================

    # quick read-only access to the landslide-ID grid
    @property
    def landslide_fid(self):
        """ndarray of float, shape (nrows, ncols) : Cumulative landslide-ID raster.

        Examples
        --------
        Access the cumulative landslide-ID raster after running one time
        step (same setup as :meth:`run_one_step`).

        >>> import numpy as np
        >>> from landlab import RasterModelGrid
        >>> from landlab.components import FlowAccumulator
        >>> from landlab.components import CrisisComponent
        >>> from landlab.components.crisis import build_parameters
        >>> nrows, ncols = 3, 100
        >>> dx = 1.0  # grid spacing (m)
        >>> grid = RasterModelGrid((nrows, ncols), xy_spacing=dx)
        >>> slope = 0.9  # rise per unit run
        >>> elevation = slope * grid.x_of_node
        >>> _ = grid.add_field("topographic__elevation", elevation, at="node")
        >>> s = grid.add_zeros("topographic__slope", at="node")
        >>> s[:] = np.degrees(grid.calc_slope_at_node("topographic__elevation"))
        >>> grid.set_closed_boundaries_at_grid_edges(True, True, True, True)
        >>> outlet_id = grid.nodes_at_left_edge[1]
        >>> grid.status_at_node[outlet_id] = grid.BC_NODE_IS_FIXED_VALUE
        >>> fa = FlowAccumulator(grid, runoff_rate=1)
        >>> fa.run_one_step()
        >>> params = build_parameters(Ru=0.8, Hydrological_model_indicator=0)
        >>> crisis = CrisisComponent(grid, params)
        >>> crisis.run_one_step()
        >>> crisis.landslide_fid.shape
        (3, 100)
        >>> np.unique(crisis.landslide_fid[~np.isnan(crisis.landslide_fid)])
        array([ 1.,  2.], dtype=float32)
        """
        return self._state["FID_all_times"]

    # quick read-only access to the failed-cells grid
    @property
    def failed_cells(self):
        """ndarray of bool, shape (nrows, ncols) : True where any failure has occurred.

        Examples
        --------
        Access the cumulative failed-cell mask after running one time step
        (same setup as :meth:`run_one_step`).

        >>> import numpy as np
        >>> from landlab import RasterModelGrid
        >>> from landlab.components import FlowAccumulator
        >>> from landlab.components import CrisisComponent
        >>> from landlab.components.crisis import build_parameters
        >>> nrows, ncols = 3, 100
        >>> dx = 1.0  # grid spacing (m)
        >>> grid = RasterModelGrid((nrows, ncols), xy_spacing=dx)
        >>> slope = 0.9  # rise per unit run
        >>> elevation = slope * grid.x_of_node
        >>> _ = grid.add_field("topographic__elevation", elevation, at="node")
        >>> s = grid.add_zeros("topographic__slope", at="node")
        >>> s[:] = np.degrees(grid.calc_slope_at_node("topographic__elevation"))
        >>> grid.set_closed_boundaries_at_grid_edges(True, True, True, True)
        >>> outlet_id = grid.nodes_at_left_edge[1]
        >>> grid.status_at_node[outlet_id] = grid.BC_NODE_IS_FIXED_VALUE
        >>> fa = FlowAccumulator(grid, runoff_rate=1)
        >>> fa.run_one_step()
        >>> params = build_parameters(Ru=0.8, Hydrological_model_indicator=0)
        >>> crisis = CrisisComponent(grid, params)
        >>> crisis.run_one_step()
        >>> crisis.failed_cells.dtype
        dtype('bool')
        >>> int(crisis.failed_cells.sum())
        97
        """
        return self._state["Fail_Index_all_times"]

    # quick read-only access to the entire state dictionary
    @property
    def state(self):
        """dict : Full output accumulator (same structure as ``run_model`` in _crisis_core).

        Examples
        --------
        Access the state accumulator after running one time step
        (same setup as :meth:`run_one_step`).

        >>> import numpy as np
        >>> from landlab import RasterModelGrid
        >>> from landlab.components import FlowAccumulator
        >>> from landlab.components import CrisisComponent
        >>> from landlab.components.crisis import build_parameters
        >>> nrows, ncols = 3, 100
        >>> dx = 1.0  # grid spacing (m)
        >>> grid = RasterModelGrid((nrows, ncols), xy_spacing=dx)
        >>> slope = 0.9  # rise per unit run
        >>> elevation = slope * grid.x_of_node
        >>> _ = grid.add_field("topographic__elevation", elevation, at="node")
        >>> s = grid.add_zeros("topographic__slope", at="node")
        >>> s[:] = np.degrees(grid.calc_slope_at_node("topographic__elevation"))
        >>> grid.set_closed_boundaries_at_grid_edges(True, True, True, True)
        >>> outlet_id = grid.nodes_at_left_edge[1]
        >>> grid.status_at_node[outlet_id] = grid.BC_NODE_IS_FIXED_VALUE
        >>> fa = FlowAccumulator(grid, runoff_rate=1)
        >>> fa.run_one_step()
        >>> params = build_parameters(Ru=0.8, Hydrological_model_indicator=0)
        >>> crisis = CrisisComponent(grid, params)
        >>> crisis.run_one_step()
        >>> sorted(crisis.state.keys())
        ['FID_all_times', 'Fail_Index_all_times', 'Individual_Cumulative_areas', 'Individual_Cumulative_areas_with_time', 'Individual_Cumulative_triggering_depths', 'Individual_Cumulative_triggering_depths_with_time', 'Individual_Cumulative_triggering_linear_indices', 'Individual_Cumulative_triggering_linear_indices_with_time', 'Individual_Cumulative_triggering_sources', 'Individual_Cumulative_triggering_sources_with_time', 'Individual_Cumulative_volumes', 'Individual_Cumulative_volumes_with_time', 'Individual_Current_areas_with_time', 'Individual_Current_triggering_depths_with_time', 'Individual_Current_triggering_linear_indices_with_time', 'Individual_Current_triggering_sources_with_time', 'Individual_Current_volumes_with_time', 'Total_Cumulative_Area_vector_with_time', 'Total_Cumulative_NL_vector_with_time', 'Total_Cumulative_Volume_vector_with_time', 'Total_Current_Area_vector_with_time', 'Total_Current_NL_vector_with_time', 'Total_Current_Volume_vector_with_time']
        >>> crisis.state["Total_Current_NL_vector_with_time"]
        array([ 2.], dtype=float32)
        """
        return self._state

    # quick read-only access to the current time-step counter
    @property
    def current_time_index(self):
        """int : Index of the next time step to be run.

        Examples
        --------
        The counter starts at 0 and advances by one after each call to
        :meth:`run_one_step` (same setup as :meth:`run_one_step`).

        >>> import numpy as np
        >>> from landlab import RasterModelGrid
        >>> from landlab.components import FlowAccumulator
        >>> from landlab.components import CrisisComponent
        >>> from landlab.components.crisis import build_parameters
        >>> nrows, ncols = 3, 100
        >>> dx = 1.0  # grid spacing (m)
        >>> grid = RasterModelGrid((nrows, ncols), xy_spacing=dx)
        >>> slope = 0.9  # rise per unit run
        >>> elevation = slope * grid.x_of_node
        >>> _ = grid.add_field("topographic__elevation", elevation, at="node")
        >>> s = grid.add_zeros("topographic__slope", at="node")
        >>> s[:] = np.degrees(grid.calc_slope_at_node("topographic__elevation"))
        >>> grid.set_closed_boundaries_at_grid_edges(True, True, True, True)
        >>> outlet_id = grid.nodes_at_left_edge[1]
        >>> grid.status_at_node[outlet_id] = grid.BC_NODE_IS_FIXED_VALUE
        >>> fa = FlowAccumulator(grid, runoff_rate=1)
        >>> fa.run_one_step()
        >>> params = build_parameters(Ru=0.8, Hydrological_model_indicator=0)
        >>> crisis = CrisisComponent(grid, params)
        >>> crisis.current_time_index
        0
        >>> crisis.run_one_step()
        >>> crisis.current_time_index
        1
        """
        return self._current_time_index

    # =============================================================================
    # Private helpers
    # =============================================================================

    # ─────────────────────────────────────────────────────────────────────────────
    # _init_output_fields — create output grid fields with default values
    # ─────────────────────────────────────────────────────────────────────────────

    # sets up the two output fields on the grid before any steps are run
    def _init_output_fields(self):
        """Create output node fields on the grid with default values.

        Examples
        --------
        Called automatically by ``__init__``; the output fields already
        exist on the grid by the time :meth:`run_one_step` is called
        (same setup as :meth:`run_one_step`).

        >>> import numpy as np
        >>> from landlab import RasterModelGrid
        >>> from landlab.components import FlowAccumulator
        >>> from landlab.components import CrisisComponent
        >>> from landlab.components.crisis import build_parameters
        >>> nrows, ncols = 3, 100
        >>> dx = 1.0  # grid spacing (m)
        >>> grid = RasterModelGrid((nrows, ncols), xy_spacing=dx)
        >>> slope = 0.9  # rise per unit run
        >>> elevation = slope * grid.x_of_node
        >>> _ = grid.add_field("topographic__elevation", elevation, at="node")
        >>> s = grid.add_zeros("topographic__slope", at="node")
        >>> s[:] = np.degrees(grid.calc_slope_at_node("topographic__elevation"))
        >>> grid.set_closed_boundaries_at_grid_edges(True, True, True, True)
        >>> outlet_id = grid.nodes_at_left_edge[1]
        >>> grid.status_at_node[outlet_id] = grid.BC_NODE_IS_FIXED_VALUE
        >>> fa = FlowAccumulator(grid, runoff_rate=1)
        >>> fa.run_one_step()
        >>> params = build_parameters(Ru=0.8, Hydrological_model_indicator=0)
        >>> crisis = CrisisComponent(grid, params)
        >>> sorted(f for f in grid.at_node.keys()
        ...        if f.startswith("landslide") or f in ("failure__depth", "triggering__cells"))
        ['failure__depth', 'landslide__failed_accumulated', 'landslide__failure_id', 'landslide__failure_id_accumulated', 'triggering__cells']
        """

        # NaN for failure ID (no landslide yet); False for failed flag

        # add the landslide-ID field if it doesn't already exist, default to NaN
        if "landslide__failure_id_accumulated" not in self.grid.at_node:
            self.grid.add_full("landslide__failure_id_accumulated", np.nan, at="node", dtype=float)

        # add the failed flag field if it doesn't already exist, default to False
        if "landslide__failed_accumulated" not in self.grid.at_node:
            self.grid.add_zeros("landslide__failed_accumulated", at="node", dtype=bool)

        # add the failed flag field if it doesn't already exist, default to False
        if "failure__depth" not in self.grid.at_node:
            self.grid.add_zeros("failure__depth", at="node", dtype=float)

        if "triggering__cells" not in self.grid.at_node:
            self.grid.add_zeros("triggering__cells", at="node", dtype=bool)

        if "landslide__failure_id" not in self.grid.at_node:
            self.grid.add_zeros("landslide__failure_id", at="node", dtype=int)

    # ─────────────────────────────────────────────────────────────────────────────
    # _update_output_fields — push 2D results back onto the flat Landlab grid
    # ─────────────────────────────────────────────────────────────────────────────

    # copies our internal 2D results onto the Landlab grid's flat fields
    def _update_output_fields(self):
        """Flatten 2D state arrays to 1D and write them into Landlab node fields.

        Examples
        --------
        Called automatically at the end of :meth:`run_one_step`; the
        accumulated-FID field reflects the state after one time step
        (same setup as :meth:`run_one_step`).

        >>> import numpy as np
        >>> from landlab import RasterModelGrid
        >>> from landlab.components import FlowAccumulator
        >>> from landlab.components import CrisisComponent
        >>> from landlab.components.crisis import build_parameters
        >>> nrows, ncols = 3, 100
        >>> dx = 1.0  # grid spacing (m)
        >>> grid = RasterModelGrid((nrows, ncols), xy_spacing=dx)
        >>> slope = 0.9  # rise per unit run
        >>> elevation = slope * grid.x_of_node
        >>> _ = grid.add_field("topographic__elevation", elevation, at="node")
        >>> s = grid.add_zeros("topographic__slope", at="node")
        >>> s[:] = np.degrees(grid.calc_slope_at_node("topographic__elevation"))
        >>> grid.set_closed_boundaries_at_grid_edges(True, True, True, True)
        >>> outlet_id = grid.nodes_at_left_edge[1]
        >>> grid.status_at_node[outlet_id] = grid.BC_NODE_IS_FIXED_VALUE
        >>> fa = FlowAccumulator(grid, runoff_rate=1)
        >>> fa.run_one_step()
        >>> params = build_parameters(Ru=0.8, Hydrological_model_indicator=0)
        >>> crisis = CrisisComponent(grid, params)
        >>> crisis.run_one_step()
        >>> np.unique(grid.at_node["landslide__failure_id_accumulated"])
        array([  1.,   2.,  nan])
        """

        # flatten the landslide-ID grid and write it into the Landlab field
        self.grid.at_node["landslide__failure_id_accumulated"][:] = self._state["FID_all_times"].ravel()
        # flatten the failed-cells grid and write it into the Landlab field
        self.grid.at_node["landslide__failed_accumulated"][:]     = self._state["Fail_Index_all_times"].ravel()

        if len(self.LD_Z) > 0:
            self._grid.at_node['failure__depth'][self.LD_Index] = self.LD_Z
            self._grid.at_node['triggering__cells'][self.Trig_Index_Accept] = True
            self.grid.at_node["landslide__failure_id"][:] = self._FID_Index.flatten()

    # ─────────────────────────────────────────────────────────────────────────────
    # _load_field_or_h5 — load a 2D array from a grid field or an HDF5 file
    # ─────────────────────────────────────────────────────────────────────────────

    # loads a 2D array either from a Landlab field or, if missing, from an HDF5 file
    def _load_field_or_h5(self, field_name, h5_path):
        """Return a 2D array from a Landlab grid field (preferred) or an HDF5 file.

        Examples
        --------
        Neither a grid field nor a file path is available, so ``None`` is
        returned; once the field is added to the grid, it takes priority
        (same setup as :meth:`run_one_step`).

        >>> import numpy as np
        >>> from landlab import RasterModelGrid
        >>> from landlab.components import FlowAccumulator
        >>> from landlab.components import CrisisComponent
        >>> from landlab.components.crisis import build_parameters
        >>> nrows, ncols = 3, 100
        >>> dx = 1.0  # grid spacing (m)
        >>> grid = RasterModelGrid((nrows, ncols), xy_spacing=dx)
        >>> slope = 0.9  # rise per unit run
        >>> elevation = slope * grid.x_of_node
        >>> _ = grid.add_field("topographic__elevation", elevation, at="node")
        >>> s = grid.add_zeros("topographic__slope", at="node")
        >>> s[:] = np.degrees(grid.calc_slope_at_node("topographic__elevation"))
        >>> grid.set_closed_boundaries_at_grid_edges(True, True, True, True)
        >>> outlet_id = grid.nodes_at_left_edge[1]
        >>> grid.status_at_node[outlet_id] = grid.BC_NODE_IS_FIXED_VALUE
        >>> fa = FlowAccumulator(grid, runoff_rate=1)
        >>> fa.run_one_step()
        >>> params = build_parameters(Ru=0.8, Hydrological_model_indicator=0)
        >>> crisis = CrisisComponent(grid, params)
        >>> crisis._load_field_or_h5("peak_ground_acceleration", "")
        >>> _ = grid.add_full("peak_ground_acceleration", 0.3, at="node")
        >>> crisis._load_field_or_h5("peak_ground_acceleration", "")[0, :3]
        array([ 0.3,  0.3,  0.3], dtype=float32)
        """

        # Grid field takes priority to support Landlab-coupled workflows
        # prefer the live Landlab field if it exists
        if field_name in self.grid.at_node:
            return _node_to_2d(self.grid, field_name).copy().astype(np.float32)

        # otherwise fall back to loading from the given HDF5 file path
        if h5_path and Path(h5_path).exists():
            return load_h5(h5_path)

        # nothing available, so return nothing
        return None

    # ─────────────────────────────────────────────────────────────────────────────
    # _build_fd_mat_from_landlab — convert Landlab routing to ArcGIS D8 codes
    # ─────────────────────────────────────────────────────────────────────────────

    # converts Landlab's flow-routing info into the ArcGIS-style direction codes CRISIS expects
    def _build_fd_mat_from_landlab(self, grid, flow_direction):
        """Convert Landlab receiver-node indices to ArcGIS D8 direction codes.

        ArcGIS D8 codes (same convention as _crisis_core):
        E=1, SE=2, S=4, SW=8, W=16, NW=32, N=64, NE=128.
        Sink / outlet cells (receiver == self) are assigned code 0.

        Parameters
        ----------
        grid : RasterModelGrid
        flow_direction: Flow direction field name
        Returns
        -------
        FD_Mat : ndarray of float32, shape (nrows, ncols)

        Examples
        --------
        On the uniformly inclined 3x100 grid, every interior cell drains
        one step West (D8 code 16) toward the fixed outlet
        (same setup as :meth:`run_one_step`).

        >>> import numpy as np
        >>> from landlab import RasterModelGrid
        >>> from landlab.components import FlowAccumulator
        >>> from landlab.components import CrisisComponent
        >>> from landlab.components.crisis import build_parameters
        >>> nrows, ncols = 3, 100
        >>> dx = 1.0  # grid spacing (m)
        >>> grid = RasterModelGrid((nrows, ncols), xy_spacing=dx)
        >>> slope = 0.9  # rise per unit run
        >>> elevation = slope * grid.x_of_node
        >>> _ = grid.add_field("topographic__elevation", elevation, at="node")
        >>> s = grid.add_zeros("topographic__slope", at="node")
        >>> s[:] = np.degrees(grid.calc_slope_at_node("topographic__elevation"))
        >>> grid.set_closed_boundaries_at_grid_edges(True, True, True, True)
        >>> outlet_id = grid.nodes_at_left_edge[1]
        >>> grid.status_at_node[outlet_id] = grid.BC_NODE_IS_FIXED_VALUE
        >>> fa = FlowAccumulator(grid, runoff_rate=1)
        >>> fa.run_one_step()
        >>> params = build_parameters(Ru=0.8, Hydrological_model_indicator=0)
        >>> crisis = CrisisComponent(grid, params)
        >>> fd_mat = crisis._build_fd_mat_from_landlab(grid, "flow__receiver_node")
        >>> fd_mat.shape
        (3, 100)
        >>> np.unique(fd_mat)
        array([  0.,  16.], dtype=float32)
        >>> fd_mat[1, :5]
        array([  0.,  16.,  16.,  16.,  16.], dtype=float32)
        """

        # ── Setup ─────────────────────────────────────────────────────────────

        # grid dimensions
        nrows, ncols = grid.shape
        # which node each node's water flows into, reshaped to 2D
        receiver     = grid.at_node[flow_direction].reshape(nrows, ncols)
        # output grid of direction codes, starts all zero (no direction / sink)
        FD_Mat       = np.zeros((nrows, ncols), dtype=np.float32)

        # ── Direction code lookup table ───────────────────────────────────────
        # Map (row_offset, col_offset) → ArcGIS D8 code
        # Offset = receiver_position − source_position

        # lookup table: which direction code matches which row/col shift
        offset_to_code = {
            ( 0,  1):   1,   # E
            ( 1,  1):   2,   # SE
            ( 1,  0):   4,   # S
            ( 1, -1):   8,   # SW
            ( 0, -1):  16,   # W
            (-1, -1):  32,   # NW
            (-1,  0):  64,   # N
            (-1,  1): 128,   # NE
        }

        # ── Build flat index arrays ───────────────────────────────────────────
        # Build (row, col) index arrays for every node in the grid

        # build full row/column index grids for every node
        row_idx, col_idx = np.meshgrid(
            np.arange(nrows), np.arange(ncols), indexing="ij"
        )
        # flatten the row indices into a 1D list
        flat_row = row_idx.ravel()
        # flatten the column indices into a 1D list
        flat_col = col_idx.ravel()
        # flatten the receiver-node indices into a 1D list
        flat_rec = receiver.ravel()

        # ── Compute row/col offsets to each receiver ──────────────────────────
        # Convert flat receiver indices to (row, col) and compute offsets

        # turn each receiver's flat index back into a (row, col) pair
        rec_row, rec_col = np.unravel_index(flat_rec, (nrows, ncols))
        # how many rows down the water flows
        dr = rec_row - flat_row
        # how many columns over the water flows
        dc = rec_col - flat_col

        # ── Assign ArcGIS D8 direction codes ─────────────────────────────────
        # Assign D8 code; sink cells with no matching offset get code 0

        # go cell by cell and look up its direction code from the offset table
        for i in range(len(flat_row)):
            FD_Mat[flat_row[i], flat_col[i]] = offset_to_code.get(
                (int(dr[i]), int(dc[i])), 0
            )

        # hand back the finished direction-code grid
        return FD_Mat

    # ─────────────────────────────────────────────────────────────────────────────
    # _calc_hydrology — get the soil moisture and pore-pressure for one time step
    # ─────────────────────────────────────────────────────────────────────────────

    # gathers this time step's soil moisture and pore-pressure data
    def _calc_hydrology(self, t, params):
        """Load Theta_3D and Pressures_3D for time step *t*.

        Loading priority for each array:

        1. HDF5 files in ``hydro_files_dir`` / ``theta_files_dir``.
        2. Landlab node fields ``soil__pressure_head`` /
           ``soil__volumetric_water_content`` (broadcast to all depth layers).
        3. Constants (``Theta_s`` for theta; Ru-derived for pressure).

        Parameters
        ----------
        t : int
            Zero-based time-step index.
        params : dict
            CRISIS parameter dict.

        Returns
        -------
        Theta_3D : ndarray of float32, shape (rows, clms, Depth_points)
        Pressures_3D : ndarray of float32 or None
            None when ``Hydrological_model_indicator == 0``.

        Raises
        ------
        FileNotFoundError
            When variable hydrology is requested but no source is available.

        Examples
        --------
        With ``Strength_variation_with_theta=0`` and
        ``Hydrological_model_indicator=0`` (this example's defaults), Theta_3D
        is filled with the constant ``Theta_s`` and Pressures_3D is ``None``
        (same setup as :meth:`run_one_step`).

        >>> import numpy as np
        >>> from landlab import RasterModelGrid
        >>> from landlab.components import FlowAccumulator
        >>> from landlab.components import CrisisComponent
        >>> from landlab.components.crisis import build_parameters
        >>> nrows, ncols = 3, 100
        >>> dx = 1.0  # grid spacing (m)
        >>> grid = RasterModelGrid((nrows, ncols), xy_spacing=dx)
        >>> slope = 0.9  # rise per unit run
        >>> elevation = slope * grid.x_of_node
        >>> _ = grid.add_field("topographic__elevation", elevation, at="node")
        >>> s = grid.add_zeros("topographic__slope", at="node")
        >>> s[:] = np.degrees(grid.calc_slope_at_node("topographic__elevation"))
        >>> grid.set_closed_boundaries_at_grid_edges(True, True, True, True)
        >>> outlet_id = grid.nodes_at_left_edge[1]
        >>> grid.status_at_node[outlet_id] = grid.BC_NODE_IS_FIXED_VALUE
        >>> fa = FlowAccumulator(grid, runoff_rate=1)
        >>> fa.run_one_step()
        >>> params = build_parameters(Ru=0.8, Hydrological_model_indicator=0)
        >>> crisis = CrisisComponent(grid, params)
        >>> theta_3d, pressures_3d = crisis._calc_hydrology(0, params)
        >>> theta_3d.shape
        (3, 100, 8)
        >>> np.unique(theta_3d)
        array([ 0.45], dtype=float32)
        >>> pressures_3d
        """

        # ── Shortcuts ─────────────────────────────────────────────────────────
        # grid size shortcuts
        rows, clms   = self._rows, self._clms
        # number of depth layers
        Depth_points = params["Depth_points"]
        # saturated water content value
        Theta_s      = params["Theta_s"]
        # constant vs time-varying hydrology flag
        Hydro_ind    = params["Hydrological_model_indicator"]
        # whether strength depends on water content
        Theta_var    = params["Strength_variation_with_theta"]

        # ── File paths ────────────────────────────────────────────────────────

        # folder to look in for pressure-head files
        hydro_dir = Path(params.get("hydro_files_dir") or params["input_dir"])
        # folder to look in for water-content files
        theta_dir = Path(params.get("theta_files_dir") or str(hydro_dir))

        # ── Volumetric water content — Theta_3D ───────────────────────────────
        # Theta_var == 0: fully saturated — fill with Theta_s, no file needed
        # if strength doesn't depend on wetness, just assume fully saturated everywhere
        if Theta_var == 0:
            Theta_3D = np.full((rows, clms, Depth_points), Theta_s, dtype=np.float32)
        else:
            # load water-content HDF5 files from external hydrology model
            theta_h5 = theta_dir / f"Theta_T{t}Z1.h5"
            if theta_h5.exists():
                # load one file per depth layer into a 3D array
                Theta_3D = np.zeros((rows, clms, Depth_points), dtype=np.float32)
                for d in range(Depth_points):
                    with h5py.File(theta_dir / f"Theta_T{t}Z{d + 1}.h5", "r") as f:
                        Theta_3D[:, :, d] = np.flipud(f["data"][:])
            else:
                raise FileNotFoundError(
                    f"Strength_variation_with_theta == 1 but no water-content "
                    f"HDF5 files found in '{theta_dir}' for time step {t}. "
                    f"Expected: Theta_T{t}Z1.h5 ... Theta_T{t}Z{Depth_points}.h5"
                )

        # ── Pore-pressure head — Pressures_3D ────────────────────────────────
        # Only computed when Hydro_indicator == 1 (variable hydrology).
        # Pressure is capped at the layer depth — water table cannot exceed surface.

        # start with no pressure data; only fill it in if time-varying hydrology is on
        Pressures_3D = None

        if Hydro_ind == 1:
            # look for a pressure-head file for this time step
            ph_h5 = hydro_dir / f"Pressure_head_T{t}Z1.h5"
            if ph_h5.exists():
                # load one file per depth layer, capping pressure at that layer's depth
                Pressures_3D = np.zeros((rows, clms, Depth_points), dtype=np.float32)
                for d in range(Depth_points):
                    with h5py.File(
                        hydro_dir / f"Pressure_head_T{t}Z{d + 1}.h5", "r"
                    ) as f:
                        P2D = f["data"][:]
                    Pressures_3D[:, :, d] = np.minimum(P2D, self._Depths_vector[d])

            else:
                raise FileNotFoundError(
                    f"Hydrological_model_indicator == 1 but no pressure-head "
                    f"HDF5 files found in '{hydro_dir}' for time step {t}. "
                    f"Expected: Pressure_head_T{t}Z1.h5 ... Pressure_head_T{t}Z{Depth_points}.h5"
                )

        # ── Return both arrays ────────────────────────────────────────────────

        # hand back both the water-content and pressure arrays
        return Theta_3D, Pressures_3D

    # ─────────────────────────────────────────────────────────────────────────────
    # _compute_rupture_distances — distance from every cell to the fault plane
    # ─────────────────────────────────────────────────────────────────────────────

    # works out how far each cell is from the fault rupture plane
    def _compute_rupture_distances(self, params):
        """Compute closest-distance-to-rupture for Rathje flexible / Bray models.

        Parameters
        ----------
        params : dict

        Returns
        -------
        ndarray of float32, shape (rows, clms)

        Examples
        --------
        On the same 3x100 grid used by :meth:`run_one_step`, place a
        vertical fault plane along x=0 running from y=0 to y=50 m, and
        compute each cell's distance (km) to it.

        >>> import numpy as np
        >>> from landlab import RasterModelGrid
        >>> from landlab.components import FlowAccumulator
        >>> from landlab.components import CrisisComponent
        >>> from landlab.components.crisis import build_parameters
        >>> nrows, ncols = 3, 100
        >>> dx = 1.0  # grid spacing (m)
        >>> grid = RasterModelGrid((nrows, ncols), xy_spacing=dx)
        >>> slope = 0.9  # rise per unit run
        >>> elevation = slope * grid.x_of_node
        >>> _ = grid.add_field("topographic__elevation", elevation, at="node")
        >>> s = grid.add_zeros("topographic__slope", at="node")
        >>> s[:] = np.degrees(grid.calc_slope_at_node("topographic__elevation"))
        >>> grid.set_closed_boundaries_at_grid_edges(True, True, True, True)
        >>> outlet_id = grid.nodes_at_left_edge[1]
        >>> grid.status_at_node[outlet_id] = grid.BC_NODE_IS_FIXED_VALUE
        >>> fa = FlowAccumulator(grid, runoff_rate=1)
        >>> fa.run_one_step()
        >>> params = build_parameters(
        ...     Triggering_Indicator=2, Seismic_displacement_model=3,
        ...     fault_X1=0.0, fault_Y1=0.0, fault_Z1=0.0,
        ...     fault_X2=0.0, fault_Y2=50.0, fault_Z2=0.0,
        ...     fault_X3=0.0, fault_Y3=50.0, fault_Z3=1000.0,
        ...     fault_X4=0.0, fault_Y4=0.0, fault_Z4=1000.0,
        ...     pga_file="",
        ... )
        >>> crisis = CrisisComponent(grid, params)
        Computing fault rupture distances...
        <BLANKLINE>
        >>> rupture = crisis._compute_rupture_distances(params)
        Computing fault rupture distances...
        <BLANKLINE>
        >>> rupture.shape
        (3, 100)
        >>> np.round(rupture[:, :3], 3)
        array([[ 0.099,  0.098,  0.097],
               [ 0.099,  0.098,  0.097],
               [ 0.099,  0.098,  0.097]], dtype=float32)
        """

        # ── Define fault plane corners ────────────────────────────────────────
        # 4-corner fault plane; Z values are depths (positive downward)

        # the four corners of the fault plane, as x/y/z points
        corners = np.array([
            [params.get("fault_X1", 0.0), params.get("fault_Y1", 0.0), params.get("fault_Z1", 0.0)],
            [params.get("fault_X2", 0.0), params.get("fault_Y2", 0.0), params.get("fault_Z2", 0.0)],
            [params.get("fault_X3", 0.0), params.get("fault_Y3", 0.0), params.get("fault_Z3", 0.0)],
            [params.get("fault_X4", 0.0), params.get("fault_Y4", 0.0), params.get("fault_Z4", 0.0)],
        ], dtype=np.float64)

        # ── Load site coordinates at high precision ───────────────────────────
        # Use float64 to avoid precision loss with large UTM coordinate values

        # reload coordinates at float64 precision for accurate UTM arithmetic
        _x = load_h5(params["x_coordinates_file"]).astype(np.float64)
        _y = load_h5(params["y_coordinates_file"]).astype(np.float64)

        # ── Prepare 2D coordinate and elevation grids ─────────────────────────

        # elevation grid at high precision
        _elev        = self._Elev_Mat.astype(np.float64)
        # grid size shortcuts
        _rows, _clms = _elev.shape

        # Broadcast 1D coordinate vectors to 2D grids if needed
        # stretch the 1D x coordinates into a full 2D grid if needed
        _x2d = np.broadcast_to(_x[np.newaxis, :], (_rows, _clms)) if _x.ndim == 1 else _x
        # stretch the 1D y coordinates into a full 2D grid if needed
        _y2d = np.broadcast_to(_y[:, np.newaxis], (_rows, _clms)) if _y.ndim == 1 else _y

        # Replace NaN elevations with 0 (sea level) for distance calculation
        # fill missing elevation values with sea level so the math doesn't break
        _z2d = np.where(np.isnan(_elev), 0.0, _elev)

        # ── Compute and return rupture distances ──────────────────────────────

        # let the user know this (possibly slow) calculation is starting
        self._log("Computing fault rupture distances...\n")
        # compute and return the distance from every cell to the fault plane
        return Rupture_Dist(_x2d, _y2d, _z2d, corners).astype(np.float32)

    # ─────────────────────────────────────────────────────────────────────────────
    # _update_cumulative_registry — merge new landslides into the master records
    # ─────────────────────────────────────────────────────────────────────────────

    # adds this time step's newly accepted landslides into the running master records
    def _update_cumulative_registry(self, t, Trig_Index_Accept, Trig_Depth_Accept,
                                    Trig_source_Accept, Vol, Area, Cell_size):
        """Merge newly accepted landslides into the cumulative tracking registry.

        Parameters
        ----------
        t : int
            Current time-step index.
        Trig_Index_Accept : ndarray of int64
        Trig_Depth_Accept : list of float
        Trig_source_Accept : list of int
        Vol : ndarray of float32
        Area : ndarray of float32
        Cell_size : float

        Examples
        --------
        Re-applying the same triggering cells from time step 0 at time
        step 1 leaves the cumulative registry unchanged, since duplicate
        cell indices are deduplicated (same setup as :meth:`run_one_step`).

        >>> import numpy as np
        >>> from landlab import RasterModelGrid
        >>> from landlab.components import FlowAccumulator
        >>> from landlab.components import CrisisComponent
        >>> from landlab.components.crisis import build_parameters
        >>> nrows, ncols = 3, 100
        >>> dx = 1.0  # grid spacing (m)
        >>> grid = RasterModelGrid((nrows, ncols), xy_spacing=dx)
        >>> slope = 0.9  # rise per unit run
        >>> elevation = slope * grid.x_of_node
        >>> _ = grid.add_field("topographic__elevation", elevation, at="node")
        >>> s = grid.add_zeros("topographic__slope", at="node")
        >>> s[:] = np.degrees(grid.calc_slope_at_node("topographic__elevation"))
        >>> grid.set_closed_boundaries_at_grid_edges(True, True, True, True)
        >>> outlet_id = grid.nodes_at_left_edge[1]
        >>> grid.status_at_node[outlet_id] = grid.BC_NODE_IS_FIXED_VALUE
        >>> fa = FlowAccumulator(grid, runoff_rate=1)
        >>> fa.run_one_step()
        >>> params = build_parameters(Ru=0.8, Hydrological_model_indicator=0, time_points=2)
        >>> crisis = CrisisComponent(grid, params)
        >>> crisis.run_one_step()
        >>> sorted(crisis.state["Individual_Cumulative_triggering_linear_indices"].tolist())
        [100, 176]
        >>> Trig_Index_Accept  = crisis.Trig_Index_Accept
        >>> Trig_Depth_Accept  = crisis.state["Individual_Current_triggering_depths_with_time"][-1]
        >>> Trig_source_Accept = crisis.state["Individual_Current_triggering_sources_with_time"][-1]
        >>> Vol  = crisis.state["Individual_Current_volumes_with_time"][-1]
        >>> Area = crisis.state["Individual_Current_areas_with_time"][-1]
        >>> crisis._update_cumulative_registry(1, Trig_Index_Accept, Trig_Depth_Accept, Trig_source_Accept, Vol, Area, dx)
        >>> sorted(crisis.state["Individual_Cumulative_triggering_linear_indices"].tolist())
        [100, 176]
        """

        # ── Shortcuts ─────────────────────────────────────────────────────────

        # current landslide-ID grid
        FID   = self._state["FID_all_times"]
        # shortcut to the state dictionary
        state = self._state

        # the master list of triggering cell locations seen so far
        cum_idx = state["Individual_Cumulative_triggering_linear_indices"]

        # ── Merge new landslides into the running registry ────────────────────
        # First time step: initialise registry directly.
        # Subsequent steps: concatenate and deduplicate — np.unique keeps the first
        # occurrence of each linear index so re-triggered cells are not double-counted.

        # on the very first step, just start the registry with this step's results
        if len(cum_idx) == 0:
            state["Individual_Cumulative_triggering_linear_indices"] = Trig_Index_Accept.copy()
            state["Individual_Cumulative_triggering_depths"]         = list(Trig_Depth_Accept)
            state["Individual_Cumulative_triggering_sources"]        = list(Trig_source_Accept)
            state["Individual_Cumulative_volumes"]                   = Vol.copy()

        elif len(Trig_Index_Accept) > 0:
            # later steps: merge old and new data together
            merged_idx = np.concatenate((cum_idx, Trig_Index_Accept))
            merged_dep = np.concatenate((state["Individual_Cumulative_triggering_depths"],
                                         Trig_Depth_Accept))
            merged_src = np.concatenate((state["Individual_Cumulative_triggering_sources"],
                                         Trig_source_Accept))
            merged_vol = np.concatenate((state["Individual_Cumulative_volumes"], Vol))

            # drop duplicate cells, keeping only the first time each one appeared
            _, unique_idx = np.unique(merged_idx, return_index=True)
            state["Individual_Cumulative_triggering_linear_indices"] = merged_idx[unique_idx]
            state["Individual_Cumulative_triggering_depths"]         = merged_dep[unique_idx]
            state["Individual_Cumulative_triggering_sources"]        = merged_src[unique_idx]
            state["Individual_Cumulative_volumes"]                   = merged_vol[unique_idx]

        # ── Recompute cumulative areas from the FID raster ────────────────────
        # Recompute cumulative areas from the FID raster (delineation may adjust
        # body geometry across steps, so summing per-step areas would drift)

        # grab the up-to-date master list of triggering cells
        cum_trig = state["Individual_Cumulative_triggering_linear_indices"]

        if len(cum_trig) > 0:
            # only look at cells that actually belong to a landslide
            valid_fid = FID[~np.isnan(FID)]
            # count how many cells belong to each landslide ID
            fids_uniq, counts = np.unique(valid_fid, return_counts=True)
            # turn that cell count into a real area in square meters, per landslide ID
            fid_to_area = {int(f): cnt * Cell_size ** 2 for f, cnt in zip(fids_uniq, counts)}
            # look up which landslide ID each triggering cell belongs to
            fid_of_trig = FID.flat[cum_trig]
            # look up the area for each triggering cell's landslide
            cum_areas   = np.array([fid_to_area.get(int(f), 0.0) for f in fid_of_trig],
                                   dtype=np.float32)
        else:
            # no triggering cells yet, so no areas to report
            cum_areas = np.array([], dtype=np.float32)

        # save the freshly computed cumulative areas
        state["Individual_Cumulative_areas"] = cum_areas

        # ── Append per-timestep snapshots for Excel output ────────────────────
        # one entry per column in the final Excel workbook

        # record this step's new triggering cell locations
        state["Individual_Current_triggering_linear_indices_with_time"].append(Trig_Index_Accept)
        # record this step's new failure depths
        state["Individual_Current_triggering_depths_with_time"].append(Trig_Depth_Accept)
        # record this step's new trigger sources
        state["Individual_Current_triggering_sources_with_time"].append(Trig_source_Accept)
        # record this step's new landslide areas
        state["Individual_Current_areas_with_time"].append(Area)
        # record this step's new landslide volumes
        state["Individual_Current_volumes_with_time"].append(Vol)

        # save a snapshot of the full master list of triggering cells, for this step
        state["Individual_Cumulative_triggering_linear_indices_with_time"].append(
            state["Individual_Cumulative_triggering_linear_indices"].copy())
        # save a snapshot of the full master list of failure depths, for this step
        state["Individual_Cumulative_triggering_depths_with_time"].append(
            state["Individual_Cumulative_triggering_depths"].copy())
        # save a snapshot of the full master list of trigger sources, for this step
        state["Individual_Cumulative_triggering_sources_with_time"].append(
            state["Individual_Cumulative_triggering_sources"].copy())
        # save a snapshot of the cumulative areas, for this step
        state["Individual_Cumulative_areas_with_time"].append(cum_areas)
        # save a snapshot of the full master list of volumes, for this step
        state["Individual_Cumulative_volumes_with_time"].append(
            state["Individual_Cumulative_volumes"].copy())

    def get_topographic_slope(self,
                              slope='topographic__slope'):
        """Return the slope array (degrees), preferring the live grid field.

        Parameters
        ----------
        slope : str, optional
            Name of the Landlab node field to read the slope from, if
            present. Defaults to ``"topographic__slope"``.

        Returns
        -------
        ndarray of float, shape (n_nodes,)

        Examples
        --------
        The grid already carries a ``topographic__slope`` field, so it is
        returned directly (same setup as :meth:`run_one_step`).

        >>> import numpy as np
        >>> from landlab import RasterModelGrid
        >>> from landlab.components import FlowAccumulator
        >>> from landlab.components import CrisisComponent
        >>> from landlab.components.crisis import build_parameters
        >>> nrows, ncols = 3, 100
        >>> dx = 1.0  # grid spacing (m)
        >>> grid = RasterModelGrid((nrows, ncols), xy_spacing=dx)
        >>> slope = 0.9  # rise per unit run
        >>> elevation = slope * grid.x_of_node
        >>> _ = grid.add_field("topographic__elevation", elevation, at="node")
        >>> s = grid.add_zeros("topographic__slope", at="node")
        >>> s[:] = np.degrees(grid.calc_slope_at_node("topographic__elevation"))
        >>> grid.set_closed_boundaries_at_grid_edges(True, True, True, True)
        >>> outlet_id = grid.nodes_at_left_edge[1]
        >>> grid.status_at_node[outlet_id] = grid.BC_NODE_IS_FIXED_VALUE
        >>> fa = FlowAccumulator(grid, runoff_rate=1)
        >>> fa.run_one_step()
        >>> params = build_parameters(Ru=0.8, Hydrological_model_indicator=0)
        >>> crisis = CrisisComponent(grid, params)
        >>> crisis.get_topographic_slope()[:3]
        array([ 41.9872125,  41.9872125,  41.9872125])
        """

        if self._grid.has_field(slope, at='node'):
            slope_mat = self._grid.at_node[slope]
            slope_mat[slope_mat < -900] = np.nan
        else: # else: slope file loaded and stored at the initialization

            slope_mat = self._Slope_Mat

        return slope_mat

    def get_drainage_area(self,
                              drainage_area='drainage_area'):
        """Return the drainage area array (m^2), preferring the live grid field.

        Parameters
        ----------
        drainage_area : str, optional
            Name of the Landlab node field to read the drainage area from,
            if present. Defaults to ``"drainage_area"``.

        Returns
        -------
        ndarray of float, shape (n_nodes,)

        Examples
        --------
        The grid already carries a ``drainage_area`` field (created by
        ``FlowAccumulator``), so it is returned directly
        (same setup as :meth:`run_one_step`).

        >>> import numpy as np
        >>> from landlab import RasterModelGrid
        >>> from landlab.components import FlowAccumulator
        >>> from landlab.components import CrisisComponent
        >>> from landlab.components.crisis import build_parameters
        >>> nrows, ncols = 3, 100
        >>> dx = 1.0  # grid spacing (m)
        >>> grid = RasterModelGrid((nrows, ncols), xy_spacing=dx)
        >>> slope = 0.9  # rise per unit run
        >>> elevation = slope * grid.x_of_node
        >>> _ = grid.add_field("topographic__elevation", elevation, at="node")
        >>> s = grid.add_zeros("topographic__slope", at="node")
        >>> s[:] = np.degrees(grid.calc_slope_at_node("topographic__elevation"))
        >>> grid.set_closed_boundaries_at_grid_edges(True, True, True, True)
        >>> outlet_id = grid.nodes_at_left_edge[1]
        >>> grid.status_at_node[outlet_id] = grid.BC_NODE_IS_FIXED_VALUE
        >>> fa = FlowAccumulator(grid, runoff_rate=1)
        >>> fa.run_one_step()
        >>> params = build_parameters(Ru=0.8, Hydrological_model_indicator=0)
        >>> crisis = CrisisComponent(grid, params)
        >>> crisis.get_drainage_area()[:3]
        array([ 0.,  0.,  0.])
        """

        if self._grid.has_field(drainage_area, at='node'):
            Area_mat = self._grid.at_node[drainage_area]
            Area_mat[Area_mat < -900] = np.nan
        else: # else: drainage area file loaded and stored at the initialization
            Area_mat = self._Area_Mat

        return Area_mat

    def _reset_output_fields(self):
        """Clear the per-time-step output fields back to their default values.

        Called at the start of every :meth:`run_one_step` call so that
        ``triggering__cells``, ``failure__depth``, and
        ``landslide__failure_id`` reflect only the current time step, not
        cells triggered in a previous one.

        Examples
        --------
        After one time step, ``triggering__cells`` has some ``True``
        entries; resetting clears them back to their defaults
        (same setup as :meth:`run_one_step`).

        >>> import numpy as np
        >>> from landlab import RasterModelGrid
        >>> from landlab.components import FlowAccumulator
        >>> from landlab.components import CrisisComponent
        >>> from landlab.components.crisis import build_parameters
        >>> nrows, ncols = 3, 100
        >>> dx = 1.0  # grid spacing (m)
        >>> grid = RasterModelGrid((nrows, ncols), xy_spacing=dx)
        >>> slope = 0.9  # rise per unit run
        >>> elevation = slope * grid.x_of_node
        >>> _ = grid.add_field("topographic__elevation", elevation, at="node")
        >>> s = grid.add_zeros("topographic__slope", at="node")
        >>> s[:] = np.degrees(grid.calc_slope_at_node("topographic__elevation"))
        >>> grid.set_closed_boundaries_at_grid_edges(True, True, True, True)
        >>> outlet_id = grid.nodes_at_left_edge[1]
        >>> grid.status_at_node[outlet_id] = grid.BC_NODE_IS_FIXED_VALUE
        >>> fa = FlowAccumulator(grid, runoff_rate=1)
        >>> fa.run_one_step()
        >>> params = build_parameters(Ru=0.8, Hydrological_model_indicator=0)
        >>> crisis = CrisisComponent(grid, params)
        >>> crisis.run_one_step()
        >>> crisis._reset_output_fields()
        >>> grid.at_node["triggering__cells"].any()
        False
        >>> grid.at_node["failure__depth"].sum()
        0.0
        >>> grid.at_node["landslide__failure_id"].sum()
        0
        """

        # Reset output
        self._grid.at_node['triggering__cells'][:] = False
        self._grid.at_node['failure__depth'].fill(0.0)
        self._grid.at_node["landslide__failure_id"].fill(0)