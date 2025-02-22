#!/usr/bin/env python3
"""
Model bedrock incision and gravel transport and abrasion in a network of rivers.

@author: gtucker
"""

import numpy as np
from landlab import Component
from landlab import HexModelGrid
from landlab.grid.diagonals import DiagonalsMixIn

use_cfuncs = False
if use_cfuncs:
    from .cfuncs import _calc_sediment_influx
    from .cfuncs import _calc_sediment_rate_of_change
    from .cfuncs import _estimate_max_time_step_size_ext

_DT_MAX = 1.0e-2
_ONE_SIXTH = 1.0 / 6.0
_SEVEN_SIXTHS = 7.0 / 6.0
_D8_CHAN_LENGTH_FACTOR = 1.0  # 0.5 * (1.0 + 2.0**0.5)  # D8 raster: average of straight and diagonal
_SEC_PER_YEAR = 365.25 * 24.0 * 3600.0
_EARTH_GRAV = 9.81


def _calc_chan_width_fixed_width(coeff, expt, discharge, out=None):
    """Calculate channel width using empirical formula.

    b = coeff * Q**expt

    Parameters
    ----------
    coeff : float
        Empirical coefficient
    expt : float
        Empirical exponent
    discharge : array
        Discharge
    out : array, optional
        Array to store output

    Returns
    -------
    array
        Channel width

    Examples
    --------
    >>> calc_chan_width_fixed_width(1.0, 0.5, np.array([1.0]))
    1.0
    >>> calc_chan_width_fixed_width(6e-8, 0.5, np.array([100.0]))
    6e-7
    """

    if out is None:
        out = np.empty_like(discharge)
    out[:] = coeff * discharge ** expt
    return out


def _calc_shear_stress_coef(rho_w, mannings_n, g=_EARTH_GRAV):
    """
    Calculate prefactor for:
    tau = rho*g*n^(3/5)(Q...)

    so rho*g*n^(3/5) is the prefactor.

    Examples
    --------
    >>> round(calc_shear_stress_coef(1000, _EARTH_GRAV, 0.05))
    1626. 
    """
    return rho_w * g * mannings_n ** (.6)


def _calc_shear_stress(shear_stress_coef, discharge, width, slope, out=None):
    """Calculate shear stress using Manning's equation.
    """

    if out is None:
        out = np.empty_like(discharge)
    out[:] = shear_stress_coef * np.divide(discharge,
                                           width,
                                           where=width > 0,
                                           out=np.zeros_like(out)) ** (.6) * slope ** (.7)
    return out


def _tau_crit_fixed_width(rho_sed, rho_w, grain_size, tau_star_crit, g=_EARTH_GRAV, out=None):
    """Calculate transport rate using Meyer-Peter Mueller.
    Parameters
    ----------
    rho_sed : float
        Density of sediment
    rho_w : float
        Density of water
    grain_size : float
        Grain size
    tau_star_crit : array
        Dimensionless critical shear stress
    g : float, optional
        Acceleration due to gravity
    out : array, optional
        Array to store output


    """

    if out is None:
        out = np.empty_like(tau_star_crit)

    out[:] = tau_star_crit * (rho_sed - rho_w) * g * grain_size
    return out


def _trans_rate_coeff_fixed_width(mpm, rho_sed, rho_w, g=_EARTH_GRAV):
    """Calculate transport rate coefficient to use in calc_transport_rate_fixed_width"""

    a = _SEC_PER_YEAR * mpm
    b = (rho_w ** 0.5) * (rho_sed - rho_w) * g

    return a / b


# transport rate function
def calc_transport_rate_fixed_width(trans_rate_coeff, tau_crit, width, tau, out=None):
    """Calculate transport rate using methods of Wickert & Schildgen (2019), 
    but using a the empirical formula for channel width and shear stress.
    """

    if out is None:
        out = np.empty_like(tau_crit)

    out[:] = trans_rate_coeff * ((tau[:, np.newaxis] - tau_crit) ** 1.5) * width
    return out


# plucking rate function
def calc_plucking_rate_fixed_width(coeff_pluck, width_chan, width_val, tau, tau_crit, out=None):
    """ see Overleaf document for description; SMM """

    if out is None:
        out = np.empty_like(tau_crit)

    ratio = np.divide(width_chan,
                      width_val,
                      where=width_val > 0,
                      out=np.zeros_like(width_chan))

    out[:] = coeff_pluck * (tau - tau_crit) ** 1.5 * ratio
    return out


class GravelBedrockEroder(Component):
    """Drainage network evolution of rivers with gravel alluvium overlying bedrock.

    Model drainage network evolution for a network of rivers that have
    a layer of gravel alluvium overlying bedrock.

    :class:`~.GravelBedrockEroder` is designed to operate together with a flow-routing
    component such as :class:`~.FlowAccumulator`, so that each grid node has
    a defined flow direction toward one of its neighbor nodes. Each core node
    is assumed to contain one outgoing fluvial channel, and (depending on
    the drainage structure) zero, one, or more incoming channels. These channels are
    treated as effectively sub-grid-scale features that are embedded in valleys
    that have a width of one grid cell.

    As with the :class:`~.GravelRiverTransporter` component, the rate of gravel
    transport out of a given node is calculated as the product of bankfull discharge,
    channel gradient (to the 7/6 power), a dimensionless transport coefficient, and
    an intermittency factor that represents the fraction of time that bankfull
    flow occurs. The derivation of the transport law is given by Wickert &
    Schildgen (2019), and it derives from the assumption that channels are
    gravel-bedded and that they "instantaneously" adjust their width such that
    bankfull bed shear stress is just slightly higher than the threshold for
    grain motion. The substrate is assumed to consist entirely of gravel-size
    material with a given bulk porosity. The component calculates the loss of
    gravel-sized material to abrasion (i.e., conversion to finer sediment, which
    is not explicitly tracked) as a function of the volumetric transport rate,
    an abrasion coefficient with units of inverse length, and the local transport
    distance (for example, if a grid node is carrying a gravel load ``Qs`` to a
    neighboring node ``dx`` meters downstream, the rate of gravel loss in volume per
    time per area at the node will be ``beta * Qs * dx``, where ``beta`` is the abrasion
    coefficient).

    Sediment mass conservation is calculated across each entire
    grid cell. For example, if a cell has surface area ``A``, a total volume influx
    ``Qin``, and downstream transport rate ``Qs``, the resulting rate of change of
    alluvium thickness will be ``(Qin - Qs / (A * (1 - phi))``, plus gravel produced by
    plucking erosion of bedrock (``phi`` is porosity).

    Bedrock is eroded by a combination of abrasion and plucking. Abrasion per unit
    channel length is calculated as the product of volumetric sediment discharge
    and an abrasion coefficient. Sediment produced by abrasion is assumed to
    go into wash load that is removed from the model domain. Plucking is calculated
    using a discharge-slope expression, and a user-defined fraction of plucked
    material is added to the coarse alluvium.

    Parameters
    ----------
    grid : ModelGrid
        A Landlab model grid object
    intermittency_factor : float (default 0.01)
        Fraction of time that bankfull flow occurs
    transport_coefficient : float (default 0.041)
        if near-threshold:
            Dimensionless transport efficiency factor; see Wickert & Schildgen 2019
        if empirical width:
            See Meyer-Peter Mueller in Wong & Parker 2006
    abrasion_coefficient : float (default 0.0 1/m) *DEPRECATED*
        Abrasion coefficient with units of inverse length
    sediment_porosity : float (default 0.35)
        Bulk porosity of bed sediment
    depth_decay_scale : float (default 1.0)
        Scale for depth decay in bedrock exposure function
    plucking_coefficient : float or (n_core_nodes,) array of float (default 1.0e-4 1/m)
        Rate coefficient for bedrock erosion by plucking
        if near-threshold:
            See Gabel et al. 2024
        if empirical width:
            See [insert description here, see Overleaf doc for details; SMM]
    self._n_classes : int (default 1)
        Number of sediment abradability classes
    init_thickness_per_class : float or (n_core_nodes,) array of float (default 1 / n-classes)
        Starting thickness for each sediment fraction
    abrasion_coefficients : iterable containing floats (default 0.0 1/m)
        Abrasion coefficients; should be same length as number of sed classes
    bedrock_abrasion_coefficients : float
        Abrasion coefficient for bedrock
    coarse_fractions_from_plucking : float or (n_core_nodes,) array of float (default 1.0)
        Fraction(s) of plucked material that becomes part of gravel sediment load
    rock_abrasion_index : int (default 0)
        If multiple classes, specifies which contains the abrasion
        coefficient for bedrock
    --The following are necessary only if using the empirical width calculation--
    tau_star_crit : float (default 0.045)
        Dimensionless critical shear stress; 
    grav_accel : float (default to Earth, 9.81)
        Acceleration due to gravity; 
    width_coeff : float (default 2?)
        Empirical coefficient for channel width; 
        NOTE: units, if exponent is 1/2: 1/(m^0.5*s^0.5) -- need to make this conversion to years, not seconds
    width_expt : float (default )
        Empirical exponent for channel width;
    rho_w : float (default 1000)
        Density of water;
    rho_sed : float (default 2650)
        Density of sediment;
    D_50 : float (default 0.01)
        Median grain size;


    Notes
    -----
    The doctest below demonstrates approximate equilibrium between uplift, transport,
    and sediment abrasion in a case with effectively unlimited sediment. The
    analytical solution is:

    sediment input by uplift = sediment outflux + sediment loss to abrasion

    In math,

    U A = kq I Q S^(7/6) + 0.5 b Qs dx

    S = (U A / (kq I Q (1 + 0.5 b dx))) ^ 6/7

    S = (1.0e-4 1e6 / (0.041 0.01 10.0 1e6 (1 + 0.5 0.0005 1000.0)))^(6 / 7)

    ~ 0.0342

    The sediment abrasion rate should be, in volume per time, as follows:

    Qsout + abrasion loss rate = generation rate

    Qsout + 0.5 Qsout b dx = U dx^2

    Qsout = U dx^2 / (1 + 0.5 b dx)

    Qsout = 0.0001 1e6 / (1 + 0.5 0.0005 1e3)

    Qsout = 1e2 / 1.25 = 80

    Elevation of the single core node = 0.0342 x 1,000 m ~ 34.2 m.
    However, because of VERY long time steps, the post-erosion elevation is 1 m
    lower, at 33.2 m (it will be uplifted by a meter at the start of each step).

    Examples
    --------
    >>> from landlab import RasterModelGrid
    >>> from landlab.components import FlowAccumulator
    >>> grid = RasterModelGrid((3, 3), xy_spacing=1000.0)
    >>> elev = grid.add_zeros("topographic__elevation", at="node")
    >>> elev[4] = 1.0
    >>> sed = grid.add_zeros("soil__depth", at="node")
    >>> sed[4] = 1.0e6
    >>> grid.status_at_node[grid.perimeter_nodes] = grid.BC_NODE_IS_CLOSED
    >>> grid.status_at_node[5] = grid.BC_NODE_IS_FIXED_VALUE
    >>> fa = FlowAccumulator(grid, runoff_rate=10.0)
    >>> fa.run_one_step()
    >>> eroder = GravelBedrockEroder(
    ...     grid, sediment_porosity=0.0, abrasion_coefficients=[0.0005]
    ... )
    >>> rock_elev = grid.at_node["bedrock__elevation"]
    >>> fa.run_one_step()
    >>> dt = 10000.0
    >>> for _ in range(200):
    ...     rock_elev[grid.core_nodes] += 1.0e-4 * dt
    ...     elev[grid.core_nodes] += 1.0e-4 * dt
    ...     eroder.run_one_step(dt)
    ...
    >>> int(elev[4])
    33
    """

    _name = "GravelBedrockEroder"

    _unit_agnostic = True

    _info = {
        "bedload_sediment__rate_of_loss_to_abrasion": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m/y",
            "mapping": "node",
            "doc": "Rate of bedload sediment volume loss to abrasion per unit area",
        },
        "bedload_sediment__volume_influx": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m**3/y",
            "mapping": "node",
            "doc": "Volumetric incoming streamwise bedload sediment transport rate",
        },
        "bedload_sediment__volume_outflux": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m**3/y",
            "mapping": "node",
            "doc": "Volumetric outgoing streamwise bedload sediment transport rate",
        },
        "bedrock__abrasion_rate": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m/y",
            "mapping": "node",
            "doc": "rate of bedrock lowering by abrasion",
        },
        "bedrock__elevation": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "elevation of the bedrock surface",
        },
        "bedrock__exposure_fraction": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "fractional exposure of bedrock",
        },
        "bedrock__plucking_rate": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m/y",
            "mapping": "node",
            "doc": "rate of bedrock lowering by plucking",
        },
        "bedrock__lowering_rate": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m/y",
            "mapping": "node",
            "doc": "Rate of lowering of bedrock surface",
        },
        "flow__link_to_receiver_node": {
            "dtype": int,
            "intent": "in",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "ID of link downstream of each node, which carries the discharge",
        },
        "flow__receiver_node": {
            "dtype": int,
            "intent": "in",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "Node array of receivers (node that receives flow from current node)",
        },
        "flow__upstream_node_order": {
            "dtype": int,
            "intent": "in",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "Node array containing downstream-to-upstream ordered list of node IDs",
        },
        "sediment__rate_of_change": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m/y",
            "mapping": "node",
            "doc": "Time rate of change of sediment thickness",
        },
        "soil__depth": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Depth of soil or weathered bedrock",
        },
        "surface_water__discharge": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "m**3/y",
            "mapping": "node",
            "doc": "Volumetric discharge of surface water",
        },
        "topographic__elevation": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Land surface topographic elevation",
        },
        "topographic__steepest_slope": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "The steepest *downhill* slope",
        },
        "grains__weight": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "kg",
            "mapping": "node",
            "doc": "",
        },
    }

    def __init__(
            self,
            grid,
            intermittency_factor=0.01,
            transport_coefficient=0.041,
            sediment_porosity=0.35,
            depth_decay_scale=1.0,
            plucking_coefficient=1.0e-4,
            abrasion_coefficients=0.0,
            bedrock_abrasion_coefficient=0.01,
            fractions_from_plucking=1.0,
            rock_abrasion_index=0,
            rho_sed = 2650
    ):

        super().__init__(grid)
        super().initialize_output_fields()

        # Get the number of classes from SoilGrading component
        # In this version the number of classes can represent EITHER
        # lithology OR grain sizes
        self._n_classes = np.shape(self._grid.at_node['grains__weight'])[1]

        # Create 2D arrays for input variables
        # The component will also check the validity of
        # the user-defined variable
        self._abr_coefs = self._create_2D_array_for_input_var(
            abrasion_coefficients,
            'abrasion_coefficients')
        self._fractions_from_plucking = self._create_2D_array_for_input_var(
            fractions_from_plucking,
            'fractions_from_plucking')
        if ~np.all(np.sum(self._fractions_from_plucking,1)==1):
            raise ValueError("The sum of fractions from plucking should be equal to 1")
        self._plucking_coef = self._create_2D_array_for_input_var(
            plucking_coefficient,
            'plucking_coefficient')
        self._br_abr_coef = self._create_2D_array_for_input_var(
            bedrock_abrasion_coefficient,
            'bedrock_abrasion_coefficient')

        # Recognize whether the component deals with lithologies or grain sizes
        self._get_classes_identity()

        # Non-array parameters
        self._trans_coef = transport_coefficient
        self._intermittency_factor = intermittency_factor
        self._sediment_porosity = sediment_porosity
        self._porosity_factor = 1.0 / (1.0 - self._sediment_porosity)
        self._depth_decay_scale = depth_decay_scale
        self._rock_abrasion_index = rock_abrasion_index
        self._rho_sed = rho_sed

        # Pointers to field
        self._elev = grid.at_node["topographic__elevation"]
        self._sed = grid.at_node["soil__depth"]
        if "bedrock__elevation" in grid.at_node:
            self._bedrock__elevation = grid.at_node["bedrock__elevation"]
        else:
            self._bedrock__elevation = grid.add_zeros(
                "bedrock__elevation", at="node", dtype=float
            )
            self._bedrock__elevation[:] = self._elev - self._sed
        self._discharge = grid.at_node["surface_water__discharge"]
        self._slope = grid.at_node["topographic__steepest_slope"]
        self._receiver_node = grid.at_node["flow__receiver_node"]
        self._receiver_link = grid.at_node["flow__link_to_receiver_node"]
        self._sediment_influx = grid.at_node["bedload_sediment__volume_influx"]
        self._sediment_outflux = grid.at_node["bedload_sediment__volume_outflux"]
        self._dHdt = grid.at_node["sediment__rate_of_change"]
        self._rock_lowering_rate = grid.at_node["bedrock__lowering_rate"]
        self._rock_exposure_fraction = grid.at_node["bedrock__exposure_fraction"]
        self._rock_abrasion_rate = grid.at_node["bedrock__abrasion_rate"]
        self._pluck_rate = grid.at_node["bedrock__plucking_rate"]
        self._setup_length_of_flow_link()

        # 2D arrays in dimensions of n_nodes x n_classes
        self._thickness_by_class = np.zeros(
            (grid.number_of_nodes, self._n_classes)
        )
        self._sed_influxes = np.zeros(
            (grid.number_of_nodes, self._n_classes)
        )
        self._sed_outfluxes = np.zeros(
            (grid.number_of_nodes, self._n_classes)
        )
        self._sed_abr_rates = np.zeros(
            (grid.number_of_nodes, self._n_classes)
        )
        self._br_abrasion_coef = np.zeros(
            (grid.number_of_nodes, self._n_classes)
        )
        self._dHdt_by_class = np.zeros(
            (grid.number_of_nodes, self._n_classes)
        )
        self._get_sediment_thickness_by_class()

    def _create_2D_array_for_input_var(self,
                                   input_var,
                                   var_name):
        """""
        This procedure receive an input variable that will be used
        by the component across nodes and classes (classes represent lithology/grain size).
        The procedure will verify that the input variable match the number of classes
        and will return a 2D array (with dimensions of n_nodes x n_classes).
        If the input variable is already a 2D array, its match to n_nodes and n_classes
        will be examined
        """

        if np.ndim(input_var) == 2:
            input_var_array = input_var
        elif (
                isinstance(input_var, int) or
                isinstance(input_var, float)
        ):
            input_var_array = np.zeros_like(self._grid.nodes.flatten())[:, np.newaxis]
            input_var_array[:, 0] = input_var
        elif np.ndim(input_var) <= 1:
            if isinstance(input_var, list):
                input_var = np.array(input_var)
            if isinstance(input_var, tuple):
                input_var = np.array(list(input_var))

            input_var_array = (
                    np.ones((np.size(self._grid.nodes.flatten()), np.size(input_var))) *
                    input_var[np.newaxis, :])
        else:
            raise ValueError(f"{var_name} array format is invalid")

        # Make sure the array match the grid number of nodes and
        # the number of classes (lithology/grain sizes).
        if np.shape(input_var_array)[1] > 1:
            if (np.shape(input_var_array)[1] != self._n_classes):
                raise ValueError(f"{var_name} array dont match the number of lithology classes")
        if np.shape(input_var_array)[0] != np.size(self._grid.nodes.flatten()):
            raise ValueError(f"The size of {var_name} array dont match the number of grid nodes")

        return input_var_array


    def _get_classes_identity(self):

        """""
        The following procedure will identify if the classes represent lithology or grain sizes.
        A flag (self._classes_identity) that indicates whether the component 
        deals with lithologies or grain sizes defined as:
        0 = single lithology and grain size
        1 = multiple lithologies
        2 = multiple grain sizes
        """""

        self._classes_identity = 0
        if np.shape(self._abr_coefs)[1] > 1:
            # If multiple abrasion coefficients are given -> the classes represent lithology
            self._classes_identity = 1
        if np.shape(self._fractions_from_plucking)[1] > 1:
            # If multiple fractions_from_plucking are given -> the classes represent grain sizes
            self._classes_identity = 2

    def _setup_length_of_flow_link(self):
        """Set up a float or array containing length of the flow link from
        each node, which is needed for the abrasion rate calculations.

        Note: if raster, assumes grid.dx == grid.dy
        """
        if isinstance(self.grid, HexModelGrid):
            self._flow_link_length_over_cell_area = (
                    self.grid.spacing / self.grid.area_of_cell[0]
            )
            self._flow_length_is_variable = False
            self._grid_has_diagonals = False
        elif isinstance(self.grid, DiagonalsMixIn):
            self._flow_length_is_variable = False
            self._grid_has_diagonals = True
            self._flow_link_length_over_cell_area = (
                    _D8_CHAN_LENGTH_FACTOR * self.grid.dx / self.grid.area_of_cell[0]
            )
        else:
            self._flow_length_is_variable = True
            self._update_flow_link_length_over_cell_area()
            self._grid_has_diagonals = False

    def _get_sediment_thickness_by_class(self):
        self._thickness_by_class = np.divide(
            self._grid.at_node['grains__weight'],
         (self._rho_sed * self._sediment_porosity * self._grid.dx * self._grid.dx)
        )





    ## INSERT HERE ALL Class procedures ##

# def _calc_tau_star_c(self):
#     alpha = self._alpha
#     beta = self._beta
#     fractions_sizes = self._grid.at_node['grains_classes__size']
#     median_size_at_node = self._grid.at_node['median_size__weight'][:, np.newaxis]
#     tau_star_c = self._tau_star_c
#     tau_star_c[:] = np.inf
#     tau_star_c[self.grid.core_nodes, :] = beta * np.divide(fractions_sizes[self.grid.core_nodes, :],
#                                                            median_size_at_node[self.grid.core_nodes]) ** alpha
