

from landlab import RasterModelGrid
from gravel_bedrock_eroder_extended import GravelBedrockEroder
from landlab.components import FlowAccumulator
from soil_grading import SoilGrading
from matplotlib import pyplot as plt
import numpy as np
from landlab import imshow_grid
# import noise
# from landlab.components import LinearDiffuser

xy_spacing = 500
porosity = 0.4
grain_sizes = [0.001, 0.01, 0.05]
grains_weight = [10e8, 10e8, 10e8]


grid = RasterModelGrid((150, 150), xy_spacing=xy_spacing)
elev = grid.add_zeros("topographic__elevation", at="node")
grid.set_closed_boundaries_at_grid_edges(False, False, False, False)

sg = SoilGrading(grid,
                 meansizes=grain_sizes,
                 grains_weight=grains_weight,
                 phi=porosity)

fa = FlowAccumulator(grid, runoff_rate=0.5)
fa.run_one_step()

eroder = GravelBedrockEroder(grid)

rock_elev = grid.at_node["bedrock__elevation"]
rock_elev[:] +=100
elev[:] = rock_elev[:] + grid.at_node['soil__depth']

# noise_mat = np.zeros_like(grid.at_node['soil__depth'][grid.nodes])
# scale = 200
# octaves = 20
# persistence = 0.5
# lacunarity =55
#
# for i in range(np.shape(noise_mat)[0]):
#     for j in range(np.shape(noise_mat)[1]):
#         noise_mat[i][j] = noise.pnoise2(i / scale,
#                                         j / scale,
#                                         octaves=octaves,
#                                         persistence=persistence,
#                                         lacunarity=lacunarity,
#                                         repeatx=np.shape(noise_mat)[1],
#                                         repeaty=np.shape(noise_mat)[0],
#                                         base=2
#                                         )
# rock_elev[:] += np.abs(noise_mat[:].flatten())
#
# elev[:] = rock_elev[:] + grid.at_node['soil__depth']


n_steps = 10e6
fig_cnt = 0
xvec = np.arange(0, np.size(grid.at_node['topographic__elevation'][grid.core_nodes]))*grid.dx
for i,_ in enumerate(range(n_steps )):
    rock_elev[grid.core_nodes] += 0.001
    elev[grid.core_nodes] =rock_elev[grid.core_nodes] + grid.at_node['soil__depth'][grid.core_nodes]
    fa.run_one_step()
    eroder.run_one_step(1)
    if i>= fig_cnt:
        fig, ax = plt.subplots()
        ax.plot(xvec, grid.at_node['topographic__elevation'][grid.core_nodes], color='black')
        ax.plot(xvec, grid.at_node['bedrock__elevation'][grid.core_nodes], color='blue')
        ax2 = ax.twinx()
        ax2.plot(xvec, grid.at_node['median_size__weight'][grid.core_nodes], color='salmon')
        ax.set_xlabel('Distnace upstream')
        ax.set_ylabel('Elevation [m]')
        ax2.set_ylabel('Grainsize [m]')
        plt.title(f'time step = {i}')
        ax.set_ylim([0,np.max(grid.at_node['topographic__elevation'])+100])
        plt.tight_layout()
        plt.show()

    sg.update_median_grain_size()

