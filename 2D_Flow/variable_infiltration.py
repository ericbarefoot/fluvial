from landlab.components.overland_flow import OverlandFlow
from landlab.components import SoilInfiltrationGreenAmpt
from landlab.plot.imshow import imshow_grid
from landlab.plot.colors import water_colormap
from landlab import RasterModelGrid
from landlab.io.esri_ascii import read_esri_ascii
from matplotlib.pyplot import figure
import numpy as np
import matplotlib.pyplot as plt
from time import time
# from tqdm import tqdm
# %matplotlib inline

run_time = 100  # duration of run, (s)
h_init = 0.5  # initial thin layer of water (m)
n = 0.01  # roughness coefficient, (s/m^(1/3))
g = 9.8  # gravity (m/s^2)
alpha = 0.7  # time-step factor (nondimensional; from Bates et al., 2010)
u = 0.4  # constant velocity (m/s, de Almeida et al., 2012)
run_time_slices = (10, 50, 100)
elapsed_time = 1.0
width = 30
length = 40
dx = 10
slope = 1e-1

rmg = RasterModelGrid((width, length), dx) # replace this with Edwin's grid
z = rmg.add_zeros('topographic__elevation', at='node')
rmg.set_closed_boundaries_at_grid_edges(True, False, True, False)
np.all(rmg.at_node['topographic__elevation'] == z)
h = rmg.add_zeros("surface_water__depth", at="node", dtype=float)
h += h_init
d = rmg.add_ones("soil_water_infiltration__depth", at="node", dtype=float)
d *= 0.2
hc = rmg.add_ones('hydraulic_conductivity', at='node') * 1e-5

channel_width = 8
channel_depth = 2

infBandWidthNorm = 0.5

infBandWidth = length * dx / 2 * infBandWidthNorm

channel_left = (length * dx + channel_width) / 2
channel_right = (length * dx - channel_width) / 2

isChannel = np.logical_and(rmg.x_of_node < channel_left, rmg.x_of_node > channel_right)

highInfBand = np.logical_and(rmg.x_of_node < channel_left + infBandWidth, rmg.x_of_node > channel_right - infBandWidth)
inf_mask = np.logical_xor(highInfBand, isChannel)
hc[inf_mask] *= 10

of = OverlandFlow(rmg, steep_slopes=True)

SI = SoilInfiltrationGreenAmpt(rmg, hydraulic_conductivity=hc)

elapsed_time = 0
run_time = 1e4
iters = 0

while elapsed_time < run_time:
    # First, we calculate our time step.
    dt = of.calc_time_step()
    # Now, we can generate overland flow.
    of.overland_flow()
    SI.run_one_step(dt)
    # Increased elapsed time
    if iters % 1000 == 0:
        print('Elapsed time: ', elapsed_time)
    elapsed_time += dt
    iters += 1

imshow_grid(rmg, 'surface_water__depth', cmap='Blues')
