import matplotlib.pyplot as plt
import matplotlib.colors

color_sSC = matplotlib.colors.to_rgba('#de8f05')
color_dSC = matplotlib.colors.to_rgba('#d55e00')
color_LP = matplotlib.colors.to_rgba('#029e73')

# Plot colors scheme
colors = [color_sSC, color_dSC, color_LP]
color_CTRL = matplotlib.colors.to_rgba('#949494')
color_NTSR = matplotlib.colors.to_rgba('#0173b2')
color_GRP = matplotlib.colors.to_rgba('#cc78bc')

import matplotlib as mpl
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.left'] = True
plt.rcParams['axes.spines.bottom'] = True

plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['lines.linewidth'] = 0.75
plt.rcParams['lines.markersize'] = 2.0
plt.rcParams['patch.linewidth'] = 1.0
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['ytick.major.width'] = 0.5
plt.rcParams['xtick.minor.width'] = 0.5
plt.rcParams['ytick.minor.width'] = 0.5
plt.rcParams['xtick.major.size'] = 3.5
plt.rcParams['ytick.major.size'] = 3.5
plt.rcParams['xtick.minor.size'] = 2.0
plt.rcParams['ytick.minor.size'] = 2.0

mpl.rcParams['font.size'] = 8
mpl.rcParams['axes.titlesize'] = 8
mpl.rcParams['axes.labelsize'] = 8
mpl.rcParams['xtick.labelsize'] = 6
mpl.rcParams['ytick.labelsize'] = 6
mpl.rcParams['legend.fontsize'] = 6
mpl.rcParams['legend.title_fontsize'] = 6
mpl.rcParams['legend.fontsize'] = 6

mpl.rcParams['font.family'] = "sans-serif"
mpl.rcParams['font.sans-serif'] = "Arial"
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

cm = 1/2.54 # centimeters in inches