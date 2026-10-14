#%%
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sb
import pandas as pd
import pickle

tab_cmap = sb.color_palette(palette='tab10', n_colors=10, desat=0.5)
vlag_cmap = sb.color_palette("vlag", n_colors=10, desat=1)
stim_types_colors = [vlag_cmap[-1], vlag_cmap[-4], vlag_cmap[0], tab_cmap[5]]
from fig_params import *

#%%
np.random.seed(42)
with open('model_decision_variables_30.pkl', 'rb') as f:
    model_vars = pickle.load(f)

manip_vals = model_vars['mod_strengths']
manip_null_idx = np.where(manip_vals==1)[0][0]
dSC_EXPAs, dSC_EXPA25s, dSC_EXPWs, dSC_DIMMs, dSC_Bases = model_vars['dSC_EXPA'], model_vars['dSC_EXPA25'], model_vars['dSC_EXPW'], model_vars['dSC_DIMM'], model_vars['dSC_Base']
LP_EXPAs, LP_EXPA25s, LP_EXPWs, LP_DIMMs, LP_Bases = model_vars['LP_EXPA'], model_vars['LP_EXPA25'], model_vars['LP_EXPW'], model_vars['LP_DIMM'], model_vars['LP_Base']

dSC_EXPA, dSC_EXPA25, dSC_EXPW, dSC_DIMM, dSC_Base = dSC_EXPAs[manip_null_idx], dSC_EXPA25s[manip_null_idx], dSC_EXPWs[manip_null_idx], dSC_DIMMs[manip_null_idx], dSC_Bases[manip_null_idx]
LP_EXPA, LP_EXPA25, LP_EXPW, LP_DIMM, LP_Base = LP_EXPAs[manip_null_idx], LP_EXPA25s[manip_null_idx], LP_EXPWs[manip_null_idx], LP_DIMMs[manip_null_idx], LP_Bases[manip_null_idx]

#%%
print(np.mean(dSC_EXPA), np.mean(LP_EXPA))
print(np.mean(np.concatenate([dSC_EXPA25,  dSC_EXPW, dSC_DIMM])))
print(np.mean(np.concatenate([LP_EXPA25,  LP_EXPW, LP_DIMM])))

print(np.std(dSC_EXPA), np.std(LP_EXPA))
print(np.std(np.concatenate([dSC_EXPA25,  dSC_EXPW, dSC_DIMM])))
print(np.std(np.concatenate([LP_EXPA25,  LP_EXPW, LP_DIMM])))
#%%
# =====================================================================
# 4. GENERATE 1x3 COMPARISON FIGURE
# =====================================================================
wt_x, wt_y = np.mean(dSC_EXPA), np.mean(LP_EXPA)
base_x, base_y = np.mean(dSC_Base), np.mean(LP_Base)

grid_min, grid_max = -5, 5
x_range = np.linspace(grid_min, grid_max, 100)
y_range = np.linspace(grid_min, grid_max, 100)
X, Y = np.meshgrid(x_range, y_range)

fig, ax = plt.subplots(1, 1, figsize=(6*cm,6*cm))
    
# Layer 1: Empirical Clouds
lim = 1000
ax.scatter(dSC_EXPA25[:lim], LP_EXPA25[:lim], color=stim_types_colors[1], alpha=0.1, edgecolors='none', zorder=1)
ax.scatter(dSC_DIMM[:lim], LP_DIMM[:lim], color=stim_types_colors[3], alpha=0.1, edgecolors='none', zorder=2)
ax.scatter(dSC_EXPW[:lim], LP_EXPW[:lim], color=stim_types_colors[2], alpha=0.1, edgecolors='none', zorder=3)
ax.scatter(dSC_EXPA[:lim], LP_EXPA[:lim], color=stim_types_colors[0], alpha=0.1, edgecolors='none', zorder=4)
ax.set_xlabel("dSC decision value ($x_{dSC}$)")
ax.set_ylabel("LP decision value ($y_{LP}$)")

ax.set_xlim([grid_min, grid_max])
ax.set_ylim([grid_min, grid_max])

plt.tight_layout()
plt.savefig('stim_densities.pdf', dpi=300)
plt.show()

#%%
lim = 2000
df = pd.DataFrame(
    {
        "dSC": np.concatenate([dSC_EXPA[:lim], dSC_EXPA25[:lim], dSC_EXPW[:lim], dSC_DIMM[:lim]]),
        "LP": np.concatenate([LP_EXPA[:lim],LP_EXPA25[:lim],LP_EXPW[:lim],LP_DIMM[:lim]]),
        "condition": (["EXPA"] * len(dSC_EXPA[:lim]) + ["EXPA25"] * len(dSC_EXPA25[:lim]) + ["EXPW"] * len(dSC_EXPW[:lim]) + ["DIMM"] * len(dSC_DIMM[:lim])),
    }
)

palette = {
    "EXPA": stim_types_colors[0],
    "EXPA25": stim_types_colors[1],
    "EXPW": stim_types_colors[2],
    "DIMM": stim_types_colors[3],
}

g = sb.jointplot(
    data=df,
    x="dSC",
    y="LP",
    hue="condition",
    palette=palette,
    kind="scatter",
    joint_kws={"alpha": 0.1, "edgecolor": "none", "s": 3},
    marginal_kws={"common_norm": False, "fill": False},
    height=6 * cm,  # seaborn jointplot uses square figure size via height
)

# 4. Axis labels and limits
g.ax_joint.set_xlabel(r"dSC decision value")
g.ax_joint.set_ylabel(r"LP decision value")
g.ax_joint.set_xlim([grid_min, grid_max])
g.ax_joint.set_ylim([grid_min, grid_max])
sb.move_legend(
    g.ax_joint,
    loc="center left",
    bbox_to_anchor=(
        1.25,
        0.5,
    ),  # shifts outside past the right marginal axis
    frameon=False,  # removes the legend border
)
# Set full alpha (opacity) for the legend markers
for handle in g.ax_joint.get_legend().legend_handles:
    handle.set_alpha(1.0)
    handle.set_markersize(5)

fig_path = r''
plt.savefig(os.path.join(fig_path,"stim_densities.pdf"), dpi=300, bbox_inches="tight")
plt.show()
