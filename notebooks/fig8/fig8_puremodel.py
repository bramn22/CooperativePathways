#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import warnings
import os
import seaborn as sb

# Color palettes
tab_cmap = sb.color_palette(palette='tab10', n_colors=10, desat=0.5)
vlag_cmap = sb.color_palette("vlag", n_colors=10, desat=1)
stim_types_colors = [vlag_cmap[-1], vlag_cmap[-4], vlag_cmap[0], tab_cmap[5]]

from fig_params import *

fig_path = r''
os.makedirs(fig_path, exist_ok=True) # Ensure directory exists

# --- SIMULATION CONFIGURATION ---
CLOUD_SIGMA = 0.8            # Variance of the behavioral state distributions
BOUNDARY_THRESHOLD = 0       # Downstream motor activation threshold
GATE_OFFSET = -0.1          # Mathematical offset required for the non-linear gate

# --- 1. DEFINING THE STATE SPACE ---
# Define centroids for behavioral states (Decoder Decision Values)
baseline_mu = np.array([-2.0, -2.0])
non_threat_mu = np.array([-2.0, -2.0])
threat_mu = np.array([2.0, 2.0])

# Manipulation target vectors
inhibit_dsc_target = np.array([-0.5, 2.0])
inhibit_lp_target = np.array([2.0, -0.5])
opto_dsc_target = np.array([0.5, -2.0])
opto_lp_target = np.array([-2.0, 0.5])

# Expanded Grid for perfect integration (plotted limits are cropped later)
x_val = np.linspace(-6, 6, 300)
y_val = np.linspace(-6, 6, 300)
X, Y = np.meshgrid(x_val, y_val)

def gaussian_2d(X, Y, mu, sigma):
    """Generates a 2D Gaussian density blob for smooth state visualization."""
    return np.exp(-((X - mu[0])**2 + (Y - mu[1])**2) / (2 * sigma**2))

Z_threat_blob = gaussian_2d(X, Y, threat_mu, CLOUD_SIGMA)
Z_non_threat_blob = gaussian_2d(X, Y, non_threat_mu, CLOUD_SIGMA)

# Shared Colormaps and Extent for consistent state space visualization
threat_color = stim_types_colors[0]
cmap_red = mcolors.LinearSegmentedColormap.from_list('fade_red', [threat_color+(0.0,), threat_color+(1,)])
cmap_gray = mcolors.LinearSegmentedColormap.from_list('fade_gray', [(0.3, 0.3, 0.3, 0.0), (0.3, 0.3, 0.3, 1)])
extent = [x_val.min(), x_val.max(), y_val.min(), y_val.max()]

def draw_vectors(ax):
    """Overlays the experimental manipulation vectors onto the given axes."""
    arrow_props = dict(arrowstyle="->", lw=1, color='black')
    ax.annotate('', xy=inhibit_dsc_target, xytext=threat_mu, arrowprops=arrow_props)
    ax.annotate('', xy=inhibit_lp_target, xytext=threat_mu, arrowprops=arrow_props)
    ax.annotate('', xy=opto_dsc_target, xytext=baseline_mu, arrowprops=arrow_props)
    ax.annotate('', xy=opto_lp_target, xytext=baseline_mu, arrowprops=arrow_props)

# --- FIGURE 1: STATE SPACE ---
fig_state, ax_state = plt.subplots(figsize=(5*cm, 5*cm))

ax_state.imshow(Z_non_threat_blob, extent=extent, origin='lower', cmap=cmap_gray, aspect='auto')
ax_state.imshow(Z_threat_blob, extent=extent, origin='lower', cmap=cmap_red, aspect='auto')

ax_state.axhline(0, color='black', linestyle='--', linewidth=1, zorder=1)
ax_state.axvline(0, color='black', linestyle='--', linewidth=1, zorder=1)
draw_vectors(ax_state)

ax_state.set(xlim=(-4, 4), ylim=(-4, 4), xlabel="dSC decision value ($x_{dSC}$)", ylabel="LP decision value ($y_{LP}$)")
handles, labels = ax_state.get_legend_handles_labels()
ax_state.legend(handles=[mpatches.Patch(color=(0.3, 0.3, 0.3, 0.6), label='Non-Threat'), 
                         mpatches.Patch(color=(0.7, 0.1, 0.1, 0.6), label='Threat')] + handles, loc='upper left')
fig_state.tight_layout()
plt.savefig(os.path.join(fig_path, 'state_space.pdf'), dpi=300)
plt.show()

#%%
# --- FIGURE 2: THE MODELS (1x3 Panel) ---
fig_models, axes_models = plt.subplots(1, 3, figsize=(12*cm, 4*cm), sharey=True)

# Note: GATE_OFFSET is exclusively applied to the Driver-Gate model
Zs = [
    1.00 * (X - Y),                                  # Competition
    1.00 * (X + Y),                                  # Linear Sum
    1.00 * X * (1 / (1 + np.exp(-1.0 * Y)))  # Driver-Gate (Optimized Sweet Spot)
]

titles = ["Competition", "Linear Sum", "Driver-Gate"]
levels = np.linspace(-7, 7, 60)
 
for i, ax in enumerate(axes_models):
    ax.set_title(titles[i], fontweight='bold')
    c = ax.contourf(X, Y, Zs[i], levels=levels, cmap='vlag', vmin=-5, vmax=5, extend='both', rasterized=True)
    
    # draw_vectors(ax)
    ax.set(xlim=(-4, 4), ylim=(-4, 4), xlabel="dSC decision value ($x_{dSC}$)")
    if i == 0: ax.set_ylabel("LP decision value ($y_{LP}$)", fontsize=8)

cbar_ax = fig_models.add_axes([0.92, 0.15, 0.015, 0.7])
cbar = fig_models.colorbar(c, cax=cbar_ax)
if cbar.solids is not None:
    cbar.solids.set_rasterized(True)
cbar.set_label('Downstream decision value', rotation=270, labelpad=20, fontsize=8, fontweight='bold')
plt.subplots_adjust(left=0.06, right=0.9, wspace=0.15)
plt.savefig(os.path.join(fig_path, 'models_ideal.pdf'), dpi=300)
plt.show()

#%%
# --- FIGURE GENERATOR: PARAMETER SWEEPS ---
def get_simulated_p_escape(model_func, mu_x_arr, mu_y_arr, p1, p2, b):
    """Calculates P(Escape) via Volume-Over-Threshold."""
    p_escape = np.zeros_like(mu_x_arr)
    
    Z_model_raw = model_func(X, Y, p1, p2, b)
    Z_escape_mask = Z_model_raw > BOUNDARY_THRESHOLD  # Boolean mask of the escape zone
    
    for i in range(len(mu_x_arr)):
        Z_cloud = gaussian_2d(X, Y, [mu_x_arr[i], mu_y_arr[i]], CLOUD_SIGMA)
        Z_cloud /= (np.sum(Z_cloud) + 1e-9)
        # Probability = the fraction of the cloud's volume inside the escape zone
        p_escape[i] = np.sum(Z_cloud[Z_escape_mask])

    return p_escape

def create_param_sweep_fig(fig_title, start_mu, target_mu):
    """Generates a 2x6 grid pairing P(Escape) line plots with 2D Decision Boundaries."""
    fig = plt.figure(figsize=(26*cm, 9*cm), layout="constrained")
    fig.suptitle(fig_title, fontweight='bold', fontsize=16)
    gs = fig.add_gridspec(2, 6)

    # Models return unclipped raw drive
    models = [
        {'name': 'Competition', 'func': lambda x, y, p1, p2, b: p1*x - p2*y + b},
        {'name': 'Linear Sum', 'func': lambda x, y, p1, p2, b: p1*x + p2*y + b},
        {'name': 'Driver-Gate', 'func': lambda x, y, p1, p2, b: p1 * x * (1 / (1 + np.exp(-p2 * (y)))) + b}
    ]

    # Geometrically symmetric weights to produce visually symmetric slopes around the 45-degree diagonal
    sym_w2 = np.array([0.25, 0.5, 0.8, 1.25, 2.0, 4.0])

    params = [
            (1.0, 1.0, sym_w2, "$w_2$ (LP weight)"),
            (1.0, 1.0, sym_w2, "$w_2$ (LP weight)"),
            (1.0, 1.0, np.linspace(1, 4.1, 6), "$k$ (LP steepness)")
    ]
    
    strengths = np.linspace(0, 1, 50)
    traj_x = start_mu[0] + strengths * (target_mu[0] - start_mu[0])
    traj_y = start_mu[1] + strengths * (target_mu[1] - start_mu[1])
    cmap = plt.cm.viridis

    for col, model in enumerate(models):
        p1_def, p2_def, p2_sweep, p2_name = params[col]

        # Use larger, model-specific steps to cleanly separate the lines across the state space
        if model['name'] == 'Driver-Gate':
            # Restrict bias strictly to negative values so boundaries stay on the right-hand side (x > 0)
            b_sweep = np.linspace(-2.5, -0.1, 6)
        else:
            b_sweep = np.linspace(-3.0, 3.0, 6)

        for row, (sweep_vals, param_name, sweep_type) in enumerate([
            (p2_sweep, p2_name, 'p2'),
            (b_sweep, "Gate Offset ($b$)" if model['name'] == 'Driver-Gate' else "Model Bias ($b$)", 'b')
        ]):
            ax_line = fig.add_subplot(gs[row, col*2])
            ax_bound = fig.add_subplot(gs[row, col*2 + 1])
            

            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=len(sweep_vals)-1))
            
            for i, val in enumerate(sweep_vals):
                norm_val = i / (len(sweep_vals) - 1)
                
                # Assign parameters dynamically
                p1_curr = p1_def
                p2_curr = val if sweep_type == 'p2' else p2_def
                if model['name'] == 'Driver-Gate':
                    b_curr = val if sweep_type == 'b' else GATE_OFFSET
                else:
                    b_curr = val if sweep_type == 'b' else 0
                                
                # Evaluate and Plot
                p_escape = get_simulated_p_escape(model['func'], traj_x, traj_y, p1_curr, p2_curr, b_curr)
                ax_line.plot(strengths, p_escape, color=cmap(norm_val), lw=1, alpha=0.8)
                
                Z = model['func'](X, Y, p1_curr, p2_curr, b_curr)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ax_bound.contour(X, Y, Z, levels=[BOUNDARY_THRESHOLD], colors=[cmap(norm_val)], linewidths=1, alpha=0.9, zorder=3)
            
            cbar = fig.colorbar(sm, ax=ax_bound, fraction=0.046, pad=0.04)
            cbar.set_label(f"Varying {param_name}", fontsize=8)
            cbar.set_ticks(range(len(sweep_vals)))
            cbar.set_ticklabels([str(np.round(v, 2)) for v in sweep_vals])

            ax_line.set(ylim=(-0.05, 1.05), xlim=(-0.02, 1.02), xlabel="Manipulation Strength")
            ax_line.grid(True, linestyle='--', alpha=0.5)
            if col == 0: ax_line.set_ylabel("Simulated P(Escape)", fontsize=8)
            
            # Use consistent state space background
            ax_bound.imshow(Z_non_threat_blob, extent=extent, origin='lower', cmap=cmap_gray, aspect='auto', zorder=1)
            ax_bound.imshow(Z_threat_blob, extent=extent, origin='lower', cmap=cmap_red, aspect='auto', zorder=1)
            
            # Draw manipulation vector arrow
            # ax_bound.annotate('', xy=target_mu, xytext=start_mu, arrowprops=dict(arrowstyle="->", lw=2.0, color='black'), zorder=4)
            
            ax_bound.axhline(0, color='black', linestyle='--', alpha=0.5, zorder=2)
            ax_bound.axvline(0, color='black', linestyle='--', alpha=0.5, zorder=2)
            ax_bound.set(xlim=(-4, 4), ylim=(-4, 4), xlabel="dSC decision value ($x_{dSC}$)")
            if col == 0: ax_bound.set_ylabel("LP decision value ($y_{LP}$)", fontsize=10)
    
    return fig

# Manipulation target vectors
inhibit_dsc_target = np.array([-2, 2.0])
inhibit_lp_target = np.array([2.0, -2])
opto_dsc_target = np.array([3, -2.0])
opto_lp_target = np.array([-2.0, 3])

fig_sweep_lp_inh = create_param_sweep_fig("LP Inhibition (From Threat State)", threat_mu, inhibit_lp_target)
plt.savefig(os.path.join(fig_path, 'manip_LP_inh.pdf'), dpi=300)

fig_sweep_dsc_inh = create_param_sweep_fig("dSC Inhibition (From Threat State)", threat_mu, inhibit_dsc_target)
plt.savefig(os.path.join(fig_path, 'manip_dSC_inh.pdf'), dpi=300)

fig_sweep_opto_lp = create_param_sweep_fig("Opto LP Activation (From Baseline)", baseline_mu, opto_lp_target)
plt.savefig(os.path.join(fig_path, 'manip_LP_opto.pdf'), dpi=300)

fig_sweep_opto_dsc = create_param_sweep_fig("Opto dSC Activation (From Baseline)", baseline_mu, opto_dsc_target)
plt.savefig(os.path.join(fig_path, 'manip_dSC_opto.pdf'), dpi=300)

plt.show()