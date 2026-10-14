#%%
import os
import pandas as pd
import datajoint as dj
# from datetime import datenid
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.stats import kruskal, mannwhitneyu
import copy
from scipy import ndimage
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import notebooks.CoordinatedCellTypes.fig2.fig2_preprocessing as fig2_preprocessing
from notebooks.CoordinatedCellTypes.final.fig_params import *



vlag_cmap = sb.color_palette("vlag", n_colors=10, desat=1)
tab_cmap = sb.color_palette(palette='tab10', n_colors=10, desat=0.5)

def run_pca(frs_mats, n_stim, n_cells=None, concat=True, pca=None):
    if concat:
        frs_mats_flat_concat = np.concatenate(frs_mats, axis=0)
    else:
        frs_mats_flat_concat = frs_mats
    if n_cells is not None:
        # randomly sample cells without replacement
        frs_mats_flat_concat = frs_mats_flat_concat[np.random.choice(frs_mats_flat_concat.shape[0], n_cells, replace=False), :]

    # Apply PCA
    if pca is None:
        pca = PCA(n_components=None)
        frs_mat_flat_pca = pca.fit_transform(frs_mats_flat_concat.T).T  # shape: n_components, n_samples
    else:
        frs_mat_flat_pca = pca.transform(frs_mats_flat_concat.T).T  # shape: n_components, n_samples
    frs_mat_pca = fig2_preprocessing.unflatten_frs_mat(frs_mat_flat_pca, n_stim=n_stim, n_trials=1, n_timepoints=800)
    return pca, frs_mat_pca

def visualize_pca_components(frs_mat_pca, pca, stim_idxs, stim_labels, area, latent_dims=3, stim_types_colors=None, fig_path=None):
    explained_variance = pca.explained_variance_ratio_

    if stim_types_colors is None:
        stim_types_colors = tab_cmap

    fig, axs = plt.subplots(1, latent_dims, figsize=(6*latent_dims*cm, 5*cm), constrained_layout=True, sharey=False)

    for comp in range(latent_dims):
        ax = axs[comp]
        comp_means_all = []
        comp_means_stim = []
        for i, stim_idx in enumerate(stim_idxs):
            comp_means = np.mean(frs_mat_pca[comp,stim_idx,:,200-50:200+int(200*3)],axis=0)
            comp_means_stim.append(comp_means)
            # ax.plot(comp_means, alpha=0.5, c=cmap[stim_idx])
        comp_means_all.append(np.array(comp_means_stim))
        comp_means_all = np.array(comp_means_all)
        comp_means_avg = np.mean(comp_means_all, axis=0)
        comps_means_sem = np.std(comp_means_all, axis=0) / np.sqrt(comp_means_all.shape[0])
        for i, stim_idx in enumerate(stim_idxs):
            ax.plot(comp_means_avg[i], linewidth=1, c=stim_types_colors[i], label=stim_labels[i])
            ax.fill_between(np.arange(comp_means_avg.shape[1]),
                            comp_means_avg[i] - comps_means_sem[i],
                            comp_means_avg[i] + comps_means_sem[i],
                            color=stim_types_colors[i], alpha=0.2)
            ax.set(xticks=[50,100,150], xticklabels=[0,0.25,0.5], xlabel='time (s)')
        # get min and max y limits
        y_min = np.min(comp_means_avg - comps_means_sem)
        y_max = np.max(comp_means_avg + comps_means_sem)
        y_lim = np.max(np.abs([y_min, y_max]))
        ax.set(ylim=(-y_lim-0.05*y_lim, y_lim+0.05*y_lim)) # , xlim=[0, 150]
        ax.set_title(f'Component {comp+1} - {explained_variance[comp]*100:.1f}%')
    axs[0].set(ylabel='PCA component activity (a.u.)')
    axs[-1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, frameon=False)
    plt.suptitle(f'{area}')
    # plt.suptitle(f'{area} {experiments_to_include[0]}')
    for ax in axs:
        ax.set(xlim=[0, 150])
    if fig_path is not None:
        plt.savefig(os.path.join(fig_path, f'{area}_PCA_components.pdf'), dpi=300)
    plt.show()

def visualize_pca_3d(frs_mat_pca, stim_idxs, stim_labels, area, stim_types_colors=None, animation=False, fig_path=None, t_start=150, t_end=750, rotate=False, repeats=1, plot_params=None):
    from matplotlib.animation import FuncAnimation
    # if animation:
    #     try:
    #         plt.switch_backend('TkAgg')
    #     except Exception as e:
    #         print(f"Warning: Could not switch to TkAgg backend. Animation might not display correctly. Error: {e}")
    plt.close('all')
    if stim_types_colors is None:
        stim_types_colors = sb.color_palette(palette='rocket', n_colors=len(stim_idxs))

    fig = plt.figure(figsize=(5*cm,5*cm))
    ax = fig.add_subplot(111, projection='3d')
    
    # Time window indices (approx -0.25s to 2.75s relative to stim onset at 200)
    
    data_to_plot = []
    for i, stim_idx in enumerate(stim_idxs):
        # Shape: (3, time)
        traj = frs_mat_pca[:3, stim_idx, 0, t_start:t_end] 
        data_to_plot.append(traj)

    # Calculate limits
    all_data = np.concatenate(data_to_plot, axis=1)

    # x_lims = (np.min(all_data[0]), np.max(all_data[0]))
    # y_lims = (np.min(all_data[1]), np.max(all_data[1]))
    # z_lims = (np.min(all_data[2]), np.max(all_data[2]))
    
    # Make axes equal scale
    x_range = np.max(all_data[0]) - np.min(all_data[0])
    y_range = np.max(all_data[1]) - np.min(all_data[1])
    z_range = np.max(all_data[2]) - np.min(all_data[2])
    max_range = max(x_range, y_range, z_range)
    
    mid_x = (np.max(all_data[0]) + np.min(all_data[0])) * 0.5
    mid_y = (np.max(all_data[1]) + np.min(all_data[1])) * 0.5
    mid_z = (np.max(all_data[2]) + np.min(all_data[2])) * 0.5

    x_lims = (mid_x - max_range/2, mid_x + max_range/2)
    y_lims = (mid_y - max_range/2, mid_y + max_range/2)
    z_lims = (mid_z - max_range/2, mid_z + max_range/2)
    elev = 30
    azim = 15

    if plot_params:
        x_lims = plot_params.get('xlim', x_lims)
        y_lims = plot_params.get('ylim', y_lims)
        z_lims = plot_params.get('zlim', z_lims)
        elev = plot_params.get('elev', 30)
        azim = plot_params.get('azim', 15)
    z_min = z_lims[0]

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title(f'PCA Trajectories 3D - {area}')
    ax.grid(False)

    if not animation:
        ax.set_xlim(x_lims)
        ax.set_ylim(y_lims)
        ax.set_zlim(z_lims)
        ax.view_init(elev=elev, azim=azim)
        for i, traj in enumerate(data_to_plot):
            ax.plot(traj[0], traj[1], traj[2], c=stim_types_colors[i], label=stim_labels[i], linewidth=1)
            # Add shadow
            ax.plot(traj[0], traj[1], zs=z_min, zdir='z', c=stim_types_colors[i], alpha=0.15, linewidth=0.6)
            # Mark start (circle) and end (triangle)
            expansion_end = 250-t_start # 200 to stim start -- 50 to 0.5s, 100 to 0.5s
            stim_start = 200-t_start
            stim_end = 300-t_start
            ax.scatter(traj[0,stim_start], traj[1, stim_start], traj[2, stim_start], c=[stim_types_colors[i]], marker='o', s=10)
            ax.scatter(traj[0, stim_end], traj[1, stim_end], traj[2, stim_end], c=[stim_types_colors[i]], marker='^', s=10)
            ax.scatter(traj[0,expansion_end], traj[1,expansion_end], traj[2,expansion_end], c=[stim_types_colors[i]], marker='s', s=10)
        # Add legend without frame to the right side of the plot
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), frameon=False)
        if fig_path is not None:
            plt.savefig(os.path.join(fig_path, f'PCA_trajectories_{area}.pdf'), dpi=300)
        plt.show()
    else:
       
        # Define the frames for a single pass
        base_frames = range(1, data_to_plot[0].shape[1], 2)
        
        # Repeat frames if requested
        animation_frames = list(base_frames) * repeats
        n_total_animation_frames = len(animation_frames)

        def update(frame_info):
            frame_idx, traj_idx = frame_info
            ax.clear()
            ax.set_xlim(x_lims)
            ax.set_ylim(y_lims)
            ax.set_zlim(z_lims)
            ax.view_init(elev=30, azim=15)
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_zlabel('PC3')
            ax.set(xticks=[], yticks=[], zticks=[])
            ax.set_title(f'PCA Trajectories 3D - {area}')
            ax.grid(False)
            

            if rotate:
                # Rotate the view
                angle = (frame_idx / n_total_animation_frames) * 80 * repeats
                # ax.view_init(elev=20, azim=angle)
                ax.view_init(elev=elev, azim=angle)
            for i, traj in enumerate(data_to_plot):
                current_traj = traj[:, :traj_idx]
                if current_traj.shape[1] > 0:
                    ax.plot(current_traj[0], current_traj[1], current_traj[2], c=stim_types_colors[i], label=stim_labels[i])
                    # Add shadow
                    ax.plot(current_traj[0], current_traj[1], zs=z_min, zdir='z', c=stim_types_colors[i], alpha=0.3, linewidth=1)
                    ax.scatter(current_traj[0, -1], current_traj[1, -1], current_traj[2, -1], c=[stim_types_colors[i]], marker='o', s=20)
            # remove legend
            ax.legend().remove()
            # ax.legend(loc='upper left')
            # ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), frameon=False)

            return ax,

        ani = FuncAnimation(fig, update, frames=enumerate(animation_frames), interval=50, blit=False)
        if fig_path:
            save_path = os.path.join(fig_path, f'PCA_trajectories_{area}.gif')
            writer = 'pillow' if save_path.endswith('.gif') else 'ffmpeg'
            ani.save(save_path, writer=writer, fps=30, dpi=300)
        plt.show()
        return ani
    
def visualize_pca_components_multi(frs_mat_pcas, pca, stim_idxs, stim_labels, area, group_styles, latent_dims=3, stim_types_colors=None, fig_path=None):
    explained_variance = pca.explained_variance_ratio_

    if stim_types_colors is None:
        stim_types_colors = tab_cmap

    fig, axs = plt.subplots(1, latent_dims, figsize=(6*latent_dims*cm, 5*cm), constrained_layout=True, sharey=False)

    for frs_mat_pca, group_style in zip(frs_mat_pcas, group_styles):
        for comp in range(latent_dims):
            ax = axs[comp]
            comp_means_all = []
            comp_means_stim = []
            for i, stim_idx in enumerate(stim_idxs):
                comp_means = np.mean(frs_mat_pca[comp,stim_idx,:,200-50:200+int(200*3)],axis=0)
                comp_means_stim.append(comp_means)
                # ax.plot(comp_means, alpha=0.5, c=cmap[stim_idx])
            comp_means_all.append(np.array(comp_means_stim))
            comp_means_all = np.array(comp_means_all)
            comp_means_avg = np.mean(comp_means_all, axis=0)
            comps_means_sem = np.std(comp_means_all, axis=0) / np.sqrt(comp_means_all.shape[0])
            for i, stim_idx in enumerate(stim_idxs):
                ax.plot(comp_means_avg[i], linewidth=1, c=stim_types_colors[i], label=stim_labels[i], linestyle=group_style)
                # ax.fill_between(np.arange(comp_means_avg.shape[1]),
                #                 comp_means_avg[i] - comps_means_sem[i],
                #                 comp_means_avg[i] + comps_means_sem[i],
                #                 color=stim_types_colors[i], alpha=0.2)
                ax.set(xticks=[50,100,150], xticklabels=[0,0.25,0.5], xlabel='time (s)')
            # get min and max y limits
            y_min = np.min(comp_means_avg - comps_means_sem)
            y_max = np.max(comp_means_avg + comps_means_sem)
            y_lim = np.max(np.abs([y_min, y_max]))
            # ax.set(ylim=(-y_lim-0.05*y_lim, y_lim+0.05*y_lim)) # , xlim=[0, 150]
            ax.set_title(f'Component {comp+1} - {explained_variance[comp]*100:.1f}%')
    axs[0].set(ylabel='PCA component activity (a.u.)')
    axs[-1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, frameon=False)
    plt.suptitle(f'{area}')
    # plt.suptitle(f'{area} {experiments_to_include[0]}')
    for ax in axs:
        ax.set(xlim=[0, 150])
    if fig_path is not None:
        plt.savefig(os.path.join(fig_path, f'{area}_PCA_components.pdf'), dpi=300)
    plt.show()


def visualize_pca_3d_multi(frs_mat_pcas, stim_idxs, stim_labels, area, group_styles, stim_types_colors=None, animation=False, fig_path=None, t_start=150, t_end=750, rotate=False, repeats=1, plot_params=None):
    from matplotlib.animation import FuncAnimation
    # if animation:
    #     try:
    #         plt.switch_backend('TkAgg')
    #     except Exception as e:
    #         print(f"Warning: Could not switch to TkAgg backend. Animation might not display correctly. Error: {e}")
    plt.close('all')
    if stim_types_colors is None:
        stim_types_colors = sb.color_palette(palette='rocket', n_colors=len(stim_idxs))

    fig = plt.figure(figsize=(5*cm,5*cm))
    ax = fig.add_subplot(111, projection='3d')
    
    # Time window indices (approx -0.25s to 2.75s relative to stim onset at 200)
    for frs_mat_pca, group_style in zip(frs_mat_pcas, group_styles):
        data_to_plot = []
        for i, stim_idx in enumerate(stim_idxs):
            # Shape: (3, time)
            traj = frs_mat_pca[:3, stim_idx, 0, t_start:t_end] 
            data_to_plot.append(traj)

        # Calculate limits
        all_data = np.concatenate(data_to_plot, axis=1)

        # x_lims = (np.min(all_data[0]), np.max(all_data[0]))
        # y_lims = (np.min(all_data[1]), np.max(all_data[1]))
        # z_lims = (np.min(all_data[2]), np.max(all_data[2]))
        
        # Make axes equal scale
        x_range = np.max(all_data[0]) - np.min(all_data[0])
        y_range = np.max(all_data[1]) - np.min(all_data[1])
        z_range = np.max(all_data[2]) - np.min(all_data[2])
        max_range = max(x_range, y_range, z_range)
        
        mid_x = (np.max(all_data[0]) + np.min(all_data[0])) * 0.5
        mid_y = (np.max(all_data[1]) + np.min(all_data[1])) * 0.5
        mid_z = (np.max(all_data[2]) + np.min(all_data[2])) * 0.5
        
        x_lims = (mid_x - max_range/2, mid_x + max_range/2)
        y_lims = (mid_y - max_range/2, mid_y + max_range/2)
        z_lims = (mid_z - max_range/2, mid_z + max_range/2)
        elev = 30
        azim = 15
        
        if plot_params:
            x_lims = plot_params.get('xlim', x_lims)
            y_lims = plot_params.get('ylim', y_lims)
            z_lims = plot_params.get('zlim', z_lims)
            elev = plot_params.get('elev', 30)
            azim = plot_params.get('azim', 15)
        z_min = z_lims[0]

        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title(f'PCA Trajectories 3D - {area}')
        ax.grid(False)

        if not animation:
            ax.set_xlim(x_lims)
            ax.set_ylim(y_lims)
            ax.set_zlim(z_lims)
            ax.view_init(elev=elev, azim=azim)
            for i, traj in enumerate(data_to_plot):
                ax.plot(traj[0], traj[1], traj[2], c=stim_types_colors[i], label=stim_labels[i], linewidth=1, linestyle=group_style)
                # Add shadow
                ax.plot(traj[0], traj[1], zs=z_min, zdir='z', c=stim_types_colors[i], alpha=0.15, linewidth=0.6, linestyle=group_style)
                # Mark start (circle) and end (triangle)
                expansion_end = 250-t_start # 200 to stim start -- 50 to 0.5s, 100 to 0.5s
                stim_start = 200-t_start
                stim_end = 300-t_start
                ax.scatter(traj[0,stim_start], traj[1, stim_start], traj[2, stim_start], c=[stim_types_colors[i]], marker='o', s=10)
                ax.scatter(traj[0, stim_end], traj[1, stim_end], traj[2, stim_end], c=[stim_types_colors[i]], marker='^', s=10)
                ax.scatter(traj[0,expansion_end], traj[1,expansion_end], traj[2,expansion_end], c=[stim_types_colors[i]], marker='s', s=10)
            # Add legend without frame to the right side of the plot
            ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), frameon=False)
    if fig_path is not None:
        plt.savefig(os.path.join(fig_path, f'PCA_trajectories_{area}.pdf'), dpi=300)
    plt.show()