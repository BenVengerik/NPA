import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import gaussian_filter
import math

# =========================================================
# 1. CONFIGURATION
# =========================================================
SESSION = "pup1"        # Change this to compare animals
INCLUDE_MUA = True      

STIMULI = [
    "brush_tail",      
    "brush_contra",
    "brush_ipsi"
]

# --- MANUAL SETTINGS ---
FIXED_VMAX = 120  # e.g., 50. Set to None for auto-scaling per session.

# Time Window
PRE_TIME = 0.1     # Seconds before stim
POST_TIME = 1.5    # Seconds after stim

# Heatmap Bins
BIN_TIME = 0.005   # 5ms
BIN_DEPTH = 20     # 20µm

# =========================================================
# 2. HELPER FUNCTIONS
# =========================================================
def get_val(r, col):
    return float(r[col]) if col in r and not pd.isna(r[col]) else np.nan

def load_session_data(session, stim):
    # Load Meta
    csv_path = Path("data/sessions.csv")
    meta = pd.read_csv(csv_path)
    row = meta.loc[(meta['session'] == session) & (meta['stim'] == stim)]
    if len(row) == 0: return None
    row = row.iloc[0]
    
    # Load Anatomy
    s1_upper = get_val(row, 's1_upper_um')
    s1_lower = get_val(row, 's1_lower_um')
    
    # --- LAYERS ---
    layers = {
        'L1':   get_val(row, 'l1_end'),
        'L2/3': get_val(row, 'l23_end'),
        'L4':   get_val(row, 'l4_end'),
        'L5/6': get_val(row, 'l6_end')  # Merged L5 and L6
    }
    
    # Load Times
    stim_path = Path(f"results/summaries/{session}_{stim}_onsets.txt")
    if not stim_path.exists(): return None
    stim_times = np.loadtxt(stim_path)
    if stim_times.ndim == 0: stim_times = [stim_times]
    
    # Load Spikes
    phy_dir = Path(row['phy_dir'])
    spike_times = np.load(phy_dir / 'spike_times.npy').flatten() / 30000.0
    spike_clusters = np.load(phy_dir / 'spike_clusters.npy').flatten()
    cluster_info = pd.read_csv(phy_dir / 'cluster_info.tsv', sep='\t')
    
    # Templates for depth
    coords = np.load(phy_dir / 'channel_positions.npy')
    templates = np.load(phy_dir / 'templates.npy')
    spike_templates = np.load(phy_dir / 'spike_templates.npy').flatten()
    
    # Calculate Depths
    template_peaks = np.max(np.abs(templates), axis=1)
    template_max_chans = np.argmax(template_peaks, axis=1)
    all_depths = coords[template_max_chans[spike_templates], 1]
    
    # Filter S1 Units
    lbl = 'KSLabel' if 'KSLabel' in cluster_info.columns else 'group'
    valid_labels = ['good', 'mua'] if INCLUDE_MUA else ['good']
    
    s1_units = cluster_info[
        (cluster_info['depth'] >= s1_lower) & 
        (cluster_info['depth'] <= s1_upper) &
        (cluster_info[lbl].isin(valid_labels))
    ]
    
    valid_ids = set(s1_units['cluster_id'])
    mask = np.isin(spike_clusters, list(valid_ids))
    
    return {
        'stim_times': stim_times,
        'spikes': spike_times[mask],
        'depths': all_depths[mask],
        'layers': layers,
        'bounds': (s1_lower, s1_upper)
    }

# =========================================================
# 3. PLOTTING FUNCTION
# =========================================================
def plot_trial_grid(session, stim):
    data = load_session_data(session, stim)
    if data is None: 
        print(f"Skipping {session} {stim} (Data missing)")
        return

    print(f"Generating Trial Grid for: {session} | {stim}")
    
    stim_times = data['stim_times']
    n_trials = len(stim_times)
    s1_top = data['bounds'][1]
    s1_bottom = data['bounds'][0]
    
    # 1. GENERATE HEATMAPS
    t_bins = np.arange(-PRE_TIME, POST_TIME + BIN_TIME, BIN_TIME)
    d_bins = np.arange(s1_bottom, s1_top + BIN_DEPTH, BIN_DEPTH)
    
    trial_heatmaps = []
    
    for onset in stim_times:
        t0, t1 = onset - PRE_TIME, onset + POST_TIME
        idx0 = np.searchsorted(data['spikes'], t0)
        idx1 = np.searchsorted(data['spikes'], t1)
        
        chunk_t = data['spikes'][idx0:idx1] - onset
        chunk_d = data['depths'][idx0:idx1]
        
        if len(chunk_t) == 0:
            H = np.zeros((len(t_bins)-1, len(d_bins)-1))
        else:
            H, _, _ = np.histogram2d(chunk_t, chunk_d, bins=[t_bins, d_bins])
            
        H = H / BIN_TIME
        H_smooth = gaussian_filter(H, sigma=(1, 1))
        trial_heatmaps.append(H_smooth)

    # 2. VMAX
    if FIXED_VMAX:
        vmax = FIXED_VMAX
    else:
        all_max = [np.percentile(h, 99.5) for h in trial_heatmaps]
        vmax = max(all_max) if all_max else 10
    
    print(f"   Color Scale: 0 - {vmax:.1f} Hz")

    # 3. SETUP GRID
    cols = 4
    rows = math.ceil(n_trials / cols)
    fig = plt.figure(figsize=(16, rows * 3))
    gs = gridspec.GridSpec(rows, cols, wspace=0.1, hspace=0.4)
    
    # 4. PLOT LOOP
    for i in range(n_trials):
        row_idx = i // cols
        col_idx = i % cols
        ax = fig.add_subplot(gs[row_idx, col_idx])
        
        im = ax.imshow(trial_heatmaps[i].T, aspect='auto', cmap='inferno', origin='lower',
                       extent=[t_bins[0], t_bins[-1], d_bins[0], d_bins[-1]],
                       vmax=vmax)
        
        # Overlays
        ax.axvline(0, color='cyan', linestyle='--', alpha=0.8)
        
        # Layers (Merged L5/6)
        current_top = data['bounds'][1]
        layers = data['layers']
        
        for name in ['L1', 'L2/3', 'L4', 'L5/6']:
            bot = layers.get(name, np.nan)
            if not np.isnan(bot):
                ax.axhline(bot, color='white', linestyle='-', linewidth=0.5, alpha=0.5)
                if col_idx == 0:
                    ax.text(-PRE_TIME + 0.02, (current_top + bot)/2, name, 
                            color='white', fontsize=8, va='center', fontweight='bold')
                current_top = bot
        
        # Styling
        t_onset = stim_times[i]
        ax.set_title(f"Trial {i+1} | t={t_onset:.1f}s", fontsize=10)
        
        if row_idx == rows - 1:
            ax.set_xlabel("Time (s)")
        else:
            ax.set_xticklabels([])
            
        # --- Y-AXIS TRANSFORMATION (FORCED 0 START) ---
        if col_idx == 0:
            ax.set_ylabel("Depth (µm)")
            
            # 1. Define Tick Spacing (e.g., every 250 um)
            tick_step = 250
            
            # 2. Generate "Depth" labels starting exactly at 0
            # Example: [0, 250, 500, 750 ...]
            max_depth = s1_top - s1_bottom
            depth_labels = np.arange(0, max_depth, tick_step)
            
            # 3. Convert back to RAW Y coordinates
            # (Raw Y = Top - Depth)
            raw_locs = s1_top - depth_labels
            
            # 4. Apply
            ax.set_yticks(raw_locs)
            ax.set_yticklabels([f"{int(d)}" for d in depth_labels])
            
        else:
            ax.set_yticklabels([])
            
    # Remove empty axes
    for i in range(n_trials, rows * cols):
        fig.delaxes(fig.add_subplot(gs[i // cols, i % cols]))

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7]) 
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label(f"Firing Rate (Hz) | Max: {vmax:.1f}", fontsize=12)
    
    fig.suptitle(f"{session} | {stim} : Single Trial Dynamics", fontsize=16, y=0.99)
    
    # Save
    save_path = Path("results/figures/SingleTrials") / f"{session}_{stim}_Filmstrip.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   Saved: {save_path}")

# =========================================================
# 4. RUN BATCH
# =========================================================
for stim in STIMULI:
    plot_trial_grid(SESSION, stim)