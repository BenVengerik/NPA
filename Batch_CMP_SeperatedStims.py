import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
    "brush_tail",      # Start with the strongest response
    "brush_contra",
    "brush_ipsi"
]

# --- MANUAL SETTINGS ---
# Set this to standardize colors (Hz). 
# If None, it calculates a "Global Max" from the data itself.
FIXED_VMAX = 25  # e.g., 50. Set to None for auto-scaling per session.

# Time Window (Zoomed in for single trials)
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
    csv_path = Path("data/sessions_local.csv") if Path("data/sessions_local.csv").exists() else Path("data/sessions.csv")
    meta = pd.read_csv(csv_path)
    row = meta.loc[(meta['session'] == session) & (meta['stim'] == stim)]
    if len(row) == 0: return None
    row = row.iloc[0]
    
    # Load Anatomy
    s1_upper = get_val(row, 's1_upper_um')
    s1_lower = get_val(row, 's1_lower_um')
    layers = {
        'L1':   get_val(row, 'l1_end'),
        'L2/3': get_val(row, 'l23_end'),
        'L4':   get_val(row, 'l4_end'),
        'L5':   get_val(row, 'l5_end'),
        'L6':   get_val(row, 'l6_end')
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
    
    # 1. GENERATE HEATMAPS FOR ALL TRIALS
    t_bins = np.arange(-PRE_TIME, POST_TIME + BIN_TIME, BIN_TIME)
    d_bins = np.arange(data['bounds'][0], data['bounds'][1] + BIN_DEPTH, BIN_DEPTH)
    
    trial_heatmaps = []
    
    for onset in stim_times:
        t0, t1 = onset - PRE_TIME, onset + POST_TIME
        idx0 = np.searchsorted(data['spikes'], t0)
        idx1 = np.searchsorted(data['spikes'], t1)
        
        # Extract spike chunk
        chunk_t = data['spikes'][idx0:idx1] - onset
        chunk_d = data['depths'][idx0:idx1]
        
        if len(chunk_t) == 0:
            H = np.zeros((len(t_bins)-1, len(d_bins)-1))
        else:
            H, _, _ = np.histogram2d(chunk_t, chunk_d, bins=[t_bins, d_bins])
            
        # Smooth and convert to Hz
        H = H / BIN_TIME # Raw count -> Hz
        H_smooth = gaussian_filter(H, sigma=(1, 1))
        trial_heatmaps.append(H_smooth)

    # 2. DETERMINE GLOBAL VMAX
    # We want all subplots to share the same color scale
    if FIXED_VMAX:
        vmax = FIXED_VMAX
    else:
        # Find the robust max across all trials combined
        all_max = [np.percentile(h, 99.5) for h in trial_heatmaps]
        vmax = max(all_max) if all_max else 10
    
    print(f"   Color Scale: 0 - {vmax:.1f} Hz")

    # 3. SETUP GRID
    cols = 4
    rows = math.ceil(n_trials / cols)
    
    # Dynamic Figure Height: 3 inches per row
    fig = plt.figure(figsize=(16, rows * 3))
    gs = gridspec.GridSpec(rows, cols, wspace=0.1, hspace=0.3)
    
    # 4. PLOT LOOP
    for i in range(n_trials):
        row_idx = i // cols
        col_idx = i % cols
        ax = fig.add_subplot(gs[row_idx, col_idx])
        
        # Plot Heatmap
        im = ax.imshow(trial_heatmaps[i].T, aspect='auto', cmap='inferno', origin='lower',
                       extent=[t_bins[0], t_bins[-1], d_bins[0], d_bins[-1]],
                       vmax=vmax)
        
        # Overlays
        ax.axvline(0, color='cyan', linestyle='--', alpha=0.8)
        
        # Draw Layers
        current_top = data['bounds'][1]
        layers = data['layers']
        for name in ['L1', 'L2/3', 'L4', 'L5', 'L6']:
            bot = layers.get(name, np.nan)
            if not np.isnan(bot):
                ax.axhline(bot, color='white', linestyle='-', linewidth=0.5, alpha=0.5)
                # Add text only on the first column to reduce clutter
                if col_idx == 0:
                    ax.text(-PRE_TIME + 0.02, (current_top + bot)/2, name, 
                            color='white', fontsize=8, va='center', fontweight='bold')
                current_top = bot
        
        # Styling
        ax.set_title(f"Trial {i+1}", fontsize=10, fontweight='bold', color='white', 
                     bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))
        
        if row_idx == rows - 1:
            ax.set_xlabel("Time (s)")
        else:
            ax.set_xticklabels([])
            
        if col_idx == 0:
            ax.set_ylabel("Depth (µm)")
        else:
            ax.set_yticklabels([])
            
    # Remove empty axes if any
    for i in range(n_trials, rows * cols):
        fig.delaxes(fig.add_subplot(gs[i // cols, i % cols]))

    # Add Colorbar (Global)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7]) # x, y, width, height
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