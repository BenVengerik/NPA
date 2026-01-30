import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import gaussian_filter


# =========================================================
# 1. CONFIGURATION
# =========================================================
SESSION = "a1"         # animal id
INCLUDE_MUA = True       

# Define all stimuli to process
STIMULI = [
    "brush_contra",
    "brush_ipsi", 
    "brush_tail",
    "pinch_contra",
    "pinch_ipsi",
    "pinch_tail"
]

# Time Window
PRE_TIME = 0.25    # Seconds before stim
POST_TIME = 1.25    # Seconds after stim

# Heatmap Bins
BIN_TIME = 0.005   # 5ms time bins
BIN_DEPTH = 20     # 20µm depth bins

# Colors 
LAYER_COLORS = {
    'L1': '#E8F5E9', 
    'L2/3': '#C8E6C9', 
    'L4': '#A5D6A7', 
    'L5': '#81C784', 
    'L6': '#66BB6A'
}


# =========================================================
# 2. HELPER FUNCTION
# =========================================================
def get_val(r, col):
    return float(r[col]) if col in r and not pd.isna(r[col]) else np.nan


# =========================================================
# 3. PROCESSING FUNCTION
# =========================================================
def process_stimulus(session, stim):
    """
    Generate integrated visualization for one stimulus.
    Returns True on success, False on failure.
    """
    try:
        print(f"\n{'='*60}")
        print(f"PROCESSING: {session} | {stim}")
        print(f"{'='*60}")

        # Load metadata
        csv_path = Path("data/sessions_local.csv") if Path("data/sessions_local.csv").exists() else Path("data/sessions.csv")
        sessions = pd.read_csv(csv_path)
        row = sessions.loc[(sessions['session'].str.lower() == session.lower()) & 
                           (sessions['stim'].str.lower() == stim.lower())]

        if len(row) == 0:
            print(f"⚠️  No metadata found for {session} | {stim}. Skipping.")
            return False

        row = row.iloc[0]
        phy_dir = Path(row['phy_dir'])

        # Get anatomy
        s1_upper = get_val(row, 's1_upper_um')
        s1_lower = get_val(row, 's1_lower_um')

        if np.isnan(s1_upper) or np.isnan(s1_lower):
            print(f"⚠️  Missing S1 boundaries for {session} | {stim}. Skipping.")
            return False

        layers = {
            'L1':   get_val(row, 'l1_end'),
            'L2/3': get_val(row, 'l23_end'),
            'L4':   get_val(row, 'l4_end'),
            'L5':   get_val(row, 'l5_end'),
            'L6':   get_val(row, 'l6_end')
        }

        # Load stimulus times
        stim_path = Path(f"results/summaries/{session}_{stim}_onsets.txt")
        if not stim_path.exists():
            print(f"⚠️  Onset file not found: {stim_path}. Skipping.")
            return False

        stim_times = np.loadtxt(stim_path)
        if stim_times.ndim == 0: 
            stim_times = [stim_times]

        print(f"✓ Loaded {len(stim_times)} stimulus onsets")

        # Load spikes
        spike_times = np.load(phy_dir / 'spike_times.npy').flatten()
        spike_clusters = np.load(phy_dir / 'spike_clusters.npy').flatten()
        spike_fs = 30000.0
        spike_times_sec = spike_times / spike_fs

        # Cluster info
        cluster_info = pd.read_csv(phy_dir / 'cluster_info.tsv', sep='\t')
        label_col = 'KSLabel' if 'KSLabel' in cluster_info.columns else 'group'

        # Templates
        coords = np.load(phy_dir / 'channel_positions.npy')
        templates = np.load(phy_dir / 'templates.npy')
        spike_templates = np.load(phy_dir / 'spike_templates.npy').flatten()

        # ===== RASTER DATA =====
        valid_labels = ['good', 'mua'] if INCLUDE_MUA else ['good']

        s1_units = cluster_info[
            (cluster_info['depth'] >= s1_lower) & 
            (cluster_info['depth'] <= s1_upper) &
            (cluster_info[label_col].isin(valid_labels))
        ]

        if len(s1_units) == 0:
            print(f"⚠️  No valid units found in S1. Skipping.")
            return False

        unit_depth_map = dict(zip(s1_units['cluster_id'], s1_units['depth']))

        mask_s1_spikes = np.isin(spike_clusters, s1_units['cluster_id'])
        s1_spike_times = spike_times_sec[mask_s1_spikes]
        s1_spike_ids = spike_clusters[mask_s1_spikes]

        print(f"✓ Processing {len(s1_units)} S1 units")

        raster_data = []
        for t_stim in stim_times:
            t_start = t_stim - PRE_TIME
            t_end = t_stim + POST_TIME

            idx_start = np.searchsorted(s1_spike_times, t_start)
            idx_end = np.searchsorted(s1_spike_times, t_end)

            window_spikes = s1_spike_times[idx_start:idx_end]
            window_ids = s1_spike_ids[idx_start:idx_end]

            rel_times = window_spikes - t_stim
            depths = [unit_depth_map[uid] for uid in window_ids]

            raster_data.append(pd.DataFrame({
                'time': rel_times,
                'depth': depths
            }))

        df_raster = pd.concat(raster_data) if raster_data else pd.DataFrame()
        print(f"✓ Raster: {len(df_raster)} total spikes")

        # ===== HEATMAP DATA =====
        template_peaks = np.max(np.abs(templates), axis=1) 
        template_max_chans = np.argmax(template_peaks, axis=1)
        all_spike_depths = coords[template_max_chans[spike_templates], 1]

        good_ids = set(s1_units['cluster_id'])
        mask_good = np.isin(spike_clusters, list(good_ids))
        spike_times_filtered = spike_times_sec[mask_good]
        spike_depths_filtered = all_spike_depths[mask_good]

        t_bins = np.arange(-PRE_TIME, POST_TIME + BIN_TIME, BIN_TIME)
        d_bins = np.arange(s1_lower, s1_upper + BIN_DEPTH, BIN_DEPTH)

        heatmap_2d = np.zeros((len(t_bins)-1, len(d_bins)-1))

        for onset in stim_times:
            t_start, t_end = onset - PRE_TIME, onset + POST_TIME

            idx_start = np.searchsorted(spike_times_filtered, t_start)
            idx_end = np.searchsorted(spike_times_filtered, t_end)

            st_win = spike_times_filtered[idx_start:idx_end]
            sd_win = spike_depths_filtered[idx_start:idx_end]

            if len(st_win) == 0: 
                continue

            aligned_times = st_win - onset
            H, _, _ = np.histogram2d(aligned_times, sd_win, bins=[t_bins, d_bins])
            heatmap_2d += H

        heatmap_2d = (heatmap_2d / len(stim_times)) / BIN_TIME
        heatmap_smooth = gaussian_filter(heatmap_2d, sigma=(1, 1))
        vmax = np.percentile(heatmap_smooth, 99.9)

        print(f"✓ Heatmap: max {np.max(heatmap_smooth):.1f} Hz, robust max {vmax:.1f} Hz")

        # ===== DEPTH PROFILE =====
        depth_profile = np.mean(heatmap_smooth, axis=0)
        depth_centers = (d_bins[:-1] + d_bins[1:]) / 2

        # ===== CREATE FIGURE =====
        fig = plt.figure(figsize=(16, 8))
        gs = gridspec.GridSpec(1, 3, width_ratios=[2, 2, 0.6], wspace=0.08)

        ax_raster = fig.add_subplot(gs[0])
        ax_heatmap = fig.add_subplot(gs[1], sharey=ax_raster)
        ax_profile = fig.add_subplot(gs[2], sharey=ax_raster)

        # === LEFT PANEL - RASTER ===
        current_top = s1_upper
        for layer_name in ['L1', 'L2/3', 'L4', 'L5', 'L6']:
            layer_bot = layers[layer_name]
            if not np.isnan(layer_bot):
                ax_raster.axhspan(layer_bot, current_top, 
                                 color=LAYER_COLORS.get(layer_name, 'white'), 
                                 alpha=0.5, lw=0)
                ax_raster.axhline(layer_bot, color='white', linestyle='-', linewidth=1)
                current_top = layer_bot

        if len(df_raster) > 0:
            ax_raster.scatter(df_raster['time'], df_raster['depth'], 
                             c='black', s=6, alpha=0.6, edgecolors='none')

        ax_raster.axvline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.8, label='Stimulus')
        ax_raster.set_xlim(-PRE_TIME, POST_TIME)
        ax_raster.set_ylim(s1_lower - 50, s1_upper + 50)
        ax_raster.set_xlabel("Time from Stimulus (s)", fontsize=12)
        ax_raster.set_ylabel("Cortical Depth (µm)", fontsize=12)
        ax_raster.set_title("Spike Raster", fontsize=13, fontweight='bold')
        ax_raster.legend(loc='upper right')

        # === MIDDLE PANEL - HEATMAP ===
        im = ax_heatmap.imshow(heatmap_smooth.T, aspect='auto', cmap='inferno', origin='lower',
                               extent=[t_bins[0], t_bins[-1], d_bins[0], d_bins[-1]], vmax=vmax)

        cbar = plt.colorbar(im, ax=ax_heatmap)
        cbar.set_label("Firing Rate (Hz)", fontsize=11)

        for layer_name in ['L1', 'L2/3', 'L4', 'L5', 'L6']:
            layer_bot = layers[layer_name]
            if not np.isnan(layer_bot):
                ax_heatmap.axhline(layer_bot, color='white', linestyle='-', linewidth=1.5, alpha=0.8)

        ax_heatmap.axvline(0, color='cyan', linestyle='--', linewidth=1.5, alpha=0.9)
        ax_heatmap.set_xlim(-PRE_TIME, POST_TIME)
        ax_heatmap.set_xlabel("Time from Stimulus (s)", fontsize=12)
        ax_heatmap.set_title("Population Activity (PSTH)", fontsize=13, fontweight='bold')
        plt.setp(ax_heatmap.get_yticklabels(), visible=False)

        # === RIGHT PANEL - PROFILE ===
        current_top = s1_upper
        for layer_name in ['L1', 'L2/3', 'L4', 'L5', 'L6']:
            layer_bot = layers[layer_name]
            if not np.isnan(layer_bot):
                ax_profile.axhline(layer_bot, color='gray', linestyle=':', linewidth=1, alpha=0.6)
                layer_mid = (current_top + layer_bot) / 2
                ax_profile.text(np.max(depth_profile) * 1.15, layer_mid, 
                               layer_name, ha='left', va='center',
                               fontsize=9, fontweight='bold', color='black')
                current_top = layer_bot

        ax_profile.fill_betweenx(depth_centers, 0, depth_profile, color='darkblue', alpha=0.7)
        ax_profile.plot(depth_profile, depth_centers, color='navy', linewidth=2)
        ax_profile.set_xlim(0, np.max(depth_profile) * 1.3)
        ax_profile.set_xlabel("Avg Rate (Hz)", fontsize=10)
        ax_profile.set_title("Laminar\nProfile", fontsize=11, fontweight='bold')
        ax_profile.grid(axis='x', alpha=0.3, linestyle=':')
        plt.setp(ax_profile.get_yticklabels(), visible=False)

        # === FINALIZE ===
        fig.suptitle(f"Cortical Response: {session} | {stim}\n{len(stim_times)} Trials", 
                    fontsize=15, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save
        save_path = Path("results/figures/Cortex Master Plot") / f"{session}_{stim}_CortexMasterPlot.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)  # Close to free memory

        print(f"✅ SUCCESS! Saved to: {save_path}")
        return True

    except Exception as e:
        print(f"❌ ERROR processing {session} | {stim}:")
        print(f"   {type(e).__name__}: {str(e)}")
        return False


# =========================================================
# 4. BATCH PROCESSING
# =========================================================
print("="*60)
print("BATCH PROCESSING: INTEGRATED VISUALIZATIONS")
print("="*60)

results = {}

for stim in STIMULI:
    success = process_stimulus(SESSION, stim)
    results[stim] = success

# =========================================================
# 5. SUMMARY
# =========================================================
print("\n" + "="*60)
print("BATCH PROCESSING COMPLETE")
print("="*60)

successful = [s for s, success in results.items() if success]
failed = [s for s, success in results.items() if not success]

print(f"\n✅ Successful: {len(successful)}/{len(STIMULI)}")
for s in successful:
    print(f"   - {s}")

if failed:
    print(f"\n❌ Failed: {len(failed)}/{len(STIMULI)}")
    for s in failed:
        print(f"   - {s}")

print("\n" + "="*60)
