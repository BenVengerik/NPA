import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import gaussian_filter

# =========================================================
# 1. CONFIGURATION
# =========================================================
ANIMALS  = ["a2", "pup1", "pup2"]
STIMULI  = ["brush_contra", "brush_ipsi", "brush_tail"]
INCLUDE_MUA = True

ANIMAL_NAMES = {
    "a2":   "Adult",
    "pup1": "Pup 1",
    "pup2": "Pup 2",
}
STIM_NAMES = {
    "brush_contra": "Contralateral",
    "brush_ipsi":   "Ipsilateral",
    "brush_tail":   "Tail",
}

FIXED_VMAX = 120
PRE_TIME   = 0.1
POST_TIME  = 1.5
BIN_TIME   = 0.005
BIN_DEPTH  = 20

FONT_SIZE  = 9
TITLE_SIZE = 10

plt.rcParams.update({
    'font.size':       FONT_SIZE,
    'font.family':     'Arial',
    'axes.titlesize':  TITLE_SIZE,
    'axes.labelsize':  FONT_SIZE,
    'xtick.labelsize': FONT_SIZE - 1,
    'ytick.labelsize': FONT_SIZE - 1,
})

save_dir = Path("results/publication_figures")
save_dir.mkdir(parents=True, exist_ok=True)

# =========================================================
# 2. HELPERS
# =========================================================
def get_val(r, col):
    return float(r[col]) if col in r and not pd.isna(r[col]) else np.nan


def load_session_data(session, stim):
    csv_path = (
        Path("data/sessions_local.csv")
        if Path("data/sessions_local.csv").exists()
        else Path("data/sessions.csv")
    )
    meta = pd.read_csv(csv_path)

    row = meta.loc[(meta['session'] == session) & (meta['stim'] == stim)]
    if len(row) == 0:
        return None
    row = row.iloc[0]

    s1_upper = get_val(row, 's1_upper_um')
    s1_lower = get_val(row, 's1_lower_um')
    layers = {
        'L2/3': get_val(row, 'l23_end'),
        'L4':   get_val(row, 'l4_end'),
        'L5/6': get_val(row, 'l6_end'),
    }

    stim_path = Path(f"results/summaries/{session}_{stim}_onsets.txt")
    if not stim_path.exists():
        return None
    stim_times = np.loadtxt(stim_path)
    if stim_times.ndim == 0:
        stim_times = np.array([float(stim_times)])

    phy_dir = Path(row['phy_dir'])
    if not (phy_dir / 'spike_times.npy').exists():
        return None

    spike_times    = np.load(phy_dir / 'spike_times.npy').flatten() / 30000.0
    spike_clusters = np.load(phy_dir / 'spike_clusters.npy').flatten()
    cluster_info   = pd.read_csv(phy_dir / 'cluster_info.tsv', sep='\t')

    coords          = np.load(phy_dir / 'channel_positions.npy')
    templates       = np.load(phy_dir / 'templates.npy')
    spike_templates = np.load(phy_dir / 'spike_templates.npy').flatten().astype(int)

    template_peaks     = np.max(np.abs(templates), axis=1)
    template_max_chans = np.argmax(template_peaks, axis=1)
    all_depths         = coords[template_max_chans[spike_templates], 1]

    lbl          = 'KSLabel' if 'KSLabel' in cluster_info.columns else 'group'
    valid_labels = ['good', 'mua'] if INCLUDE_MUA else ['good']

    s1_units  = cluster_info[
        (cluster_info['depth'] >= s1_lower) &
        (cluster_info['depth'] <= s1_upper) &
        (cluster_info[lbl].isin(valid_labels))
    ]
    valid_ids = set(s1_units['cluster_id'])
    mask      = np.isin(spike_clusters, list(valid_ids))

    return {
        'stim_times': stim_times,
        'spikes':     spike_times[mask],
        'depths':     all_depths[mask],
        'layers':     layers,
        'bounds':     (s1_lower, s1_upper),
    }


def compute_mean_heatmap(data):
    """Average the smoothed heatmap across all trials."""
    s1_bottom, s1_top = data['bounds']
    t_bins = np.arange(-PRE_TIME, POST_TIME + BIN_TIME, BIN_TIME)
    d_bins = np.arange(s1_bottom, s1_top + BIN_DEPTH, BIN_DEPTH)

    trial_heatmaps = []
    for onset in data['stim_times']:
        t0, t1 = onset - PRE_TIME, onset + POST_TIME
        idx0   = np.searchsorted(data['spikes'], t0)
        idx1   = np.searchsorted(data['spikes'], t1)
        chunk_t = data['spikes'][idx0:idx1] - onset
        chunk_d = data['depths'][idx0:idx1]

        if len(chunk_t) == 0:
            H = np.zeros((len(t_bins) - 1, len(d_bins) - 1))
        else:
            H, _, _ = np.histogram2d(chunk_t, chunk_d, bins=[t_bins, d_bins])

        H        = H / BIN_TIME
        H_smooth = gaussian_filter(H, sigma=(1, 1))
        trial_heatmaps.append(H_smooth)

    mean_H = np.mean(trial_heatmaps, axis=0)
    return mean_H, t_bins, d_bins

# =========================================================
# 3. BUILD FIGURE  (rows = animals, cols = stimuli)
# =========================================================
n_rows = len(ANIMALS)
n_cols = len(STIMULI)

# Panel height scales with cortical depth range — adult panels taller than pup
# Use a fixed height per panel; width per panel is uniform
PANEL_W = 2.4
PANEL_H = 2.8

fig = plt.figure(figsize=(
    n_cols * PANEL_W + 1.2,   # +1.2 for colorbar
    n_rows * PANEL_H + 0.6,   # +0.6 for column title row
))

gs = gridspec.GridSpec(
    n_rows, n_cols,
    wspace=0.08,
    hspace=0.35,
    left=0.10,
    right=0.88,
    bottom=0.09,
    top=0.93,
)

x_ticks       = np.arange(0, POST_TIME + 0.01, 0.5)
x_tick_labels = [f"{t:.1f}" for t in x_ticks]

im_last = None   # keep reference for colorbar

for row_idx, session in enumerate(ANIMALS):
    for col_idx, stim in enumerate(STIMULI):

        ax = fig.add_subplot(gs[row_idx, col_idx])

        data = load_session_data(session, stim)
        animal_label = ANIMAL_NAMES.get(session, session)
        stim_label   = STIM_NAMES.get(stim, stim)

        if data is None:
            ax.text(0.5, 0.5, "No data", ha='center', va='center',
                    transform=ax.transAxes, color='grey', fontsize=FONT_SIZE)
            ax.set_xticks([])
            ax.set_yticks([])
            # Still draw border so the grid looks complete
            for spine in ax.spines.values():
                spine.set_edgecolor('#cccccc')
            # Column headers (top row only)
            if row_idx == 0:
                ax.set_title(stim_label, fontsize=TITLE_SIZE,
                             fontweight='bold', pad=6)
            continue

        mean_H, t_bins, d_bins = compute_mean_heatmap(data)
        s1_bottom, s1_top = data['bounds']

        im_last = ax.imshow(
            mean_H.T,
            aspect='auto',
            cmap='inferno',
            origin='lower',
            extent=[t_bins[0], t_bins[-1], d_bins[0], d_bins[-1]],
            vmin=0,
            vmax=FIXED_VMAX,
        )

        # Stimulus onset line
        ax.axvline(0, color='cyan', linestyle='--', linewidth=1.0, alpha=0.9)

        # Layer boundaries
        current_top = s1_top
        for name in ['L2/3', 'L4', 'L5/6']:
            bot = data['layers'].get(name, np.nan)
            if not np.isnan(bot):
                ax.axhline(bot, color='white', linestyle='-',
                           linewidth=0.6, alpha=0.6)
                # Layer labels on leftmost column only
                if col_idx == 0:
                    ax.text(
                        -PRE_TIME + 0.005,
                        (current_top + bot) / 2,
                        name,
                        color='white',
                        fontsize=FONT_SIZE - 1,
                        va='center',
                        fontweight='bold',
                        bbox=dict(fc='black', alpha=0.3, pad=0.5, linewidth=0),
                    )
                current_top = bot

        # ── Column headers: stimulus name, top row only ──
        if row_idx == 0:
            ax.set_title(stim_label, fontsize=TITLE_SIZE,
                         fontweight='bold', pad=6)

        # ── Row headers: animal name, leftmost column only ──
        if col_idx == 0:
            ax.set_ylabel(
                animal_label,
                fontsize=TITLE_SIZE,
                fontweight='bold',
                labelpad=8,
            )

        # ── X-axis: ticks on all panels, labels on bottom row only ──
        ax.set_xticks(x_ticks)
        if row_idx == n_rows - 1:
            ax.set_xticklabels(x_tick_labels, fontsize=FONT_SIZE - 1)
        else:
            ax.set_xticklabels([])

        # ── Y-axis: depth ticks on leftmost column only ──
        if col_idx == 0:
            tick_step    = 200
            max_depth    = s1_top - s1_bottom
            depth_labels = np.arange(0, max_depth + 1, tick_step)
            raw_locs     = s1_top - depth_labels
            ax.set_yticks(raw_locs)
            ax.set_yticklabels(
                [f"{int(d)}" for d in depth_labels],
                fontsize=FONT_SIZE - 1,
            )
        else:
            ax.set_yticks([])

        ax.set_xlim(t_bins[0], t_bins[-1])
        ax.set_ylim(s1_bottom, s1_top)

# ── Shared axis labels ──────────────────────────────────────────────────────
fig.text(
    0.49, 0.02,
    "Time from stimulus onset (s)",
    ha='center', va='bottom',
    fontsize=FONT_SIZE, fontweight='bold',
)
fig.text(
    0.01, 0.5,
    "Depth from S1 surface (µm)",
    ha='left', va='center',
    rotation='vertical',
    fontsize=FONT_SIZE, fontweight='bold',
)

# ── Single shared colorbar ──────────────────────────────────────────────────
if im_last is not None:
    cbar_ax = fig.add_axes([0.90, 0.12, 0.015, 0.75])
    cbar    = plt.colorbar(im_last, cax=cbar_ax)
    cbar.set_label("Mean Firing Rate (Hz)", fontsize=FONT_SIZE)
    cbar.ax.tick_params(labelsize=FONT_SIZE - 1)

# ── Save ────────────────────────────────────────────────────────────────────
save_stem = save_dir / "raster_heatmap_3x3"
fig.savefig(f"{save_stem}.tiff", dpi=300, format='tiff', bbox_inches='tight')
fig.savefig(f"{save_stem}.png",  dpi=300,                bbox_inches='tight')
plt.close(fig)

print(f"✓ Saved: {save_stem}.tiff")
print(f"✓ Saved: {save_stem}.png")