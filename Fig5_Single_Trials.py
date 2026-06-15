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
SESSION     = "a2"
INCLUDE_MUA = True

STIMULI = [
    "brush_tail",
    "brush_contra",
    "brush_ipsi"
]

ANIMAL_NAMES = {
    "a2":   "Adult",
    "pup1": "Pup 1",
    "pup2": "Pup 2"
}
STIM_NAMES = {
    "brush_tail":   "Tail",
    "brush_contra": "Contra",
    "brush_ipsi":   "Ipsi"
}

FIXED_VMAX = 120

PRE_TIME  = 0.1
POST_TIME = 1.5

BIN_TIME  = 0.005
BIN_DEPTH = 20

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

save_dir = Path("results/publication_figures/single_trials")
save_dir.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2. HELPER FUNCTIONS
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
        'L1':   get_val(row, 'l1_end'),
        'L2/3': get_val(row, 'l23_end'),
        'L4':   get_val(row, 'l4_end'),
        'L5/6': get_val(row, 'l6_end')
    }

    stim_path = Path(f"results/summaries/{session}_{stim}_onsets.txt")
    if not stim_path.exists():
        return None
    stim_times = np.loadtxt(stim_path)
    if stim_times.ndim == 0:
        stim_times = [stim_times]

    phy_dir = Path(row['phy_dir'])
    if not (phy_dir / 'spike_times.npy').exists():
        return None

    spike_times    = np.load(phy_dir / 'spike_times.npy').flatten() / 30000.0
    spike_clusters = np.load(phy_dir / 'spike_clusters.npy').flatten()
    cluster_info   = pd.read_csv(phy_dir / 'cluster_info.tsv', sep='\t')

    coords          = np.load(phy_dir / 'channel_positions.npy')
    templates       = np.load(phy_dir / 'templates.npy')
    spike_templates = np.load(phy_dir / 'spike_templates.npy').flatten()

    template_peaks     = np.max(np.abs(templates), axis=1)
    template_max_chans = np.argmax(template_peaks, axis=1)
    all_depths         = coords[template_max_chans[spike_templates], 1]

    lbl          = 'KSLabel' if 'KSLabel' in cluster_info.columns else 'group'
    valid_labels = ['good', 'mua'] if INCLUDE_MUA else ['good']

    s1_units = cluster_info[
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
        'bounds':     (s1_lower, s1_upper)
    }


# =========================================================
# 3. PLOTTING FUNCTION
# =========================================================
def plot_trial_grid(session, stim):
    data = load_session_data(session, stim)
    if data is None:
        print(f"Skipping {session} {stim} (data missing)")
        return

    animal_label = ANIMAL_NAMES.get(session, session)
    stim_label   = STIM_NAMES.get(stim, stim)
    print(f"Generating: {animal_label} | {stim_label}")

    stim_times = data['stim_times']
    n_trials   = len(stim_times)
    s1_top     = data['bounds'][1]
    s1_bottom  = data['bounds'][0]

    # --- Build heatmaps ---
    t_bins = np.arange(-PRE_TIME,  POST_TIME + BIN_TIME,  BIN_TIME)
    d_bins = np.arange(s1_bottom,  s1_top    + BIN_DEPTH, BIN_DEPTH)

    trial_heatmaps = []
    for onset in stim_times:
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

    # --- Color scale ---
    vmax = FIXED_VMAX if FIXED_VMAX else max(
        [np.percentile(h, 99.5) for h in trial_heatmaps] or [10]
    )
    print(f"   Color scale: 0 - {vmax:.1f} Hz")

    # --- Layout ---
    cols = 4
    rows = math.ceil(n_trials / cols)

    # Extra bottom/left margin for shared axis labels
    fig = plt.figure(figsize=(7.2, rows * 2.2))
    gs  = gridspec.GridSpec(
        rows, cols,
        wspace=0.08, hspace=0.35,
        left=0.10, right=0.88,   # left room for shared y-label
        bottom=0.12, top=0.95    # bottom room for shared x-label
    )

    # X-tick positions: balanced around onset, every 0.5s
    # -0.0 (onset), 0.5, 1.0, 1.5
    x_ticks      = np.arange(0, POST_TIME + 0.01, 0.5)
    x_tick_labels = [f"{t:.1f}" for t in x_ticks]

    im = None
    for i in range(n_trials):
        row_idx = i // cols
        col_idx = i % cols
        ax      = fig.add_subplot(gs[row_idx, col_idx])

        im = ax.imshow(
            trial_heatmaps[i].T,
            aspect='auto', cmap='inferno', origin='lower',
            extent=[t_bins[0], t_bins[-1], d_bins[0], d_bins[-1]],
            vmin=0, vmax=vmax
        )

        # Stimulus onset line
        ax.axvline(0, color='cyan', linestyle='--', linewidth=1.0, alpha=1.0)

        # Layer boundaries + labels (left column only)
        current_top = data['bounds'][1]
        for name in ['L1', 'L2/3', 'L4', 'L5/6']:
            bot = data['layers'].get(name, np.nan)
            if not np.isnan(bot):
                ax.axhline(bot, color='white', linestyle='-',
                           linewidth=0.5, alpha=0.5)
                if col_idx == 0:
                    ax.text(
                        -PRE_TIME + 0.005,
                        (current_top + bot) / 2,
                        name,
                        color='white', fontsize=FONT_SIZE - 1,
                        va='center', fontweight='bold',
                        bbox=dict(fc='black', alpha=0.3, pad=0.5, linewidth=0)
                    )
                current_top = bot

        # Panel title
        ax.set_title(f"Trial {i + 1}", fontsize=TITLE_SIZE, pad=3)

        # X-axis ticks — all panels, labels on bottom row only
        ax.set_xticks(x_ticks)
        if row_idx == rows - 1:
            ax.set_xticklabels(x_tick_labels, fontsize=FONT_SIZE - 1)
        else:
            ax.set_xticklabels([])

        # Y-axis — left column only, depth relative to S1 top
        if col_idx == 0:
            tick_step    = 250
            max_depth    = s1_top - s1_bottom
            depth_labels = np.arange(0, max_depth, tick_step)
            raw_locs     = s1_top - depth_labels
            ax.set_yticks(raw_locs)
            ax.set_yticklabels([f"{int(d)}" for d in depth_labels],
                               fontsize=FONT_SIZE - 1)
        else:
            ax.set_yticks([])

        # Remove individual axis labels — shared labels used instead
        ax.set_xlabel("")
        ax.set_ylabel("")

    # Remove empty subplots
    for i in range(n_trials, rows * cols):
        r, c = divmod(i, cols)
        fig.add_subplot(gs[r, c]).axis('off')

    # --- Shared axis labels (no clashing, appear once) ---
    fig.text(0.49, 0.02, "Time (s)",
             ha='center', va='bottom',
             fontsize=FONT_SIZE, fontweight='bold')
    fig.text(0.01, 0.5, "Depth (µm)",
             ha='left', va='center', rotation='vertical',
             fontsize=FONT_SIZE, fontweight='bold')

    # --- Colorbar ---
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    cbar    = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label("Firing Rate (Hz)", fontsize=FONT_SIZE)
    cbar.ax.tick_params(labelsize=FONT_SIZE - 1)

    # --- Save ---
    save_stem = save_dir / f"{session}_{stim}_SingleTrials_publication"
    fig.savefig(f"{save_stem}.tiff", dpi=300, format='tiff', bbox_inches='tight')
    fig.savefig(f"{save_stem}.png",  dpi=300,                bbox_inches='tight')
    plt.close(fig)

    print(f"   Saved: {save_stem}.tiff")
    print(f"   Saved: {save_stem}.png")


# =========================================================
# 4. RUN BATCH
# =========================================================
for stim in STIMULI:
    plot_trial_grid(SESSION, stim)
