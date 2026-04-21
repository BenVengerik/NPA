import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from pathlib import Path

# =========================================================
# CONFIGURATION
# =========================================================
CSV_PATH   = Path("data/sessions.csv")
OUTPUT_DIR = Path("results/unit_summary")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SESSIONS = ['a2', 'pup1', 'pup2']
TARGET_STIMS    = ['brush_contra', 'brush_ipsi', 'brush_tail']
INCLUDE_MUA     = True

ANIMAL_LABELS = {
    'a2':   'Adult',
    'pup1': 'Pup 1',
    'pup2': 'Pup 2',
}
STIM_LABELS = {
    'brush_contra': 'Contralateral',
    'brush_ipsi':   'Ipsilateral',
    'brush_tail':   'Tail',
}

LAYER_COLORS = {
    'L2/3': '#A8C5A0',
    'L4':   '#4A7C59',
    'L5/6': '#1B4332',
}
LAYER_ORDER  = ['L2/3', 'L4', 'L5/6']
ANIMAL_ORDER = ['Adult', 'Pup 1', 'Pup 2']
STIM_ORDER   = ['Contralateral', 'Ipsilateral', 'Tail']

# =========================================================
# FIX 2: assign_layer — l5_end removed (L5 and L6 grouped)
# =========================================================
def assign_layer(depth, l23_end, l4_end, l6_end):
    if depth > l23_end:
        return 'L2/3'
    elif depth > l4_end:
        return 'L4'
    elif depth >= l6_end:
        return 'L5/6'
    else:
        return None

# =========================================================
# LOAD & FILTER SESSIONS
# =========================================================
sessions = pd.read_csv(CSV_PATH, sep=',')
sessions = sessions.dropna(subset=['session', 'stim'])
sessions['session'] = sessions['session'].str.strip().str.lower()
sessions['stim']    = sessions['stim'].str.strip().str.lower()

target_df = sessions[
    sessions['session'].isin(TARGET_SESSIONS) &
    sessions['stim'].isin(TARGET_STIMS)
].copy()

print(f"Found {len(target_df)} session-stimulus combinations.\n")

# =========================================================
# MAIN LOOP
# =========================================================
results = []

for _, row in target_df.iterrows():
    session_id = row['session']
    stim       = row['stim']
    phy_dir    = Path(row['phy_dir']) if pd.notna(row['phy_dir']) else None

    s1_upper = float(row['s1_upper_um']) if pd.notna(row['s1_upper_um']) else np.nan
    s1_lower = float(row['s1_lower_um']) if pd.notna(row['s1_lower_um']) else np.nan
    l23_end  = float(row['l23_end'])     if pd.notna(row['l23_end'])     else np.nan
    l4_end   = float(row['l4_end'])      if pd.notna(row['l4_end'])      else np.nan
    l6_end   = float(row['l6_end'])      if pd.notna(row['l6_end'])      else np.nan

    animal_label = ANIMAL_LABELS.get(session_id, session_id.upper())
    stim_label   = STIM_LABELS.get(stim, stim)

    print(f"Processing: {animal_label} | {stim_label} ...", end=" ")

    if phy_dir is None or not phy_dir.exists():
        print("SKIPPED — phy_dir not found.")
        results.append({
            'animal': animal_label, 'stim': stim_label,
            'total': None, 'L2/3': 0, 'L4': 0, 'L5/6': 0,
            'SU': None, 'MUA': None, 'note': 'phy_dir missing'
        })
        continue

    try:
        channel_positions = np.load(phy_dir / 'channel_positions.npy')
        templates         = np.load(phy_dir / 'templates.npy')
        spike_clusters    = np.load(phy_dir / 'spike_clusters.npy').flatten()
        spike_templates   = np.load(phy_dir / 'spike_templates.npy').flatten().astype(int)

        ks_path = phy_dir / 'cluster_group.tsv'
        if not ks_path.exists():
            ks_path = phy_dir / 'cluster_KSLabel.tsv'
        df_ks = pd.read_csv(ks_path, sep='\t')

    except FileNotFoundError as e:
        print(f"SKIPPED — {e}")
        results.append({
            'animal': animal_label, 'stim': stim_label,
            'total': None, 'L2/3': 0, 'L4': 0, 'L5/6': 0,
            'SU': None, 'MUA': None, 'note': str(e)
        })
        continue

    label_col    = 'KSLabel' if 'KSLabel' in df_ks.columns else 'group'
    valid_labels = ['good', 'mua'] if INCLUDE_MUA else ['good']
    target_ids   = df_ks[df_ks[label_col].isin(valid_labels)]['cluster_id'].values

    unit_records = []
    for cid in target_ids:
        quality = df_ks[df_ks['cluster_id'] == cid][label_col].values[0]
        mask = spike_clusters == cid
        if not np.any(mask):
            continue

        # FIX 3: use np.bincount instead of scipy stats.mode — faster and version-safe
        dominant_template_idx = int(np.bincount(spike_templates[mask]).argmax())
        temp_waveform         = templates[dominant_template_idx]
        peak_channel_idx      = int(np.argmax(np.max(np.abs(temp_waveform), axis=0)))
        y_um                  = channel_positions[peak_channel_idx, 1]

        if np.isnan(s1_upper) or np.isnan(s1_lower):
            continue
        if not (s1_lower <= y_um <= s1_upper):
            continue

        layer = assign_layer(y_um, l23_end, l4_end, l6_end)
        if layer is None:
            continue

        unit_records.append({'quality': quality, 'layer': layer})

    df_units = pd.DataFrame(unit_records)

    if df_units.empty:
        print("WARNING — 0 units in S1.")
        results.append({
            'animal': animal_label, 'stim': stim_label,
            'total': 0, 'L2/3': 0, 'L4': 0, 'L5/6': 0,
            'SU': 0, 'MUA': 0, 'note': 'no units in S1'
        })
        continue

    n_su    = len(df_units[df_units['quality'] == 'good'])
    n_mua   = len(df_units[df_units['quality'] == 'mua'])
    n_total = len(df_units)
    n_l23   = len(df_units[df_units['layer'] == 'L2/3'])
    n_l4    = len(df_units[df_units['layer'] == 'L4'])
    n_l56   = len(df_units[df_units['layer'] == 'L5/6'])

    print(f"OK — {n_total} units ({n_su} SU, {n_mua} MUA) | L2/3:{n_l23} L4:{n_l4} L5/6:{n_l56}")

    results.append({
        'animal': animal_label, 'stim': stim_label,
        'total': n_total, 'L2/3': n_l23, 'L4': n_l4, 'L5/6': n_l56,
        'SU': n_su, 'MUA': n_mua, 'note': 'ok'
    })

df_results = pd.DataFrame(results)
df_results.to_csv(OUTPUT_DIR / "unit_counts_summary.csv", index=False)
print(f"\n✓ Data saved.\n")

# =========================================================
# FIGURE
# =========================================================
pio.templates.default = "plotly_white"

# Build ordered rows top-to-bottom, then reverse for Plotly
ordered_rows = []
for animal in ANIMAL_ORDER:
    for stim in STIM_ORDER:
        ordered_rows.append((animal, stim))

# Reversed for Plotly (reads bottom-to-top)
plot_rows = ordered_rows[::-1]
n_rows    = len(plot_rows)

# FIX 1: use integer Y positions — unique per row, no string collisions
y_positions = list(range(n_rows))

# Tick labels: show stimulus name only
tick_labels = [stim for (_, stim) in plot_rows]

fig = go.Figure()

for layer in LAYER_ORDER:
    x_vals    = []
    text_vals = []

    for (animal, stim) in plot_rows:
        row = df_results[
            (df_results['animal'] == animal) &
            (df_results['stim']   == stim)
        ]
        val = 0
        if not row.empty and row.iloc[0]['note'] == 'ok':
            val = int(row.iloc[0][layer]) if pd.notna(row.iloc[0][layer]) else 0
        x_vals.append(val)
        text_vals.append(str(val) if val > 0 else '')

    fig.add_trace(go.Bar(
        name=layer,
        x=x_vals,
        y=y_positions,           # integer positions — unique per row
        orientation='h',
        marker_color=LAYER_COLORS[layer],
        marker_line_width=0,
        text=text_vals,
        textposition='inside',
        insidetextanchor='middle',
        textfont=dict(color='white', size=12, family='Arial'),
    ))

# ── Animal group annotations ───────────────────────────────────────────────
annotations = []
for animal in ANIMAL_ORDER:
    indices = [i for i, (a, _) in enumerate(plot_rows) if a == animal]
    if not indices:
        continue
    mid_pos = indices[len(indices) // 2]
    annotations.append(dict(
        x=-0.22,              # <-- pushed further left, away from tick labels
        y=mid_pos,
        xref='paper',
        yref='y',
        text=f"<b>{animal}</b>",
        showarrow=False,
        xanchor='right',
        yanchor='middle',
        font=dict(size=13, family='Arial', color='#1B1B1B'),
    ))

# ── Divider lines between animal groups ───────────────────────────────────
# Find where one animal ends and the next begins in y_positions
shapes = []
for i in range(len(plot_rows) - 1):
    current_animal = plot_rows[i][0]
    next_animal    = plot_rows[i + 1][0]
    if current_animal != next_animal:
        # Midpoint between the two integer positions
        divider_y = i + 0.5
        shapes.append(dict(
            type='line',
            x0=0, x1=1, xref='paper',
            y0=divider_y, y1=divider_y, yref='y',
            line=dict(color='#BBBBBB', width=1.2, dash='dot'),
        ))

fig.update_layout(
    barmode='stack',
    height=500,
    width=660,
    plot_bgcolor='white',
    paper_bgcolor='white',
    annotations=annotations,
    shapes=shapes,
    bargap=0.25,
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.03,
        xanchor='left',
        x=0,
        font=dict(size=12, family='Arial'),
        traceorder='normal',
        title=dict(text='Layer   ', font=dict(size=12, family='Arial')),
    ),
    margin=dict(l=180, r=40, t=60, b=60),  
    xaxis=dict(
        title=dict(
            text='Unit Count',
            font=dict(size=13, family='Arial'),
            standoff=15,
        ),
        tickfont=dict(size=12, family='Arial'),
        showgrid=True,
        gridcolor='#EEEEEE',
        zeroline=False,
    ),
    yaxis=dict(
        tickmode='array',
        tickvals=y_positions,    # integer positions
        ticktext=tick_labels,    # displayed stimulus labels
        tickfont=dict(size=12, family='Arial'),
        showgrid=False,
        ticklabelstandoff=10,
        range=[-0.5, n_rows - 0.5],
    ),
)

fig.update_traces(cliponaxis=False)

fig_path = OUTPUT_DIR / "unit_counts_by_layer.png"
fig.write_image(str(fig_path), scale=3)
print(f"✓ Figure saved to: {fig_path}")