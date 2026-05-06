"""
Unified statistics visualiser — all figures in one script.
Outputs both PNG and SVG for every figure.

Figures produced
────────────────
Furniture (reads furniture_stats_tables/t7_raw.csv):
  A3_faces_violin.svg/png      — face count half-violin, whole dataset
  A3_edges_violin.svg/png      — edge count half-violin, whole dataset
  A3_vertices_violin.svg/png   — vertex count half-violin, whole dataset
  A3_faces_bycat.svg/png       — face count violin by category
  A3_edges_bycat.svg/png       — edge count violin by category
  A3_vertices_bycat.svg/png    — vertex count violin by category

Scene (reads scene_stats_tables/t7_raw.csv):
  S1_room_type_counts.svg/png  — horizontal bar, scene count per room type
  S2_furniture_count.svg/png   — furniture count half-violin, whole dataset
  S3_scene_faces.svg/png       — scene face count half-violin, whole dataset

Usage:
  python plot_all.py            # all figures
  python plot_all.py --furniture
  python plot_all.py --scenes
"""

import os, sys, colorsys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde

# ══════════════════════════════════════════════════════════════════════════════
#  STYLE CONFIGURATION  ← edit anything in this block
# ══════════════════════════════════════════════════════════════════════════════

# ── Font ──────────────────────────────────────────────────────────────────────
# Common choices: 'DejaVu Sans', 'Arial', 'Helvetica', 'Times New Roman',
#                 'Georgia', 'Palatino', 'Courier New'
FONT_FAMILY      = 'DejaVu Sans'
FONT_SIZE_BASE   = 11    # axis labels, tick labels
FONT_SIZE_TITLE  = 13    # figure / subplot titles
FONT_SIZE_ANNOT  = 8     # P25/P50/P75 labels and median annotations
FONT_SIZE_LEGEND = 9     # legend text
FONT_SIZE_STATS  = 9     # top-right summary box

# ── Resolution & figure sizes ─────────────────────────────────────────────────
FIGURE_DPI   = 150       # screen preview DPI
SAVE_DPI     = 200       # PNG save DPI  (use 300 for publication)
FIG_WIDE     = 11        # inches — wide single-strip figures (half-violins)
FIG_TALL     = 3.2       # inches — height of single-strip figures
FIG_CAT_W    = 13        # inches — per-category violin figures
FIG_CAT_H    = 6         # inches

# ── Global colour saturation multiplier ───────────────────────────────────────
# 1.0 = original colour  |  < 1.0 = more muted  |  > 1.0 = more vivid
COLOR_SATURATION = 1.0

# ── Base colours (hex) — saturation applied on top ────────────────────────────
# Furniture half-violin (one per metric)
COLOR_FACES      = '#4C72B0'
COLOR_EDGES      = '#DD8452'
COLOR_VERTICES   = '#55A868'

# Scene figures
COLOR_SCENE_FURN = '#DD8452'
COLOR_SCENE_FACE = '#55A868'

# Per-category palette (9 categories)
CAT_COLORS_BASE = ['#4C72B0','#DD8452','#55A868','#C44E52','#8172B3',
                   '#937860','#DA8BC3','#8C8C8C','#64B5CD']

# Room-type palette (16 room types)
ROOM_COLORS_BASE = {
    'LivingDiningRoom': '#4C72B0', 'LivingRoom':    '#5588CC',
    'MasterBedroom':    '#DD8452', 'Bedroom':       '#E8A080',
    'SecondBedroom':    '#F0C080', 'KidsRoom':      '#55A868',
    'Library':          '#8172B3', 'DiningRoom':    '#C44E52',
    'Corridor':         '#8C8C8C', 'CloakRoom':     '#937860',
    'OtherRoom':        '#DA8BC3', 'Hallway':       '#64B5CD',
    'ElderlyRoom':      '#CCB974', 'Lounge':        '#95C8D8',
    'StorageRoom':      '#B0C4B1', 'Unknown':       '#CCCCCC',
}

# ── Violin shape ──────────────────────────────────────────────────────────────
VIOLIN_ALPHA       = 0.65   # fill transparency
VIOLIN_EDGE_LW     = 1.2    # outline linewidth
VIOLIN_EDGE_ALPHA  = 0.90
KDE_BW_METHOD      = 'scott'   # 'scott', 'silverman', or float e.g. 0.3
CLIP_QUANTILE      = 0.995  # clip above this quantile before KDE

# ── Rug / scatter strip ───────────────────────────────────────────────────────
RUG_SIZE           = 2.5    # dot size pt²
RUG_ALPHA          = 0.18
RUG_JITTER_LOW     = -0.08
RUG_JITTER_HIGH    = 0.0

# ── Percentile markers ────────────────────────────────────────────────────────
PERC_COLOR         = 'black'
PERC_ALPHA         = 0.75
PERC_YMIN          = 0.05   # axes fraction inside violin height
PERC_YMAX          = 0.55

# ── Grid ──────────────────────────────────────────────────────────────────────
GRID_STYLE         = '--'
GRID_ALPHA         = 0.30

# ── Stats summary box ─────────────────────────────────────────────────────────
STATS_TEXT_COLOR   = '#444444'
STATS_BG_COLOR     = 'white'
STATS_BG_ALPHA     = 0.70

# ── S1 bar chart ─────────────────────────────────────────────────────────────
S1_MERGE_THRESHOLD = 30     # room types with fewer scenes → merged into OtherRoom

# ══════════════════════════════════════════════════════════════════════════════
#  PATHS
# ══════════════════════════════════════════════════════════════════════════════
BASE          = r"D:\Lzm_Temp_Data\Li_temp_project"
FURN_CSV      = os.path.join(BASE, "furniture_stats_tables", "t7_raw.csv")
SCENE_CSV     = os.path.join(BASE, "scene_stats_tables",    "t7_raw.csv")
OUT_DIR       = os.path.join(BASE, "stats_figures_v2")
os.makedirs(OUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
#  COLOUR UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
def _sat(hex_color, factor=COLOR_SATURATION):
    """Adjust HSV saturation of a hex colour by factor (clamped 0–1)."""
    h = hex_color.lstrip('#')
    r, g, b = (int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    hh, s, v = colorsys.rgb_to_hsv(r, g, b)
    s = max(0.0, min(1.0, s * factor))
    r2, g2, b2 = colorsys.hsv_to_rgb(hh, s, v)
    return '#{:02x}{:02x}{:02x}'.format(int(r2*255), int(g2*255), int(b2*255))

COLOR_FACES      = _sat(COLOR_FACES)
COLOR_EDGES      = _sat(COLOR_EDGES)
COLOR_VERTICES   = _sat(COLOR_VERTICES)
COLOR_SCENE_FURN = _sat(COLOR_SCENE_FURN)
COLOR_SCENE_FACE = _sat(COLOR_SCENE_FACE)
CAT_COLORS       = [_sat(c) for c in CAT_COLORS_BASE]
ROOM_COLORS      = {k: _sat(v) for k, v in ROOM_COLORS_BASE.items()}

# ══════════════════════════════════════════════════════════════════════════════
#  MATPLOTLIB GLOBAL SETTINGS
# ══════════════════════════════════════════════════════════════════════════════
matplotlib.rcParams.update({
    'font.family':       FONT_FAMILY,
    'font.size':         FONT_SIZE_BASE,
    'axes.titlesize':    FONT_SIZE_TITLE,
    'axes.labelsize':    FONT_SIZE_BASE,
    'legend.fontsize':   FONT_SIZE_LEGEND,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.spines.left':  False,
    'figure.dpi':        FIGURE_DPI,
    'savefig.dpi':       SAVE_DPI,
    'savefig.bbox':      'tight',
})

# ══════════════════════════════════════════════════════════════════════════════
#  SAVE HELPER  — writes both PNG and SVG
# ══════════════════════════════════════════════════════════════════════════════
def _save(fig, stem):
    """Save figure as both PNG and SVG. stem = filename without extension."""
    for ext in ('png', 'svg'):
        path = os.path.join(OUT_DIR, f'{stem}.{ext}')
        fig.savefig(path, format=ext)
    plt.close(fig)
    print(f'  → {stem}.png / .svg')

# ══════════════════════════════════════════════════════════════════════════════
#  SHARED DRAWING PRIMITIVES
# ══════════════════════════════════════════════════════════════════════════════
def _draw_halfviolin(ax, vals, color, log=False):
    """
    Draw a single horizontal half-violin onto an existing axes `ax`.
    Returns the axes for further decoration by the caller.
    """
    vals = np.asarray(vals, dtype=float)
    vals = vals[vals > 0]
    upper     = np.percentile(vals, CLIP_QUANTILE * 100)
    vals_clip = vals[vals <= upper]

    x      = np.log10(vals_clip) if log else vals_clip
    kde    = gaussian_kde(x, bw_method=KDE_BW_METHOD)
    xg     = np.linspace(x.min(), x.max(), 500)
    dens   = kde(xg); dens /= dens.max()
    xplot  = 10 ** xg if log else xg

    ax.fill_between(xplot, 0, dens, color=color, alpha=VIOLIN_ALPHA, linewidth=0)
    ax.plot(xplot, dens, color=color, linewidth=VIOLIN_EDGE_LW, alpha=VIOLIN_EDGE_ALPHA)

    rng    = np.random.default_rng(42)
    jitter = rng.uniform(RUG_JITTER_LOW, RUG_JITTER_HIGH, size=len(vals_clip))
    ax.scatter(vals_clip, jitter, color=color, s=RUG_SIZE,
               alpha=RUG_ALPHA, linewidths=0, zorder=2)

    for pct, ls, lw in [(25, '--', 1.0), (50, '-', 1.8), (75, '--', 1.0)]:
        v = np.percentile(vals_clip, pct)
        ax.axvline(v, ymin=PERC_YMIN, ymax=PERC_YMAX,
                   color=PERC_COLOR, linestyle=ls, linewidth=lw, alpha=PERC_ALPHA)
        ax.text(v, 0.60, f'P{pct}\n{v:,.0f}',
                ha='center', va='bottom', fontsize=FONT_SIZE_ANNOT, color='#222222')

    if log:
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda v, _: f'{int(v):,}' if v >= 1 else f'{v:.1f}'
        ))
    else:
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    ax.set_ylim(-0.15, 1.30)
    ax.set_yticks([])
    ax.grid(axis='x', linestyle=GRID_STYLE, alpha=GRID_ALPHA)

    # Stats box
    n   = len(vals)
    med = float(np.median(vals))
    mn  = float(vals.mean())
    p95 = float(np.percentile(vals, 95))
    ax.text(0.99, 0.97,
            f'N={n:,}  |  Median={med:,.0f}  |  Mean={mn:,.0f}  |  P95={p95:,.0f}',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=FONT_SIZE_STATS, color=STATS_TEXT_COLOR,
            bbox=dict(boxstyle='round,pad=0.3', fc=STATS_BG_COLOR,
                      alpha=STATS_BG_ALPHA, ec='none'))
    return ax


def _halfviolin_figure(vals, xlabel, title, stem, color, log=False):
    """Standalone full figure with a single half-violin."""
    fig, ax = plt.subplots(figsize=(FIG_WIDE, FIG_TALL))
    _draw_halfviolin(ax, vals, color, log=log)
    ax.set_xlabel(xlabel, fontsize=FONT_SIZE_BASE)
    ax.set_title(title, fontweight='bold', pad=10)
    plt.tight_layout()
    _save(fig, stem)


# ══════════════════════════════════════════════════════════════════════════════
#  FURNITURE DATA  ── load once
# ══════════════════════════════════════════════════════════════════════════════
CAT_ORDER  = ['Bed','Cabinet(storage_furniture)','Chair','Desk',
               'Desktop_object','Lamp','Sofa','other_furniture','indoor_scene']
CAT_LABELS = ['Bed','Cabinet','Chair','Desk','Desktop','Lamp','Sofa','Other','Indoor']

df_furn = None

def _load_furn():
    global df_furn
    if df_furn is not None:
        return df_furn
    if not os.path.exists(FURN_CSV):
        print(f'[SKIP] Furniture CSV not found: {FURN_CSV}')
        return None
    df = pd.read_csv(FURN_CSV, low_memory=False)
    df = df[df['step_ok'].astype(str).str.lower() == 'true'].copy()
    df_furn = df
    print(f'  Furniture: {len(df):,} models loaded.')
    return df_furn


# ── A3 whole-dataset half-violins (3 figures) ─────────────────────────────────
def plot_A3_halfviolins():
    df = _load_furn()
    if df is None: return
    for col, xlabel, title, stem, color in [
        ('n_faces',    'Face Count (log)',   'A3 — Faces per Model — Whole Dataset',
         'A3_faces_violin',    COLOR_FACES),
        ('n_edges',    'Edge Count (log)',   'A3 — Edges per Model — Whole Dataset',
         'A3_edges_violin',    COLOR_EDGES),
        ('n_vertices', 'Vertex Count (log)', 'A3 — Vertices per Model — Whole Dataset',
         'A3_vertices_violin', COLOR_VERTICES),
    ]:
        vals = df[col].dropna().astype(float).values
        _halfviolin_figure(vals, xlabel, title, stem, color, log=True)


# ── A3 per-category violins (3 figures) ───────────────────────────────────────
def plot_A3_bycat_violins():
    df = _load_furn()
    if df is None: return

    cats_present = [c for c in CAT_ORDER if c in df['category'].values]
    labels_p     = [CAT_LABELS[CAT_ORDER.index(c)] for c in cats_present]
    colors_p     = [CAT_COLORS[CAT_ORDER.index(c)] for c in cats_present]
    n_cats       = len(cats_present)

    for col, ylabel, title, stem in [
        ('n_faces',    'Face Count (log)',   'A3 — Faces per Model by Category',    'A3_faces_bycat'),
        ('n_edges',    'Edge Count (log)',   'A3 — Edges per Model by Category',    'A3_edges_bycat'),
        ('n_vertices', 'Vertex Count (log)', 'A3 — Vertices per Model by Category', 'A3_vertices_bycat'),
    ]:
        data_by_cat = []
        for cat in cats_present:
            v = df[df['category'] == cat][col].dropna().astype(float)
            v = v[v > 0]
            if len(v) > 1:
                v = v[v <= v.quantile(CLIP_QUANTILE)]
            data_by_cat.append(v.values)

        fig, ax = plt.subplots(figsize=(max(FIG_CAT_W, n_cats * 1.4), FIG_CAT_H))
        positions = np.arange(1, n_cats + 1)

        for i, (vals, color) in enumerate(zip(data_by_cat, colors_p)):
            if len(vals) < 2: continue
            parts = ax.violinplot(vals, positions=[positions[i]],
                                  showmedians=True, showextrema=True, widths=0.7)
            for pc in parts['bodies']:
                pc.set_facecolor(color); pc.set_alpha(VIOLIN_ALPHA)
                pc.set_edgecolor('white'); pc.set_linewidth(0.5)
            for key in ('cmedians','cmins','cmaxes','cbars'):
                if key in parts:
                    parts[key].set_color('black')
                    parts[key].set_linewidth(1.4 if key == 'cmedians' else 0.9)
            med = float(np.median(vals))
            ax.text(positions[i], med * 1.05, f'{med:,.0f}',
                    ha='center', va='bottom', fontsize=FONT_SIZE_ANNOT, color='#333333')

        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f'{int(x):,}' if x >= 1 else f'{x:.2f}'
        ))
        ax.set_xticks(positions)
        ax.set_xticklabels(labels_p, rotation=20, ha='right', fontsize=FONT_SIZE_BASE)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight='bold')
        ax.grid(axis='y', linestyle=GRID_STYLE, alpha=GRID_ALPHA)

        handles = [mpatches.Patch(facecolor=c, alpha=0.75, label=l)
                   for c, l in zip(colors_p, labels_p)]
        ax.legend(handles=handles, fontsize=FONT_SIZE_LEGEND, ncol=3,
                  loc='upper right', framealpha=0.75)
        plt.tight_layout()
        _save(fig, stem)


# ── A1 — Category count bar chart ────────────────────────────────────────────
def plot_A1_category_counts():
    t1_path = os.path.join(BASE, "furniture_stats_tables", "t1_overview.csv")
    if not os.path.exists(t1_path):
        print(f'[SKIP] t1_overview.csv not found'); return

    t1 = pd.read_csv(t1_path)
    # Rows whose Metric starts with "  Category:"
    cat_rows = t1[t1['Metric'].str.strip().str.startswith('Category:')].copy()
    cat_rows['label'] = cat_rows['Metric'].str.replace(r'^\s*Category:\s*', '', regex=True).str.strip()
    cat_rows['count'] = pd.to_numeric(cat_rows['Value'], errors='coerce').fillna(0).astype(int)
    # Map raw names to short labels
    label_map = dict(zip(CAT_ORDER, CAT_LABELS))
    cat_rows['short'] = cat_rows['label'].map(label_map).fillna(cat_rows['label'])
    # Order by CAT_ORDER
    order_map = {c: i for i, c in enumerate(CAT_ORDER)}
    cat_rows['sort_key'] = cat_rows['label'].map(order_map).fillna(99)
    cat_rows = cat_rows.sort_values('sort_key')

    colors = [CAT_COLORS[CAT_ORDER.index(r)] if r in CAT_ORDER else '#CCCCCC'
              for r in cat_rows['label']]

    fig, ax = plt.subplots(figsize=(max(8, len(cat_rows) * 1.1), 5))
    bars = ax.bar(range(len(cat_rows)), cat_rows['count'].values,
                  color=colors, edgecolor='white', linewidth=0.5, width=0.72)

    for bar, v in zip(bars, cat_rows['count'].values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + cat_rows['count'].max() * 0.01,
                f'{v:,}', ha='center', va='bottom', fontsize=FONT_SIZE_ANNOT)

    ax.set_xticks(range(len(cat_rows)))
    ax.set_xticklabels(cat_rows['short'].values, rotation=20, ha='right', fontsize=FONT_SIZE_BASE)
    ax.set_ylabel('Model Count')
    ax.set_title('A1 — Furniture Model Count by Category', fontweight='bold')
    ax.set_ylim(0, cat_rows['count'].max() * 1.14)
    ax.grid(axis='y', linestyle=GRID_STYLE, alpha=GRID_ALPHA)

    total = cat_rows['count'].sum()
    ax.text(0.99, 0.97, f'Total: {total:,} models',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=FONT_SIZE_STATS, color=STATS_TEXT_COLOR,
            bbox=dict(boxstyle='round,pad=0.3', fc=STATS_BG_COLOR, alpha=STATS_BG_ALPHA, ec='none'))

    plt.tight_layout()
    _save(fig, 'A1_category_counts')


# ── A2 — Surface & curve type pie charts ──────────────────────────────────────
def plot_A2_type_pies():
    t3_path = os.path.join(BASE, "furniture_stats_tables", "t3_surface_types.csv")
    t4_path = os.path.join(BASE, "furniture_stats_tables", "t4_curve_types.csv")
    if not os.path.exists(t3_path) or not os.path.exists(t4_path):
        print('[SKIP] t3/t4 CSVs not found'); return

    t3 = pd.read_csv(t3_path)
    t4 = pd.read_csv(t4_path)

    # Keep only types with Total_instances > 0
    t3 = t3[pd.to_numeric(t3['Total_instances'], errors='coerce').fillna(0) > 0].copy()
    t4 = t4[pd.to_numeric(t4['Total_instances'], errors='coerce').fillna(0) > 0].copy()
    t3['Total_instances'] = pd.to_numeric(t3['Total_instances'])
    t4['Total_instances'] = pd.to_numeric(t4['Total_instances'])

    # Short display names
    SURF_LABEL = {
        'PLANE': 'Plane', 'CYLINDRICAL_SURFACE': 'Cylinder',
        'CONICAL_SURFACE': 'Cone', 'SPHERICAL_SURFACE': 'Sphere',
        'TOROIDAL_SURFACE': 'Torus', 'B_SPLINE_SURFACE_WITH_KNOTS': 'BSpline',
        'B_SPLINE_SURFACE': 'BSpline', 'SURFACE_OF_REVOLUTION': 'Revolution',
        'SURFACE_OF_LINEAR_EXTRUSION': 'Extrusion', 'OFFSET_SURFACE': 'Offset',
    }
    CURVE_LABEL = {
        'LINE': 'Line', 'CIRCLE': 'Circle', 'ELLIPSE': 'Ellipse',
        'HYPERBOLA': 'Hyperbola', 'B_SPLINE_CURVE_WITH_KNOTS': 'BSpline',
        'B_SPLINE_CURVE': 'BSpline', 'TRIMMED_CURVE': 'Trimmed',
        'COMPOSITE_CURVE': 'Composite',
    }

    SURF_COLORS_MAP = ['#4C72B0','#DD8452','#55A868','#C44E52','#8172B3',
                       '#937860','#DA8BC3','#8C8C8C','#CCB974']
    CURVE_COLORS_MAP = ['#4C72B0','#DD8452','#55A868','#C44E52','#8172B3',
                        '#937860','#DA8BC3']

    def _pie(ax, df_type, label_map, colors_pool, title):
        df_type = df_type.copy()
        df_type['display'] = df_type.iloc[:, 0].map(label_map).fillna(df_type.iloc[:, 0])
        # Merge duplicated display names (e.g. two BSpline variants)
        agg = df_type.groupby('display')['Total_instances'].sum().reset_index()
        agg = agg.sort_values('Total_instances', ascending=False)
        labels_ = agg['display'].tolist()
        sizes_  = agg['Total_instances'].tolist()
        cols_   = [_sat(colors_pool[i % len(colors_pool)]) for i in range(len(labels_))]

        wedges, _, autotexts = ax.pie(
            sizes_, labels=None, colors=cols_,
            autopct='%1.1f%%', startangle=140, pctdistance=0.78,
            wedgeprops={'linewidth': 0.6, 'edgecolor': 'white'},
        )
        for at in autotexts:
            at.set_fontsize(FONT_SIZE_ANNOT)
        ax.legend(wedges, labels_, loc='lower right',
                  fontsize=FONT_SIZE_LEGEND, framealpha=0.75,
                  bbox_to_anchor=(1.22, 0.0))
        ax.set_title(title, fontweight='bold', pad=12)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    _pie(axes[0], t3, SURF_LABEL, SURF_COLORS_MAP,  'Surface Type Distribution')
    _pie(axes[1], t4, CURVE_LABEL, CURVE_COLORS_MAP, 'Curve Type Distribution')
    fig.suptitle('A2 — Geometric Primitive Distribution', fontsize=FONT_SIZE_TITLE,
                 fontweight='bold', y=1.01)
    plt.tight_layout()
    _save(fig, 'A2_type_pies')


# ══════════════════════════════════════════════════════════════════════════════
#  SCENE DATA  ── load once
# ══════════════════════════════════════════════════════════════════════════════
df_scene = None

def _load_scene():
    global df_scene
    if df_scene is not None:
        return df_scene
    if not os.path.exists(SCENE_CSV):
        print(f'[SKIP] Scene CSV not found: {SCENE_CSV}')
        return None
    df = pd.read_csv(SCENE_CSV, low_memory=False)
    df['step_ok']     = df['step_ok'].astype(str).str.lower() == 'true'
    df['has_layout']  = df['has_layout'].astype(str).str.lower() == 'true'
    df['room_type']   = df['room_type'].fillna('Unknown').astype(str).str.strip()
    df['n_furniture'] = pd.to_numeric(df['n_furniture'], errors='coerce')
    df['n_faces']     = pd.to_numeric(df['n_faces'],     errors='coerce')
    df_scene = df
    print(f'  Scene: {len(df):,} scenes loaded  '
          f'(p1={len(df[df["phase"]=="p1"]):,}  p2={len(df[df["phase"]=="p2"]):,})')
    return df_scene


# ── S1 — Room type counts bar chart ───────────────────────────────────────────
def plot_S1_room_counts():
    df = _load_scene()
    if df is None: return

    room_order  = df.groupby('room_type').size().sort_values(ascending=False).index.tolist()
    counts_raw  = df.groupby('room_type').size().reindex(room_order).dropna().astype(int)

    main  = counts_raw[counts_raw >= S1_MERGE_THRESHOLD]
    small = counts_raw[counts_raw <  S1_MERGE_THRESHOLD]
    if len(small) > 0:
        other_total = main.get('OtherRoom', 0) + small.sum()
        main = main.drop('OtherRoom', errors='ignore')
        main['OtherRoom'] = other_total
    counts = main.sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(9, max(4, len(counts) * 0.52)))
    colors  = [ROOM_COLORS.get(rt, '#CCCCCC') for rt in counts.index]
    bars    = ax.barh(counts.index[::-1], counts.values[::-1],
                      color=colors[::-1], edgecolor='white', linewidth=0.4, height=0.72)

    for bar, v in zip(bars, counts.values[::-1]):
        ax.text(bar.get_width() + counts.max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{v:,}', va='center', fontsize=FONT_SIZE_ANNOT)

    ax.set_xlabel('Scene Count')
    ax.set_title('S1 — Scene Count by Room Type (P1 + P2 merged)', fontweight='bold')
    ax.set_xlim(0, counts.max() * 1.15)
    ax.grid(axis='x', linestyle=GRID_STYLE, alpha=GRID_ALPHA)
    ax.tick_params(axis='y', labelsize=FONT_SIZE_BASE)

    if len(small) > 0:
        note = 'OtherRoom includes: ' + ', '.join(f'{rt}({n})' for rt, n in small.items())
        ax.text(0.01, -0.06, note, transform=ax.transAxes,
                ha='left', va='top', fontsize=7.5, color='#777777')

    ax.text(0.99, 0.02, f'Total: {len(df):,} scenes',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=FONT_SIZE_STATS, color='#555555')

    plt.tight_layout()
    _save(fig, 'S1_room_type_counts')


# ── S2 — Furniture count half-violin ──────────────────────────────────────────
def plot_S2_furniture_violin():
    df = _load_scene()
    if df is None: return
    sub  = df[df['has_layout'] & df['n_furniture'].notna() & (df['n_furniture'] > 0)]
    vals = sub['n_furniture'].values.astype(float)
    _halfviolin_figure(
        vals,
        xlabel='Furniture Count per Scene',
        title='S2 — Furniture Count per Scene (P1, all room types)',
        stem='S2_furniture_count',
        color=COLOR_SCENE_FURN,
        log=False,
    )


# ── S3 — Scene face count half-violin ─────────────────────────────────────────
def plot_S3_scene_faces():
    df = _load_scene()
    if df is None: return
    sub  = df[df['step_ok'] & df['n_faces'].notna() & (df['n_faces'] > 0)]
    vals = sub['n_faces'].values.astype(float)
    _halfviolin_figure(
        vals,
        xlabel='Face Count per Scene (log scale)',
        title='S3 — Scene Face Count (P1 + P2, all room types)',
        stem='S3_scene_faces',
        color=COLOR_SCENE_FACE,
        log=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    args         = sys.argv[1:]
    do_furniture = '--scenes'    not in args
    do_scenes    = '--furniture' not in args

    if do_furniture:
        print('\n── Furniture figures ─────────────────────────────────────────────────')
        plot_A1_category_counts()
        plot_A2_type_pies()
        plot_A3_halfviolins()
        plot_A3_bycat_violins()

    if do_scenes:
        print('\n── Scene figures ─────────────────────────────────────────────────────')
        plot_S1_room_counts()
        plot_S2_furniture_violin()
        plot_S3_scene_faces()

    print(f'\nAll figures → {OUT_DIR}/')


if __name__ == '__main__':
    main()
