"""
Indoor Scene Statistics Collector
Targets:
  - Scene_sy/p1_scene_output/   (assembled STEP scenes, layout JSONs in layout_outputs/)
  - Scene_sy/p2_scenes_output/  (variant STEP scenes)

For each STEP file:  topology counts (faces/edges/vertices/shells/surfaces)
From layout JSONs:   room type, room dimensions, furniture count, category mix, scale stats

Output → scene_stats.json  +  scene_stats_tables/

Dependencies: numpy
Usage: python scene_stats_collect.py
"""

import os, re, json, struct, time, csv
from pathlib import Path
import numpy as np

# ── Configuration ──────────────────────────────────────────────────────────────
SCENE_SY         = r"d:\Lzm_Temp_Data\Li_temp_project\Scene_sy"
P1_DIR           = r"d:\Lzm_Temp_Data\Li_temp_project\Scene_sy\p1_scene_output"
P2_DIR           = r"d:\Lzm_Temp_Data\Li_temp_project\Scene_sy\p2_scenes_output"
LAYOUT_DIR       = r"d:\Lzm_Temp_Data\Li_temp_project\Scene_sy\layout_outputs"
OUTPUT_JSON      = r"d:\Lzm_Temp_Data\Li_temp_project\scene_stats.json"
CHECKPOINT_FILE  = r"d:\Lzm_Temp_Data\Li_temp_project\scene_stats_ckpt.json"
TABLE_DIR        = r"d:\Lzm_Temp_Data\Li_temp_project\scene_stats_tables"

# ── STEP entity types ──────────────────────────────────────────────────────────
SURFACE_TYPES = [
    'PLANE','CYLINDRICAL_SURFACE','CONICAL_SURFACE','SPHERICAL_SURFACE',
    'TOROIDAL_SURFACE','B_SPLINE_SURFACE_WITH_KNOTS','B_SPLINE_SURFACE',
    'SURFACE_OF_REVOLUTION','SURFACE_OF_LINEAR_EXTRUSION','OFFSET_SURFACE',
]
CURVE_TYPES = [
    'LINE','CIRCLE','ELLIPSE','HYPERBOLA',
    'B_SPLINE_CURVE_WITH_KNOTS','B_SPLINE_CURVE','TRIMMED_CURVE','COMPOSITE_CURVE',
]
TOPO_TYPES = [
    'ADVANCED_FACE','EDGE_CURVE','VERTEX_POINT',
    'CLOSED_SHELL','OPEN_SHELL','MANIFOLD_SOLID_BREP',
    'FACE_BOUND','FACE_OUTER_BOUND',
]
TRACKED   = set(SURFACE_TYPES + CURVE_TYPES + TOPO_TYPES)
ENTITY_RE = re.compile(r'=\s*([A-Z][A-Z0-9_]*)\s*\(')

_STL_DTYPE = np.dtype([
    ('normal','<f4',(3,)),('v0','<f4',(3,)),
    ('v1','<f4',(3,)),('v2','<f4',(3,)),('attr','<u2'),
])


# ── Layout JSON index ──────────────────────────────────────────────────────────
def _build_layout_index(layout_dir: str) -> dict:
    """Map scene_id → layout dict.  scene_id = stem of layout file without 'layout_' prefix."""
    idx = {}
    ldir = Path(layout_dir)
    if not ldir.is_dir():
        return idx
    for jf in ldir.glob('*.json'):
        try:
            with open(jf, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # key = filename stem  e.g. "layout_a002b6aa_..._Bedroom-8563"
            idx[jf.stem] = data
        except Exception:
            pass
    return idx


def _extract_layout_meta(layout: dict) -> dict:
    """Pull scene-level stats from a parsed layout JSON."""
    meta = {
        'room_type':       layout.get('room_type', None),
        'room_length':     None, 'room_width': None, 'room_height': None,
        'room_area_m2':    None, 'room_volume_m3': None,
        'n_furniture':     0,
        'category_set':    [],
        'n_categories':    0,
        'n_unique_models': 0,
        'has_bed':         False, 'has_chair': False, 'has_sofa': False,
        'has_desk': False, 'has_cabinet': False, 'has_lamp': False,
        'avg_furniture_max_dim': None,
        'min_furniture_max_dim': None,
        'max_furniture_max_dim': None,
        'avg_rotation_z':  None,
    }
    room = layout.get('room', {})
    if room:
        L = room.get('length', 0); W = room.get('width', 0); H = room.get('height', 0)
        meta['room_length']    = round(L/1000, 3) if L else None
        meta['room_width']     = round(W/1000, 3) if W else None
        meta['room_height']    = round(H/1000, 3) if H else None
        meta['room_area_m2']   = round(L*W/1e6, 3) if L and W else None
        meta['room_volume_m3'] = round(L*W*H/1e9, 3) if L and W and H else None

    furniture = layout.get('furniture', [])
    meta['n_furniture'] = len(furniture)
    if furniture:
        cats = []
        folders = set()
        dims = []
        rots = []
        for f in furniture:
            cat = f.get('matched_category') or f.get('source_category', '')
            if cat: cats.append(cat)
            folder = f.get('folder', '')
            if folder: folders.add(folder)
            max_dim = f.get('target_max_size') or f.get('matched_max_dim')
            if max_dim: dims.append(max_dim / 1000)  # mm→m
            rot = f.get('rotation_z')
            if rot is not None: rots.append(rot)
        meta['category_set']    = sorted(set(cats))
        meta['n_categories']    = len(set(cats))
        meta['n_unique_models'] = len(folders)
        meta['has_bed']    = any('Bed'     in c for c in cats)
        meta['has_chair']  = any('Chair'   in c for c in cats)
        meta['has_sofa']   = any('Sofa'    in c for c in cats)
        meta['has_desk']   = any('Desk'    in c for c in cats)
        meta['has_cabinet']= any('Cabinet' in c for c in cats)
        meta['has_lamp']   = any('Lamp'    in c for c in cats)
        if dims:
            meta['avg_furniture_max_dim'] = round(sum(dims)/len(dims), 3)
            meta['min_furniture_max_dim'] = round(min(dims), 3)
            meta['max_furniture_max_dim'] = round(max(dims), 3)
        if rots:
            meta['avg_rotation_z'] = round(sum(rots)/len(rots), 2)
    return meta


# ── Scene Discovery ────────────────────────────────────────────────────────────
def _parse_p1_name(stem: str):
    """layout_<uuid>_<RoomType>-<num>  →  (uuid, room_type, layout_key)"""
    m = re.match(r'layout_([0-9a-f-]+)_([A-Za-z]+)-\d+', stem)
    if m:
        return m.group(1), m.group(2), stem
    return None, stem, stem


def _parse_p2_name(stem: str):
    """variant_<idx>_<RoomType>_v<n>  →  (variant_idx, room_type, variant_n)"""
    m = re.match(r'variant_(\d+)_([A-Za-z]+)_v(\d+)', stem)
    if m:
        return int(m.group(1)), m.group(2), int(m.group(3))
    return None, stem, None


def find_scenes(p1_dir, p2_dir, layout_idx):
    scenes = []
    # P1 scenes
    for f in sorted(Path(p1_dir).glob('*.step')):
        uuid, room_type, layout_key = _parse_p1_name(f.stem)
        layout = layout_idx.get(layout_key, {})
        layout_meta = _extract_layout_meta(layout) if layout else {}
        scenes.append({
            'scene_id':   f.stem,
            'phase':      'p1',
            'room_type':  layout_meta.get('room_type') or room_type,
            'step_path':  str(f),
            'step_kb':    round(f.stat().st_size / 1024, 2),
            'has_layout': bool(layout),
            **{k: layout_meta.get(k) for k in layout_meta if k != 'room_type'},
        })
    # P2 scenes
    for f in sorted(Path(p2_dir).glob('*.step')):
        vidx, room_type, vn = _parse_p2_name(f.stem)
        scenes.append({
            'scene_id':   f.stem,
            'phase':      'p2',
            'room_type':  room_type,
            'variant_idx': vidx,
            'variant_n':   vn,
            'step_path':  str(f),
            'step_kb':    round(f.stat().st_size / 1024, 2),
            'has_layout': False,
            'n_furniture': None,
        })
    return scenes


# ── STEP Parsing (scene files tend to be large — count is enough) ─────────────
def parse_step(path: str, size_limit_mb=2000) -> dict:
    counts = {e: 0 for e in TRACKED}
    counts['step_ok'] = True
    try:
        sz_mb = os.path.getsize(path) / 1024 / 1024
        if sz_mb > size_limit_mb:
            counts['step_ok'] = False
            return counts
        for enc in ('utf-8', 'latin-1', 'cp1252'):
            try:
                with open(path, 'r', encoding=enc, errors='replace') as f:
                    for line in f:
                        m = ENTITY_RE.search(line)
                        if m and m.group(1) in TRACKED:
                            counts[m.group(1)] += 1
                break
            except Exception:
                continue
    except Exception:
        counts['step_ok'] = False
    return counts


def derive_scene_topo(s: dict) -> dict:
    F = s.get('ADVANCED_FACE', 0)
    E = s.get('EDGE_CURVE', 0)
    V = s.get('VERTEX_POINT', 0)
    solids = s.get('MANIFOLD_SOLID_BREP', 0)
    shells = s.get('CLOSED_SHELL', 0) + s.get('OPEN_SHELL', 0)
    total_surf = sum(s.get(t, 0) for t in SURFACE_TYPES)
    bspline = s.get('B_SPLINE_SURFACE_WITH_KNOTS', 0) + s.get('B_SPLINE_SURFACE', 0)
    surf_div = sum(1 for t in SURFACE_TYPES if s.get(t, 0) > 0)
    return {
        'n_faces':    F,
        'n_edges':    E,
        'n_vertices': V,
        'n_shells':   shells,
        'n_solids':   solids,
        'surf_diversity': surf_div,
        'bspline_ratio': round(bspline / total_surf, 4) if total_surf > 0 else None,
        'edges_per_face': round(E / F, 4) if F > 0 else None,
        'faces_per_solid': round(F / solids, 2) if solids > 0 else None,
    }


# ── Summary Tables ─────────────────────────────────────────────────────────────
def _pct(n, d): return f"{100*n/d:.1f}%" if d > 0 else "N/A"

def _qs(vals):
    a = sorted(v for v in vals if v is not None)
    if not a: return dict(n=0, min=None, p25=None, med=None, p75=None, max=None, mean=None)
    n = len(a)
    def p(pct):
        idx=(n-1)*pct/100; lo=int(idx); f=idx-lo
        return round(a[lo]+(a[lo+1]-a[lo])*f if lo+1<n else a[lo], 2)
    return dict(n=n, min=round(a[0],2), p25=p(25), med=p(50), p75=p(75),
                max=round(a[-1],2), mean=round(sum(a)/n,2))

def write_csv(path, rows, fields):
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"  → {path}")


KNOWN_ROOM_TYPES = [
    'LivingDiningRoom','LivingRoom','MasterBedroom','Bedroom','SecondBedroom',
    'KidsRoom','Library','DiningRoom','Corridor','CloakRoom','OtherRoom',
    'ElderlyRoom','Hallway','Lounge','StorageRoom',
]

def generate_tables(results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    total = len(results)
    p1 = [r for r in results if r.get('phase') == 'p1']
    p2 = [r for r in results if r.get('phase') == 'p2']
    step_ok = [r for r in results if r.get('step_ok', False)]

    # Table 1: Overview
    rows = [
        {'Metric':'Total scenes',       'Value':total,    'Note':''},
        {'Metric':'Phase-1 scenes',     'Value':len(p1),  'Note':'layout_*.step'},
        {'Metric':'Phase-2 scenes',     'Value':len(p2),  'Note':'variant_*.step'},
        {'Metric':'STEP parsed OK',     'Value':len(step_ok), 'Note':_pct(len(step_ok),total)},
        {'Metric':'With layout JSON',   'Value':sum(1 for r in results if r.get('has_layout')),
         'Note':_pct(sum(1 for r in results if r.get('has_layout')),total)},
    ]
    # Unique room types
    room_counts = {}
    for r in results:
        rt = r.get('room_type') or 'Unknown'
        room_counts[rt] = room_counts.get(rt, 0) + 1
    rows.append({'Metric':'Unique room types', 'Value':len(room_counts), 'Note':''})
    for rt, cnt in sorted(room_counts.items(), key=lambda x:-x[1]):
        rows.append({'Metric':f'  Room: {rt}', 'Value':cnt, 'Note':_pct(cnt,total)})
    write_csv(f"{out_dir}/t1_scene_overview.csv", rows, ['Metric','Value','Note'])

    # Table 2: Topology stats overall + by phase
    topo_metrics = [
        ('n_faces','Faces'), ('n_edges','Edges'), ('n_vertices','Vertices'),
        ('n_shells','Shells'), ('n_solids','Solids (furniture pieces)'),
        ('faces_per_solid','Faces per solid'), ('surf_diversity','Surface type diversity'),
        ('bspline_ratio','BSpline surface ratio'), ('step_kb','STEP file size (KB)'),
    ]
    topo_rows = []
    for field, label in topo_metrics:
        s_all = _qs([r.get(field) for r in step_ok])
        s_p1  = _qs([r.get(field) for r in step_ok if r.get('phase')=='p1'])
        s_p2  = _qs([r.get(field) for r in step_ok if r.get('phase')=='p2'])
        topo_rows.append({'Metric':label,
                          'All_n':s_all['n'], 'All_med':s_all['med'], 'All_max':s_all['max'],
                          'P1_med':s_p1['med'], 'P1_max':s_p1['max'],
                          'P2_med':s_p2['med'], 'P2_max':s_p2['max']})
    write_csv(f"{out_dir}/t2_scene_topology.csv", topo_rows,
              ['Metric','All_n','All_med','All_max','P1_med','P1_max','P2_med','P2_max'])

    # Table 3: Room-level stats (from layout JSON, P1 only)
    p1_layout = [r for r in p1 if r.get('has_layout')]
    room_rows = []
    for rt in KNOWN_ROOM_TYPES + ['Unknown']:
        sub = [r for r in p1_layout if (r.get('room_type') or 'Unknown') == rt]
        if not sub: continue
        s_area = _qs([r.get('room_area_m2') for r in sub])
        s_furn = _qs([r.get('n_furniture') for r in sub])
        s_cats = _qs([r.get('n_categories') for r in sub])
        s_dim  = _qs([r.get('avg_furniture_max_dim') for r in sub])
        room_rows.append({
            'Room_type': rt, 'N': len(sub),
            'RoomArea_m2_med': s_area['med'], 'RoomArea_m2_max': s_area['max'],
            'Furniture_count_med': s_furn['med'], 'Furniture_count_max': s_furn['max'],
            'Category_diversity_med': s_cats['med'],
            'AvgFurnDim_m_med': s_dim['med'],
        })
    write_csv(f"{out_dir}/t3_room_layout_stats.csv", room_rows,
              ['Room_type','N','RoomArea_m2_med','RoomArea_m2_max',
               'Furniture_count_med','Furniture_count_max',
               'Category_diversity_med','AvgFurnDim_m_med'])

    # Table 4: Furniture category co-occurrence in scenes
    cat_cols = ['has_bed','has_chair','has_sofa','has_desk','has_cabinet','has_lamp']
    cat_names = ['Bed','Chair','Sofa','Desk','Cabinet','Lamp']
    cooc_rows = []
    for i, (col, name) in enumerate(zip(cat_cols, cat_names)):
        cnt = sum(1 for r in p1_layout if r.get(col))
        cooc_rows.append({'Category': name,
                          'Scenes_containing': cnt,
                          'Prevalence_%': _pct(cnt, len(p1_layout))})
    write_csv(f"{out_dir}/t4_furniture_cooccurrence.csv", cooc_rows,
              ['Category','Scenes_containing','Prevalence_%'])

    # Table 5: Surface type usage in scenes
    surf_rows = []
    for st in SURFACE_TYPES:
        with_type = [r for r in step_ok if r.get(st, 0) > 0]
        s = _qs([r.get(st) for r in with_type])
        surf_rows.append({
            'Surface_type': st,
            'Scenes_with':  len(with_type),
            'Prevalence_%': _pct(len(with_type), len(step_ok)),
            'Total_instances': sum(r.get(st,0) for r in step_ok),
            'Median_per_scene': s['med'],
            'Max': s['max'],
        })
    write_csv(f"{out_dir}/t5_surface_types_in_scenes.csv", surf_rows,
              ['Surface_type','Scenes_with','Prevalence_%','Total_instances',
               'Median_per_scene','Max'])

    # Table 6: P2 variant analysis
    if p2:
        # group by (variant_idx, room_type) — each group is 1 layout × N variants
        from collections import defaultdict
        groups = defaultdict(list)
        for r in [x for x in p2 if x.get('step_ok')]:
            key = (r.get('variant_idx'), r.get('room_type','?'))
            groups[key].append(r)
        var_rows = []
        for (vidx, rt), grp in sorted(groups.items()):
            faces = [r.get('n_faces',0) for r in grp]
            solids= [r.get('n_solids',0) for r in grp]
            var_rows.append({
                'Variant_idx': vidx, 'Room_type': rt,
                'N_versions': len(grp),
                'Faces_mean': round(sum(faces)/len(faces),1) if faces else None,
                'Faces_std':  round(float(np.std(faces)),1) if len(faces)>1 else 0,
                'Solids_mean':round(sum(solids)/len(solids),1) if solids else None,
            })
        write_csv(f"{out_dir}/t6_p2_variant_analysis.csv", var_rows,
                  ['Variant_idx','Room_type','N_versions','Faces_mean','Faces_std','Solids_mean'])

    # Table 7: Full raw dump
    all_keys = ['scene_id','phase','room_type','step_kb','has_layout','step_ok',
                'n_furniture','n_categories','n_unique_models',
                'room_length','room_width','room_height','room_area_m2','room_volume_m3',
                'avg_furniture_max_dim','min_furniture_max_dim','max_furniture_max_dim',
                'has_bed','has_chair','has_sofa','has_desk','has_cabinet','has_lamp',
                'n_faces','n_edges','n_vertices','n_shells','n_solids',
                'edges_per_face','faces_per_solid','surf_diversity','bspline_ratio',
                'variant_idx','variant_n'] + list(TRACKED)
    raw_rows = [{k: r.get(k,'') for k in all_keys} for r in results]
    write_csv(f"{out_dir}/t7_raw.csv", raw_rows, all_keys)

    print(f"\nTables saved → {out_dir}/")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print(f"Building layout index from: {LAYOUT_DIR}")
    layout_idx = _build_layout_index(LAYOUT_DIR)
    print(f"  {len(layout_idx)} layout JSONs loaded.")

    print(f"Discovering scenes in p1 and p2...")
    scenes = find_scenes(P1_DIR, P2_DIR, layout_idx)
    total  = len(scenes)
    print(f"  Found {total} scenes ({sum(1 for s in scenes if s['phase']=='p1')} p1, "
          f"{sum(1 for s in scenes if s['phase']=='p2')} p2).\n")

    results, done_ids = [], set()
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            results = json.load(f)
        done_ids = {r['scene_id'] for r in results}
        print(f"Resuming — {len(done_ids)} already done.\n")

    t0 = time.time()
    for idx, sc in enumerate(scenes):
        sid = sc['scene_id']
        if sid in done_ids:
            continue
        elapsed = time.time() - t0
        rate    = max((idx+1 - len(done_ids)) / max(elapsed,1), 1e-6)
        remain  = (total - idx - 1) / rate
        print(f"[{idx+1:>5}/{total}] {sid[:70]:<70}  ~{remain/60:.1f}min",
              end='\r', flush=True)

        rec = {**sc}
        # Convert list to string for JSON
        if isinstance(rec.get('category_set'), list):
            rec['category_set'] = ','.join(rec['category_set'])

        step = parse_step(sc['step_path'])
        rec.update(step)
        rec.update(derive_scene_topo(step))
        results.append(rec)

        if len(results) % 50 == 0:
            with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False)

    print(f"\nDone in {(time.time()-t0)/60:.1f} min.")
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"JSON → {OUTPUT_JSON}")
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
    print("\nGenerating tables...")
    generate_tables(results, TABLE_DIR)


if __name__ == '__main__':
    main()
