"""
Furniture Dataset Statistics Collector
Targets: Dataset_Categoried/1_1000  +  Dataset_Categoried/indoor_scene
Parses *_SHELL.step (topology/geometry types) and *_MESH.STL (physical metrics).
Saves incremental checkpoint; final output → furniture_stats.json + furniture_stats_tables/

Dependencies: numpy
Usage: python furniture_stats_collect.py
"""

import os, re, json, struct, time, csv
from pathlib import Path
import numpy as np

# ── Configuration ──────────────────────────────────────────────────────────────
DATASET_ROOT    = r"d:\Lzm_Temp_Data\Li_temp_project\Dataset_Categoried"
TARGET_BATCH    = "1_1000"          # change to None to scan all batches
OUTPUT_JSON     = r"d:\Lzm_Temp_Data\Li_temp_project\furniture_stats.json"
CHECKPOINT_FILE = r"d:\Lzm_Temp_Data\Li_temp_project\furniture_stats_ckpt.json"
TABLE_DIR       = r"d:\Lzm_Temp_Data\Li_temp_project\furniture_stats_tables"

CATEGORIES = [
    'Bed', 'Cabinet(storage_furniture)', 'Chair', 'Desk',
    'Desktop_object', 'Lamp', 'Sofa', 'other_furniture',
]

ALL_BATCHES = [
    '1_1000','1001_2000','2001_3000','3001_4000','4001_5000',
    '5001_6000','6001_7000','7001_8000','8001_9000','9001_10000',
]

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
TRACKED = set(SURFACE_TYPES + CURVE_TYPES + TOPO_TYPES)
ENTITY_RE = re.compile(r'=\s*([A-Z][A-Z0-9_]*)\s*\(')

_STL_DTYPE = np.dtype([
    ('normal','<f4',(3,)),('v0','<f4',(3,)),
    ('v1','<f4',(3,)),('v2','<f4',(3,)),('attr','<u2'),
])


# ── Discovery ─────────────────────────────────────────────────────────────────
def find_models(root: str, target_batch=None):
    root = Path(root)
    batches = [target_batch] if target_batch else ALL_BATCHES
    for batch in batches:
        batch_dir = root / batch
        if not batch_dir.is_dir():
            continue
        for cat in CATEGORIES:
            cat_dir = batch_dir / cat
            if not cat_dir.is_dir():
                continue
            for model_dir in sorted(cat_dir.iterdir()):
                if not model_dir.is_dir():
                    continue
                step_files = list(model_dir.glob('*_SHELL.step'))
                stl_files  = list(model_dir.glob('*_MESH.STL'))
                desc_path  = model_dir / 'description.txt'
                align_path = model_dir / 'alignment_meta.json'
                yield {
                    'model_id':    f"{batch}/{cat}/{model_dir.name}",
                    'batch':       batch,
                    'category':    cat,
                    'model_name':  model_dir.name,
                    'step_path':   str(step_files[0]) if step_files else None,
                    'stl_path':    str(stl_files[0])  if stl_files  else None,
                    'has_description': desc_path.exists(),
                    'has_alignment':   align_path.exists(),
                    'n_render_views':  _count_renders(model_dir),
                }
    # indoor_scene (no category subfolder)
    indoor = root / 'indoor_scene'
    if indoor.is_dir():
        for model_dir in sorted(indoor.iterdir()):
            if not model_dir.is_dir():
                continue
            step_files = list(model_dir.glob('*_SHELL.step'))
            stl_files  = list(model_dir.glob('*_MESH.STL'))
            desc_path  = model_dir / 'description.txt'
            align_path = model_dir / 'alignment_meta.json'
            yield {
                'model_id':    f"indoor_scene/{model_dir.name}",
                'batch':       'indoor_scene',
                'category':    'indoor_scene',
                'model_name':  model_dir.name,
                'step_path':   str(step_files[0]) if step_files else None,
                'stl_path':    str(stl_files[0])  if stl_files  else None,
                'has_description': desc_path.exists(),
                'has_alignment':   align_path.exists(),
                'n_render_views':  _count_renders(model_dir),
            }


def _count_renders(model_dir: Path) -> int:
    mv = model_dir / 'multiview_picture'
    if not mv.is_dir():
        return 0
    return len(list(mv.glob('*.png')))


# ── STEP Parsing ───────────────────────────────────────────────────────────────
def parse_step(path: str) -> dict:
    counts = {e: 0 for e in TRACKED}
    counts['step_ok'] = True
    counts['step_kb'] = 0
    try:
        counts['step_kb'] = round(os.path.getsize(path) / 1024, 2)
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


def derive_topo(s: dict) -> dict:
    V = s.get('VERTEX_POINT', 0)
    E = s.get('EDGE_CURVE', 0)
    F = s.get('ADVANCED_FACE', 0)
    S = s.get('CLOSED_SHELL', 0) + s.get('OPEN_SHELL', 0)
    solids = s.get('MANIFOLD_SOLID_BREP', 0)
    inner  = s.get('FACE_BOUND', 0)
    genus  = round(S - (V - E + F) / 2.0, 4) if S > 0 and (V + E + F) > 0 else None
    surf_div = sum(1 for t in SURFACE_TYPES if s.get(t, 0) > 0)
    total_surf = sum(s.get(t, 0) for t in SURFACE_TYPES)
    bspline = s.get('B_SPLINE_SURFACE_WITH_KNOTS', 0) + s.get('B_SPLINE_SURFACE', 0)
    return {
        'n_faces':       F,
        'n_edges':       E,
        'n_vertices':    V,
        'n_shells':      S,
        'n_solids':      solids,
        'n_inner_bounds': inner,
        'genus_approx':  genus,
        'edges_per_face': round(E / F, 4) if F > 0 else None,
        'is_multi_body': solids > 1,
        'has_holes':     inner > 0,
        'surf_diversity': surf_div,
        'bspline_ratio': round(bspline / total_surf, 4) if total_surf > 0 else None,
    }


# ── STL Parsing ────────────────────────────────────────────────────────────────
def _is_binary(path):
    sz = os.path.getsize(path)
    with open(path, 'rb') as f:
        f.read(80); raw = f.read(4)
    if len(raw) < 4: return False
    n = struct.unpack('<I', raw)[0]
    return abs(sz - (84 + n * 50)) <= 4


def _read_bin(path):
    with open(path, 'rb') as f:
        f.read(80); n = struct.unpack('<I', f.read(4))[0]; raw = f.read(n * 50)
    if len(raw) < n * 50: return None
    t = np.frombuffer(raw, dtype=_STL_DTYPE)
    return t['v0'].copy(), t['v1'].copy(), t['v2'].copy()


def _read_ascii(path):
    v0, v1, v2 = [], [], []
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        i = 0
        while i < len(lines):
            if lines[i].strip().startswith('facet normal'):
                i += 2; vs = []
                for _ in range(3):
                    p = lines[i].strip().split()
                    if len(p) == 4: vs.append([float(p[1]), float(p[2]), float(p[3])])
                    i += 1
                if len(vs) == 3: v0.append(vs[0]); v1.append(vs[1]); v2.append(vs[2])
            else: i += 1
    except Exception: return None
    if not v0: return None
    return np.array(v0, np.float32), np.array(v1, np.float32), np.array(v2, np.float32)


def parse_stl(path: str) -> dict:
    r = {'stl_ok': True, 'stl_kb': 0,
         'n_triangles': None, 'surface_area': None, 'volume': None,
         'bbox_x': None, 'bbox_y': None, 'bbox_z': None, 'bbox_diagonal': None,
         'compactness': None}
    try:
        r['stl_kb'] = round(os.path.getsize(path) / 1024, 2)
        if r['stl_kb'] > 500 * 1024:
            r['stl_ok'] = False; return r
        data = _read_bin(path) if _is_binary(path) else _read_ascii(path)
        if data is None:
            r['stl_ok'] = False; return r
        a, b, c = data
        r['n_triangles'] = len(a)
        cross = np.cross(b - a, c - a)
        r['surface_area'] = float(np.sum(0.5 * np.linalg.norm(cross, axis=1)))
        r['volume'] = abs(float(np.sum(a * np.cross(b, c))) / 6.0)
        all_v = np.vstack([a, b, c])
        lo, hi = all_v.min(0), all_v.max(0)
        bbox = hi - lo
        r['bbox_x'] = float(bbox[0])
        r['bbox_y'] = float(bbox[1])
        r['bbox_z'] = float(bbox[2])
        r['bbox_diagonal'] = float(np.linalg.norm(bbox))
        # Compactness = 36π V² / A³  (=1 for perfect sphere)
        if r['surface_area'] > 0 and r['volume'] > 0:
            r['compactness'] = round(
                36 * 3.14159265 * r['volume']**2 / r['surface_area']**3, 6)
    except Exception:
        r['stl_ok'] = False
    return r


# ── Summary Tables ─────────────────────────────────────────────────────────────
def _pct(n, d): return f"{100*n/d:.1f}%" if d > 0 else "N/A"

def _qs(vals):
    a = sorted(v for v in vals if v is not None)
    if not a: return dict(n=0, min=None, p25=None, med=None, p75=None, max=None, mean=None)
    n = len(a)
    def p(pct):
        idx = (n-1)*pct/100; lo=int(idx); f=idx-lo
        return round(a[lo]+(a[lo+1]-a[lo])*f if lo+1<n else a[lo], 3)
    return dict(n=n, min=round(a[0],3), p25=p(25), med=p(50), p75=p(75),
                max=round(a[-1],3), mean=round(sum(a)/n, 3))

def write_csv(path, rows, fields):
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"  → {path}")

CAT_ALL = CATEGORIES + ['indoor_scene']

def generate_tables(results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    total = len(results)
    step_ok = [r for r in results if r.get('step_ok', False)]
    stl_ok  = [r for r in results if r.get('stl_ok',  False)]

    # Table 1: Overview
    rows = [
        {'Metric':'Total models',          'Value':total,          'Pct':''},
        {'Metric':'STEP parsed OK',        'Value':len(step_ok),   'Pct':_pct(len(step_ok),total)},
        {'Metric':'STL parsed OK',         'Value':len(stl_ok),    'Pct':_pct(len(stl_ok),total)},
        {'Metric':'Has description.txt',   'Value':sum(1 for r in results if r.get('has_description')),
         'Pct':_pct(sum(1 for r in results if r.get('has_description')),total)},
        {'Metric':'Has alignment_meta',    'Value':sum(1 for r in results if r.get('has_alignment')),
         'Pct':_pct(sum(1 for r in results if r.get('has_alignment')),total)},
        {'Metric':'Multi-body models',     'Value':sum(1 for r in step_ok if r.get('is_multi_body')),
         'Pct':_pct(sum(1 for r in step_ok if r.get('is_multi_body')),len(step_ok))},
        {'Metric':'Models with holes',     'Value':sum(1 for r in step_ok if r.get('has_holes')),
         'Pct':_pct(sum(1 for r in step_ok if r.get('has_holes')),len(step_ok))},
    ]
    for cat in CAT_ALL:
        cnt = sum(1 for r in results if r.get('category') == cat)
        rows.append({'Metric': f'  Category: {cat}', 'Value': cnt, 'Pct': _pct(cnt, total)})
    write_csv(f"{out_dir}/t1_overview.csv", rows, ['Metric','Value','Pct'])

    # Table 2: Per-category topology summary
    topo_rows = []
    for cat in CAT_ALL:
        sub = [r for r in step_ok if r.get('category') == cat]
        if not sub: continue
        s_f = _qs([r.get('n_faces') for r in sub])
        s_e = _qs([r.get('n_edges') for r in sub])
        s_v = _qs([r.get('n_vertices') for r in sub])
        s_s = _qs([r.get('n_solids') for r in sub])
        s_b = _qs([r.get('bspline_ratio') for r in sub])
        topo_rows.append({
            'Category': cat, 'N_step': len(sub),
            'Faces_med': s_f['med'], 'Faces_max': s_f['max'],
            'Edges_med': s_e['med'], 'Verts_med': s_v['med'],
            'Solids_med': s_s['med'],
            'MultiBody_%': round(100*sum(1 for r in sub if r.get('is_multi_body'))/len(sub),1),
            'Holes_%': round(100*sum(1 for r in sub if r.get('has_holes'))/len(sub),1),
            'BSpline_ratio_med': s_b['med'],
        })
    write_csv(f"{out_dir}/t2_topology_by_category.csv", topo_rows,
              ['Category','N_step','Faces_med','Faces_max','Edges_med','Verts_med',
               'Solids_med','MultiBody_%','Holes_%','BSpline_ratio_med'])

    # Table 3: Surface type prevalence
    surf_rows = []
    for st in SURFACE_TYPES:
        with_type = [r for r in step_ok if r.get(st, 0) > 0]
        total_inst = sum(r.get(st, 0) for r in step_ok)
        s = _qs([r.get(st) for r in with_type])
        surf_rows.append({
            'Surface_type': st,
            'Models_with': len(with_type),
            'Prevalence_%': _pct(len(with_type), len(step_ok)),
            'Total_instances': total_inst,
            'Median_per_model': s['med'],
            'Max': s['max'],
        })
    write_csv(f"{out_dir}/t3_surface_types.csv", surf_rows,
              ['Surface_type','Models_with','Prevalence_%','Total_instances','Median_per_model','Max'])

    # Table 4: Curve type prevalence
    curve_rows = []
    for ct in CURVE_TYPES:
        with_type = [r for r in step_ok if r.get(ct, 0) > 0]
        total_inst = sum(r.get(ct, 0) for r in step_ok)
        s = _qs([r.get(ct) for r in with_type])
        curve_rows.append({
            'Curve_type': ct,
            'Models_with': len(with_type),
            'Prevalence_%': _pct(len(with_type), len(step_ok)),
            'Total_instances': total_inst,
            'Median_per_model': s['med'],
            'Max': s['max'],
        })
    write_csv(f"{out_dir}/t4_curve_types.csv", curve_rows,
              ['Curve_type','Models_with','Prevalence_%','Total_instances','Median_per_model','Max'])

    # Table 5: Physical metrics (STL) by category
    phys_rows = []
    for cat in CAT_ALL:
        sub = [r for r in stl_ok if r.get('category') == cat]
        if not sub: continue
        s_tri = _qs([r.get('n_triangles') for r in sub])
        s_sa  = _qs([r.get('surface_area') for r in sub])
        s_vol = _qs([r.get('volume') for r in sub])
        s_diag= _qs([r.get('bbox_diagonal') for r in sub])
        s_z   = _qs([r.get('bbox_z') for r in sub])
        s_comp= _qs([r.get('compactness') for r in sub])
        phys_rows.append({
            'Category': cat, 'N_stl': len(sub),
            'Triangles_med': s_tri['med'], 'Triangles_max': s_tri['max'],
            'SurfArea_med': s_sa['med'],
            'Volume_med': s_vol['med'],
            'BBoxDiag_med': s_diag['med'], 'BBoxDiag_max': s_diag['max'],
            'Height_med': s_z['med'],
            'Compactness_med': s_comp['med'],
        })
    write_csv(f"{out_dir}/t5_physical_by_category.csv", phys_rows,
              ['Category','N_stl','Triangles_med','Triangles_max','SurfArea_med',
               'Volume_med','BBoxDiag_med','BBoxDiag_max','Height_med','Compactness_med'])

    # Table 6: Full per-field descriptive stats
    all_fields = [
        ('n_faces','Face count'), ('n_edges','Edge count'), ('n_vertices','Vertex count'),
        ('n_shells','Shell count'), ('n_solids','Solid count'), ('n_inner_bounds','Inner bounds'),
        ('edges_per_face','Edges per face'), ('surf_diversity','Surface type diversity'),
        ('bspline_ratio','BSpline surface ratio'), ('genus_approx','Genus (approx)'),
        ('step_kb','STEP file size (KB)'),
    ]
    stl_fields = [
        ('n_triangles','Triangle count'), ('surface_area','Surface area'),
        ('volume','Volume'), ('bbox_x','BBox X'), ('bbox_y','BBox Y'), ('bbox_z','BBox Z'),
        ('bbox_diagonal','BBox diagonal'), ('compactness','Compactness'),
        ('stl_kb','STL file size (KB)'),
    ]
    stat_rows = []
    for field, label in all_fields:
        s = _qs([r.get(field) for r in step_ok])
        stat_rows.append({'Metric': label, 'Source': 'STEP', **s})
    for field, label in stl_fields:
        s = _qs([r.get(field) for r in stl_ok])
        stat_rows.append({'Metric': label, 'Source': 'STL', **s})
    write_csv(f"{out_dir}/t6_overall_descriptive_stats.csv", stat_rows,
              ['Metric','Source','n','min','p25','med','p75','max','mean'])

    # Table 7: Raw dump
    all_keys = ['model_id','batch','category','model_name','has_description',
                'has_alignment','n_render_views','step_ok','step_kb'] + \
               list(TRACKED) + \
               ['n_faces','n_edges','n_vertices','n_shells','n_solids','n_inner_bounds',
                'genus_approx','edges_per_face','is_multi_body','has_holes',
                'surf_diversity','bspline_ratio',
                'stl_ok','stl_kb','n_triangles','surface_area','volume',
                'bbox_x','bbox_y','bbox_z','bbox_diagonal','compactness']
    raw_rows = [{k: r.get(k,'') for k in all_keys} for r in results]
    write_csv(f"{out_dir}/t7_raw.csv", raw_rows, all_keys)

    print(f"\nTables saved → {out_dir}/")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print(f"Scanning: {DATASET_ROOT}  (batch={TARGET_BATCH or 'ALL'})")
    models = list(find_models(DATASET_ROOT, TARGET_BATCH))
    total  = len(models)
    print(f"Found {total} models.\n")

    results, done_ids = [], set()
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            results = json.load(f)
        done_ids = {r['model_id'] for r in results}
        print(f"Resuming — {len(done_ids)} already done.\n")

    t0 = time.time()
    for idx, m in enumerate(models):
        mid = m['model_id']
        if mid in done_ids:
            continue
        elapsed = time.time() - t0
        rate    = max((idx+1 - len(done_ids)) / max(elapsed,1), 1e-6)
        remain  = (total - idx - 1) / rate
        print(f"[{idx+1:>5}/{total}] {mid[:72]:<72}  ~{remain/60:.1f}min",
              end='\r', flush=True)

        rec = {**m}
        if m['step_path'] and os.path.exists(m['step_path']):
            s = parse_step(m['step_path'])
            rec.update(s)
            rec.update(derive_topo(s))
        else:
            rec['step_ok'] = False

        if m['stl_path'] and os.path.exists(m['stl_path']):
            rec.update(parse_stl(m['stl_path']))
        else:
            rec['stl_ok'] = False

        results.append(rec)
        if len(results) % 100 == 0:
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
