# =====================================================================
# COLAB CELL 1 of 2 — SURVEY (reads only, downloads nothing, changes nothing)
#
# Finds every cg_observations.json on your Drive and works out which scene
# each one belongs to. There are several files with this name; rather than
# guessing from the folder name, each file is FINGERPRINTED by its exact
# observation count, which we know from the local copies that produced the
# published 0.324 result. A file matches a scene only if that count is exact.
#
# What we need it for: their per-object id (`obj`). Our local
# object_points.json has {cls, conf, det, frame, x, y} and no `obj`, because
# cg_frontend_to_trace.py dropped it. Without it we cannot corrupt at the
# DETECTION level and simulate ConceptGraphs' own fusion, which is what the
# robustness claim needs to survive review.
# =====================================================================
import json, os, time

from google.colab import drive
drive.mount('/content/drive')

# Narrow this if the walk is slow — e.g. '/content/drive/MyDrive/ssnslam_colab'
ROOT = '/content/drive/MyDrive'
WANT = ('cg_observations.json', 'cg_objects.json')

# exact observation counts from the local files behind the published result
EXPECTED = {"room0": 13124, "room1": 8383, "room2": 8543, "office0": 10956,
            "office1": 7156, "office2": 11061, "office3": 12882,
            "office4": 10047}

print(f'walking {ROOT} ...', flush=True)
hits = []
for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames
                   if not d.startswith('.') and d not in ('__pycache__',)]
    for fn in filenames:
        if fn in WANT:
            hits.append(os.path.join(dirpath, fn))
print(f'found {len(hits)} candidate file(s)\n')

rows = []
for p in sorted(hits):
    if os.path.basename(p) != 'cg_observations.json':
        continue
    try:
        with open(p) as f:
            obs = json.load(f)
        obs = obs.get('points', obs) if isinstance(obs, dict) else obs
        n = len(obs)
        keys = sorted(obs[0].keys()) if n else []
        has_obj = 'obj' in keys
        n_obj = len({o['obj'] for o in obs}) if has_obj else 0
        n_cls = len({o.get('cls') for o in obs}) if n else 0
        match = [s for s, c in EXPECTED.items() if c == n]
    except Exception as e:
        print(f'  UNREADABLE {p}: {type(e).__name__} {e}')
        continue
    rows.append(dict(path=p, n=n, has_obj=has_obj, n_obj=n_obj, n_cls=n_cls,
                     scene=match[0] if len(match) == 1 else None,
                     mtime=time.strftime('%Y-%m-%d %H:%M',
                                         time.localtime(os.path.getmtime(p))),
                     mb=os.path.getsize(p) / 1e6))

print(f'{"scene":<9}{"rows":>7}{"obj?":>6}{"#obj":>6}{"#cls":>6}{"MB":>7}  '
      f'{"modified":<17}path')
print('-' * 110)
for r in sorted(rows, key=lambda r: (r['scene'] is None, r['scene'] or '')):
    tag = r['scene'] or 'NO-MATCH'
    print(f'{tag:<9}{r["n"]:>7}{("yes" if r["has_obj"] else "NO"):>6}'
          f'{r["n_obj"]:>6}{r["n_cls"]:>6}{r["mb"]:>7.1f}  {r["mtime"]:<17}'
          f'{r["path"]}')

matched = {}
for r in rows:
    if r['scene'] and r['has_obj']:
        # if two files fingerprint to the same scene they are copies; keep the
        # one whose folder actually names the scene, else the newest
        prev = matched.get(r['scene'])
        better = (prev is None
                  or (r['scene'] in r['path'] and prev['scene'] not in prev['path'])
                  or r['mtime'] > prev['mtime'])
        if better:
            matched[r['scene']] = r

print(f'\nusable (fingerprint matched AND carries obj): '
      f'{len(matched)}/8 scenes')
missing = [s for s in EXPECTED if s not in matched]
if missing:
    print(f'MISSING: {", ".join(missing)}')
    print('If a file shows obj?=NO it predates the exporter fix and is not '
          'usable.\nIf it shows NO-MATCH its row count does not match any '
          'published scene —\ndo not use it; it came from a different run.')
else:
    print('all 8 scenes present and usable — run cell 2')

PICKED = {s: r['path'] for s, r in matched.items()}
print('\nPICKED =', json.dumps(PICKED, indent=1))


# =====================================================================
# COLAB CELL 2 of 2 — ZIP AND DOWNLOAD  (run only after cell 1 looks right)
#
# Packages the matched cg_observations.json plus its cg_objects.json sibling
# into handoff/<scene>_cgfront/ layout, so it unzips straight into the repo.
# =====================================================================
import shutil, zipfile
from google.colab import files

STAGE = '/content/cg_obs_pack'
shutil.rmtree(STAGE, ignore_errors=True)

n_obs = n_objs = 0
for scene, src in PICKED.items():
    dst_dir = os.path.join(STAGE, 'handoff', f'{scene}_cgfront')
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copy2(src, os.path.join(dst_dir, 'cg_observations.json'))
    n_obs += 1
    sib = os.path.join(os.path.dirname(src), 'cg_objects.json')
    if os.path.exists(sib):
        shutil.copy2(sib, os.path.join(dst_dir, 'cg_objects.json'))
        n_objs += 1

ZIP = '/content/cg_observations_pack.zip'
with zipfile.ZipFile(ZIP, 'w', zipfile.ZIP_DEFLATED) as z:
    for dirpath, _, filenames in os.walk(STAGE):
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            z.write(full, os.path.relpath(full, STAGE))

with zipfile.ZipFile(ZIP) as z:
    names = z.namelist()
print(f'{n_obs} observation file(s), {n_objs} object file(s), '
      f'{len(names)} entries, {os.path.getsize(ZIP)/1e6:.1f} MB')
for nm in sorted(names):
    print('  ', nm)

files.download(ZIP)
