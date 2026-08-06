import json

with open('experiments/slam_3d_dashboard.py', 'r', encoding='utf-8') as f:
    text = f.read()

cells = []
for block in text.split('# %%'):
    block = block.strip()
    if not block:
        continue
    if block.startswith('[markdown]'):
        lines = block.split('\n')[1:]
        cells.append({
            'cell_type': 'markdown',
            'metadata': {},
            'source': [l + '\n' for l in lines]
        })
    else:
        cells.append({
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [l + '\n' for l in block.split('\n')]
        })

nb = {'cells': cells, 'metadata': {}, 'nbformat': 4, 'nbformat_minor': 4}

with open('experiments/slam_3d_dashboard.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Dashboard notebook successfully created.')
