import yaml
import glob

files = glob.glob('craft*modular-[0-9]-forcestop-*.yaml')

HINT_FILE_LOC = 'resources/craft/'
tasks = ['axe',
        'bed',
        'bridge',
        'cloth',
        'gem',
        'gold',
        'plank',
        'rope',
        'shears',
        'stick',
]
task_hints = [t + '.yaml' for t in tasks]

cfgs = []
for f in files:
    with open(f) as fp:
        cfgs.append(yaml.load(fp))

for cfg in cfgs:
    if cfg['seed'] == 1.0:
        continue
    print(cfg['name'])
    cfgname = cfg['name']
    for ti, task in enumerate(tasks):
        cfg['name'] = cfgname + '-' + task
        cfg['trainer']['hints'] = HINT_FILE_LOC + task_hints[ti]
        fname = '/tmp/' + cfg['name'] + '.yaml'
        with open(fname, 'w+') as fp:
            yaml.dump(cfg, fp)


