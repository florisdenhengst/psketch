import yaml
import glob

files = glob.glob('*modular*0.1.yaml')

cfgs = []
for f in files:
    with open(f) as fp:
        cfgs.append(yaml.load(fp))

for cfg in cfgs:
    if cfg['world']['name'][:5] == 'Craft':
        to_name = 'craft_modular-minimal-{}-{}.yaml'
    else:
        to_name = 'light_modular-minimal-{}-{}.yaml'
    cfg['model']['shaping_reward'] = 1.0
    to_name = to_name.format(cfg['seed'], cfg['model']['shaping_reward'])
    cfg['name'] = to_name[:-5]
    with open(to_name, 'w+') as fp:
        yaml.dump(cfg, fp)


