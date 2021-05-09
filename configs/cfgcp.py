import yaml
import glob

files = glob.glob('*.yaml')

cfgs = []
for f in files:
    with open(f) as fp:
        cfgs.append(yaml.load(fp))
for cfg in cfgs:
    to_name = cfg['name'].replace('modular', 'mamc')
    cfg['model']['name'] = 'ModularActorModularCriticModel'
    with open(to_name, 'w+') as fp:
        yaml.dump(cfg, fp)


