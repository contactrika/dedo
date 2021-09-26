import argparse
import yaml
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, required=True)
parser.add_argument('--env', type=str, required=True)
args = parser.parse_args()

yaml_file = yaml.load(open(args.file))
yaml_file['name'] = args.env
yaml_file['command'].append(args.env)
print(yaml_file)
sweep_id = wandb.sweep(yaml_file, project='hp-search-dedo')
print(sweep_id)