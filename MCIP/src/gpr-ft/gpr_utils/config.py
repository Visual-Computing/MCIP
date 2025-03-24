import yaml
from gpr_utils.utils import Dotdict

def get_config(cfg_path):

    """Read a config file and return the experiment parameters"""
   
    with open(cfg_path, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    print("Current config:", Dotdict(yaml_cfg))
    return Dotdict(yaml_cfg)