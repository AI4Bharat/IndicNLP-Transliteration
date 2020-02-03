from munch import munchify
import os, sys, yaml

def load_and_validate_cfg(config_json):
    if not os.path.isfile(config_json):
        sys.exit('Train file ' + config_json + ' NOT FOUND!')
    with open(config_json) as f:
        config = munchify(yaml.safe_load(f))
    # TODO: Check if all required params are there, and create folders
    return config