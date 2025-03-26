import fiddle as fd

from launch_pretraining import configure_recipe
from pprint import pprint
from nemo.collections import llm

def run_pretraining():
    pretrain = llm.llama3_8b.pretrain_recipe(
        name="test",
        dir="print_config",
        num_nodes=1,
        num_gpus_per_node=1,
    )
    
    pprint(fd.build(pretrain.model.config))

# This condition is necessary for the script to be compatible with Python's multiprocessing module.
if __name__ == "__main__":
    run_pretraining()