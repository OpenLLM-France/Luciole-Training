import os
import subprocess
import yaml
import argparse
import numpy as np
from collections import defaultdict
from numerize import numerize
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

"""
Categorizes and created a bar chart for posttraining datamixes. 
"""

CATEGORIES = {"_math":"Math", "_chat":"Chat", "_if":"IF", "_tools":"Tools", "_code":"Code", "_nli":"NLI", "_science":"STEM", "_rag":"RAG", "_hardcoded":"Hardcoded", "_translation":"Translation", "_multilingual":"Multiling", "_safety":"Safety"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "yaml_file",
        type=str,
        help=".yaml file that was used for `run_tokenization.py`.",
    )
    parser.add_argument(
        "--level",
        required=True,
        type=str,
        help="tokens or samples",
    )
    
    args = parser.parse_args()
    yaml_file = args.yaml_file

    # Load the YAML config
    with open(yaml_file, "r") as f:
        yaml_data = yaml.safe_load(f)

    output_dir = yaml_data["output_dir"]
    
    TOKENS=defaultdict(int)
    SAMPLES=defaultdict(int)

    # Iterate through each dataset entry and populate TOKENS and SAMPLES dicts
    for dataset in yaml_data["datasets"]:
        name = dataset["name"]
        print(name)
        weight = dataset["weight"]
        cat = [c for c in CATEGORIES.keys() if c in name]
        assert len(cat) == 1
        output_np = os.path.join(output_dir, f"{name}.npy")

        if os.path.isfile(output_np):
            #open it, take weights, calculate tokens and samples, add to category dict 
            data=np.load(output_np, allow_pickle=True)
            #print(output_np)
            #print(type(data))
            if weight < 1.0:
                rng = np.random.default_rng()
                sub = rng.choice(data, size=int(len(data)*weight), replace=False)
                TOKENS[cat[0]] += np.sum(sub)
                SAMPLES[cat[0]] += len(sub)
            else:
                TOKENS[cat[0]] += np.sum(data)
                SAMPLES[cat[0]] += len(data)         
        else:
            print("--------------------------------------")
            print(f"❌ No file found at {output_np}.")
            print("--------------------------------------")

    # Create barchart 
    print(f"📊 Creating bar chart.")
    
    """
    print("TOKENS COUNTS")
    for key in TOKENS.keys():
        print(key, TOKENS[key])
        print('===================')
    print("SAMPLES COUNTS")
    for key in SAMPLES.keys():
        print(key, SAMPLES[key])
        print('===================')
    """
    os.makedirs(os.path.join(output_dir, "visuals"), exist_ok=True)
    saved_chart_path = os.path.join(output_dir, "visuals", f"barchart_{args.level}.png")
    
    if args.level=="samples":
        FULL_COUNT = sum([SAMPLES[key] for key in SAMPLES.keys()])
        data = [(SAMPLES[key], CATEGORIES[key]) for key in SAMPLES.keys()]
    else:
        FULL_COUNT = sum(TOKENS[key] for key in TOKENS.keys())
        data = [(TOKENS[key], CATEGORIES[key]) for key in TOKENS.keys()]

    data = sorted(data, reverse=False)
    values, labels = zip(*data)
    
    #set properties
    font_light = "/lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/data/viz_assets/fonts/Montserrat-Light.ttf"
    font_medium = "/lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/data/viz_assets/fonts/Montserrat-Medium.ttf"

    prop = fm.FontProperties(fname=font_light)
    prop_title = fm.FontProperties(fname=font_medium)
    prop_title.set_size(24)

    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    palette = "mako"
    if args.level=="samples":
        palette = "magma"

    # NB: using a "mako" or "viridis" palette gives it a modern gradient look
    bars = ax.barh(labels, values, color=sns.color_palette(palette, len(values)))
    
    ax.spines['top'].set_visible(False)    # Remove top border
    ax.spines['right'].set_visible(False)  # Remove right border
    ax.spines['bottom'].set_visible(False) # Remove bottom border
    ax.get_xaxis().set_visible(False)      # Hide x-axis (we will label bars directly)

    for label in ax.get_yticklabels():
        label.set_fontproperties(prop)
        label.set_fontsize(14)

    padding = max(values) * 0.02
    # Add data labels to the end of each bar
    for i, bar in enumerate(bars):
        width = bar.get_width()
        name = numerize.numerize(int(width))
        ax.text(width + padding, bar.get_y() + bar.get_height()/2,
                f'{name}', ha='left', va='center',
                fontsize=14, fontproperties=prop, fontweight='bold', color='#444444')


    ax.set_title(f'Luciole Instruct Mix: {numerize.numerize(int(FULL_COUNT))} {args.level}', fontproperties=prop_title, pad=20, loc='left')

    plt.tight_layout()
    print('chart ', saved_chart_path)
    plt.savefig(saved_chart_path)
    plt.close()

    
