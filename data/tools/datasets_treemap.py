import os
import matplotlib.pyplot as plt
import squarify
import seaborn as sb
from utils import read_stats, apply_rehydratation

# Specify the directory path where your JSON files are located
data_path = os.path.join(os.getenv("OpenLLM_OUTPUT"), "data/tokens_ablation/stats")

df = read_stats(data_path)
df = apply_rehydratation(df, column_name='total_tokens')
df['total_tokens_estimation'] = df['total_tokens_rehydrated'] * 20

def process_names(df):
    patterns = [
        'wik',
        'gallica'
    ]
    df['name_pro'] = df['name_pro'].str.extract(f"({'|'.join(patterns)})")
    df['name_pro'] = df['name_pro'].fillna(df['name_pro'])
    return df

df = process_names(df)
df = df.groupby('name_pro')['total_tokens_estimation'].sum().reset_index()

# grouped_df = grouped_df.sort_values('total_tokens_estimation', ascending=True)

labels = [f"{name}\n{tokens/ 1e9:.1f} B" for name, tokens in zip(df['name_pro'], df['total_tokens_estimation'])]

# Treemap
plt.figure(figsize=(10, 6))
squarify.plot(sizes = df['total_tokens_estimation'], label = labels,
              pad = 0.05, alpha=0.7,
              text_kwargs = {'fontsize': 6, 'color': 'black'},
              color = sb.color_palette("rocket", len(df)))

plt.axis("off")
plt.title(f"Tokens per dataset (estimation, with rehydratation)\n Total tokens: {df['total_tokens_estimation'].sum()/ 1e9:.1f} B")
plt.savefig(os.path.join(os.getenv("OpenLLM_OUTPUT"), "data/tokens_ablation/figs/treemap.png"), dpi=300, bbox_inches='tight')
# plt.show() 
