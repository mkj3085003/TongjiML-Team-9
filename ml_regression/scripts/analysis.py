import pandas as pd
import numpy as np  #For mathematical calculatons
import seaborn as sns #For data visualization
import matplotlib.pyplot as plt # For plotting graphs
import plotly.graph_objs as go

import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 定义命令行参数
parser = argparse.ArgumentParser(description="Data Analysis Script")
parser.add_argument('--data-file', type=str, required=True, help="Path to data file (CSV)")
parser.add_argument('--task', type=str, choices=['info', 'correlation'], required=True, help="Choose a data analysis task")

# 解析命令行参数
args = parser.parse_args()

# 读取数据
df = pd.read_csv(args.data_file)

# 执行用户选择的任务
if args.task == 'info':
    df.info()
    num_unique_countries = df['Country'].nunique()
    print(f"Number of unique countries: {num_unique_countries}")
elif args.task == 'correlation':
    spearman_cormatrix = df.corr(method='spearman')
    plt.figure(figsize=(18, 10))
    ax = sns.heatmap(spearman_cormatrix, vmin=-1, vmax=1,
                     center=0, cmap="viridis", annot=True)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.title("Continuous Matrix of Spearman Correlation")
    plt.show()
