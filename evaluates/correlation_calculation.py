'''
Calculate correlation coefficient from csv
'''

import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau

# 读取CSV文件
file_dir = r"H:\Research\results\llama3_prompt_score_output.csv"
df = pd.read_csv(file_dir)

print("列名：", df.columns)

gt = df['gt_score']
marked = df['marked_score']

# 计算皮尔逊相关系数
pearson_corr, pearson_p = pearsonr(gt, marked)

# 计算斯皮尔曼等级相关系数
spearman_corr, spearman_p = spearmanr(gt, marked)

# 计算Kendall Tau等级相关系数
kendall_corr, kendall_p = kendalltau(gt, marked)

# 打印结果
print(file_dir)
print(f"Pearson correlation: {pearson_corr:.4f} (p = {pearson_p:.4g})")
print(f"Spearman correlation: {spearman_corr:.4f} (p = {spearman_p:.4g})")
print(f"Kendall  correlation: {kendall_corr:.4f} (p = {kendall_p:.4g})")