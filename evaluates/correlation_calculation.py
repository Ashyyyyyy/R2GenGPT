'''
Calculate correlation coefficient, QWK, and MSE from csv
'''

import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import cohen_kappa_score, mean_squared_error
import numpy as np

# 读取CSV文件
file_dir = r"H:\Research\results\mrscore_predictions_sigmoided_reward_custom_loss.csv"
df = pd.read_csv(file_dir)

print("列名：", df.columns)

gt = df['gt_score']
marked = df['marked_score']

# 保留原始浮点分数用于 Pearson、Spearman、Kendall、MSE
# QWK 需要离散整数，所以单独做 round + clip 处理
marked_rounded = np.clip(np.round(marked), 0, 3).astype(int)
gt_int = gt.astype(int)

# 计算皮尔逊相关系数
pearson_corr, pearson_p = pearsonr(gt, marked)

# 计算斯皮尔曼等级相关系数
spearman_corr, spearman_p = spearmanr(gt, marked)

# 计算Kendall Tau等级相关系数
kendall_corr, kendall_p = kendalltau(gt, marked)

# 计算 Quadratic Weighted Kappa
qwk = cohen_kappa_score(gt_int, marked_rounded, weights='quadratic')

# 计算 MSE
mse = mean_squared_error(gt, marked)

# 打印结果
print(file_dir)
print(f"Pearson correlation: {pearson_corr:.4f} (p = {pearson_p:.4g})")
print(f"Spearman correlation: {spearman_corr:.4f} (p = {spearman_p:.4g})")
print(f"Kendall  correlation: {kendall_corr:.4f} (p = {kendall_p:.4g})")
print(f"Quadratic Weighted Kappa (QWK): {qwk:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
