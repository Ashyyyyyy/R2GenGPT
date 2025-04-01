import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def plot_loss_from_csv(csv_path):
    # 读取 CSV 文件
    df = pd.read_csv(csv_path)

    # 筛选非空 loss 数据
    loss_data = df[df['loss'].notna()].reset_index(drop=True)

    # 绘图
    plt.figure(figsize=(10, 5))
    plt.plot(loss_data['loss'], label="Loss", color='blue')
    plt.title("Loss Curve")
    plt.xlabel("Index")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # 构造保存路径（与CSV同目录、同名，扩展名改为 .png）
    save_path = os.path.splitext(csv_path)[0] + ".png"
    plt.savefig(save_path)
    print(f"Loss plot saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot loss curve from a CSV file.")
    parser.add_argument("csv_path", type=str, help="Path to the CSV file containing loss data.")
    args = parser.parse_args()

    plot_loss_from_csv(args.csv_path)
