import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def plot_val_loss(csv_path):
    df = pd.read_csv(csv_path)
    loss_data = df[df['val_loss'].notna()].reset_index(drop=True)

    plt.figure(figsize=(10, 5))
    plt.plot(loss_data['epoch'], loss_data['val_loss'], marker='o', linestyle='-', label="Val Loss")
    plt.title("Validation Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.splitext(csv_path)[0] + ".png"
    plt.savefig(save_path)
    print(f"[INFO] Val-Loss plot saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot val_loss curve from metrics log file.")
    parser.add_argument("csv_path", type=str, help="Path to the metrics.csv file.")
    args = parser.parse_args()

    plot_val_loss(args.csv_path)
