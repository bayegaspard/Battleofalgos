import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_plots(csv_path="research/results/experiment_results_v1_partial.csv"):
    if not os.path.exists(csv_path):
        print(f"Results CSV not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # Aggregated Summary
    summary = df.groupby("Model").mean().reset_index()
    
    # 1. Accuracy Comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Model", y="Accuracy", data=summary, palette="viridis")
    plt.title("Model Accuracy Comparison (SOC Task)", fontsize=14)
    plt.ylabel("Accuracy (0.0 - 1.0)")
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("research/results/accuracy_comparison.png")
    
    # 2. Reasoning Density vs Accuracy (Bubble Plot)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="Reasoning_Density", y="Accuracy", size="Words", hue="Model", 
                    data=summary, sizes=(100, 1000), alpha=0.7)
    plt.title("Reasoning Density vs. Accuracy", fontsize=14)
    plt.xlabel("Reasoning Density (Thinking/Total Words)")
    plt.ylabel("Accuracy")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    plt.savefig("research/results/reasoning_vs_accuracy.png")

    # 3. Technical Precision Heatmap
    plt.figure(figsize=(10, 6))
    pivot = summary.pivot_table(index="Model", values="Tech_Precision")
    sns.heatmap(pivot, annot=True, cmap="YlGnBu", fmt=".3f")
    plt.title("Technical Precision Heatmap", fontsize=14)
    plt.savefig("research/results/precision_heatmap.png")

    print("\n[Vizu] IEEE-style plots generated in research/results/")

if __name__ == "__main__":
    generate_plots()
