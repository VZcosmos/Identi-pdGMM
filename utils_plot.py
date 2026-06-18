import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import wandb

def plot_latent_histograms_untrimmed(z_raw, z_true, z_hat, save_dir, filename="latent_histograms.png", max_dims=3):
    """
    Plots individual histograms for each latent dimension without trimming extreme values.
    Uses a log-scale on the Y-axis so Cauchy outliers remain visible.
    """
    # 1. Convert tensors to numpy
    if hasattr(z_raw, 'detach'): z_raw = z_raw.detach().cpu().numpy()
    if hasattr(z_true, 'detach'): z_true = z_true.detach().cpu().numpy()
    if hasattr(z_hat, 'detach'): z_hat = z_hat.detach().cpu().numpy()

    z_n = z_raw.shape[1]
    dims_to_plot = min(z_n, max_dims)

    # 2. Set up the figure grid
    fig, axes = plt.subplots(dims_to_plot, 3, figsize=(15, 4 * dims_to_plot), dpi=150)
    sns.set_theme(style="whitegrid") 
    
    titles = ["1. Raw (Pre-Mask)", "2. True (Masked)", "3. Predicted (z_hat)"]

    for d in range(dims_to_plot):
        data_lists = [z_raw[:, d], z_true[:, d], z_hat[:, d]]
        
        for col, data in enumerate(data_lists):
            ax = axes[d, col] if dims_to_plot > 1 else axes[col]
            
            # NO TRIMMING: Pass the raw data directly into the histogram.
            # We use 100 bins to give the extreme range more granularity.
            sns.histplot(
                data, 
                bins=100, 
                kde=False,                 # KDE fails mathematically on raw Cauchy ranges
                color='darkkhaki', 
                edgecolor='black',
                log_scale=(False, True),   # Log-scale Y-axis so outlier bins (count=1) are visible
                ax=ax
            )
            
            # Since bins=100, the width is the total range divided by 100
            bin_width = (data.max() - data.min()) / 100.0
            ax.text(
                0.95, 0.95, 
                f"Bin Width: {bin_width:,.2f}", 
                transform=ax.transAxes, 
                fontsize=10,
                verticalalignment='top', 
                horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray')
            )

            ax.set_xlabel(f"Values of z_{d+1}")
            ax.set_ylabel("Count")
            
            if d == 0:
                ax.set_title(titles[col], fontsize=14, fontweight='bold')

    plt.tight_layout()
    
    # 5. Save and log
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    plt.close()
    
    # Safely log to W&B if active
    try:
        wandb.log({f"Histograms/{filename}": wandb.Image(save_path)})
    except Exception as e:
        pass