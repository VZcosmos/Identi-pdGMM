import os
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = "plots"
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# Fonts for 0.32\textwidth subplots
# =========================

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 12,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 12,
})

colors = {
    "VaDE": "#5FA76E",
    "Danru": "#6FA3EF",
    "Ours": "#E377E3",
}

display_names = {
    "VaDE": "VaDE",
    "Danru": "Xu et al. (2024)",
    "Ours": "ISCA",
}

dist_display = {
    "gauss": "Gaussian",
    "gumbel": "Gumbel",
    "cauchy": "Cauchy",
}

setting_display = {
    "Simple": "Simple",
    "Complex": "Complex",
}

setting_group_name = {
    "Simple": "Simple",
    "Complex": "Complicate",
}

metric_display = {
    "mcc_spearman": "MCC (Spearman)",
    "mcc_pearson": "MCC (Pearson)",
}

stage_display = {
    "stage1": "Stage 1",
    "stage2": "Stage 2",
}

csv_files = {
    ("gauss", "Simple", "stage1", "mcc_spearman"): "csv/gauss_simple_stage1_mcc_spearman.csv",
    ("gauss", "Simple", "stage1", "mcc_pearson"): "csv/gauss_simple_stage1_mcc_pearson.csv",
    ("gauss", "Simple", "stage2", "mcc_spearman"): "csv/gauss_simple_stage2_mcc_spearman.csv",
    ("gauss", "Simple", "stage2", "mcc_pearson"): "csv/gauss_simple_stage2_mcc_pearson.csv",

    ("gauss", "Complex", "stage1", "mcc_spearman"): "csv/gauss_complex_stage1_mcc_spearman.csv",
    ("gauss", "Complex", "stage1", "mcc_pearson"): "csv/gauss_complex_stage1_mcc_pearson.csv",
    ("gauss", "Complex", "stage2", "mcc_spearman"): "csv/gauss_complex_stage2_mcc_spearman.csv",
    ("gauss", "Complex", "stage2", "mcc_pearson"): "csv/gauss_complex_stage2_mcc_pearson.csv",

    ("gumbel", "Simple", "stage1", "mcc_spearman"): "csv/gumbel_simple_stage1_mcc_spearman.csv",
    ("gumbel", "Simple", "stage1", "mcc_pearson"): "csv/gumbel_simple_stage1_mcc_pearson.csv",
    ("gumbel", "Simple", "stage2", "mcc_spearman"): "csv/gumbel_simple_stage2_mcc_spearman.csv",
    ("gumbel", "Simple", "stage2", "mcc_pearson"): "csv/gumbel_simple_stage2_mcc_pearson.csv",

    ("gumbel", "Complex", "stage1", "mcc_spearman"): "csv/gumbel_complex_stage1_mcc_spearman.csv",
    ("gumbel", "Complex", "stage1", "mcc_pearson"): "csv/gumbel_complex_stage1_mcc_pearson.csv",
    ("gumbel", "Complex", "stage2", "mcc_spearman"): "csv/gumbel_complex_stage2_mcc_spearman.csv",
    ("gumbel", "Complex", "stage2", "mcc_pearson"): "csv/gumbel_complex_stage2_mcc_pearson.csv",

    ("cauchy", "Simple", "stage1", "mcc_spearman"): "csv/cauchy_simple_stage1_mcc_spearman.csv",
    ("cauchy", "Simple", "stage1", "mcc_pearson"): "csv/cauchy_simple_stage1_mcc_pearson.csv",
    ("cauchy", "Simple", "stage2", "mcc_spearman"): "csv/cauchy_simple_stage2_mcc_spearman.csv",
    ("cauchy", "Simple", "stage2", "mcc_pearson"): "csv/cauchy_simple_stage2_mcc_pearson.csv",

    ("cauchy", "Complex", "stage1", "mcc_spearman"): "csv/cauchy_complex_stage1_mcc_spearman.csv",
    ("cauchy", "Complex", "stage1", "mcc_pearson"): "csv/cauchy_complex_stage1_mcc_pearson.csv",
    ("cauchy", "Complex", "stage2", "mcc_spearman"): "csv/cauchy_complex_stage2_mcc_spearman.csv",
    ("cauchy", "Complex", "stage2", "mcc_pearson"): "csv/cauchy_complex_stage2_mcc_pearson.csv",
}

r2_csv_files = {
    ("gauss", "Simple"): "csv/gauss_simple_stage1_r2.csv",
    ("gauss", "Complex"): "csv/gauss_complex_stage1_r2.csv",
    ("gumbel", "Simple"): "csv/gumbel_simple_stage1_r2.csv",
    ("gumbel", "Complex"): "csv/gumbel_complex_stage1_r2.csv",
    ("cauchy", "Simple"): "csv/cauchy_simple_stage1_r2.csv",
    ("cauchy", "Complex"): "csv/cauchy_complex_stage1_r2.csv",
}


def find_x_col(df, stage):
    preferred = f"{stage}/step"
    if preferred in df.columns:
        return preferred

    step_cols = [c for c in df.columns if c.endswith("/step")]
    if step_cols:
        return step_cols[0]

    raise ValueError("No step column found.")


def find_group_columns(df, group_name, metric):
    mean_col = f"Group: {group_name} - {metric}"
    min_col = f"Group: {group_name} - {metric}__MIN"
    max_col = f"Group: {group_name} - {metric}__MAX"

    if mean_col in df.columns and min_col in df.columns and max_col in df.columns:
        return mean_col, min_col, max_col

    return None, None, None


def get_last_valid_value(df, col):
    values = df[col].dropna()
    if len(values) == 0:
        return None
    return float(values.iloc[-1])


def plot_one(csv_path, dist, setting, stage, metric_name):
    df = pd.read_csv(csv_path)

    metric = f"{stage}/{metric_name}"
    x_col = find_x_col(df, stage)

    # Small figure size makes fonts more readable after LaTeX scaling.
    plt.figure(figsize=(4.2, 3.0))

    has_plot = False

    for model_key in ["VaDE", "Danru", "Ours"]:
        group_name = f"{model_key}_{setting_group_name[setting]}_{dist}"
        mean_col, min_col, max_col = find_group_columns(df, group_name, metric)

        if mean_col is None:
            print(f"[Skip] {csv_path}: missing columns for {group_name} - {metric}")
            continue

        x = df[x_col]
        y_mean = df[mean_col]
        y_min = df[min_col]
        y_max = df[max_col]

        plt.plot(
            x,
            y_mean,
            label=display_names[model_key],
            color=colors[model_key],
            linewidth=2.4,
        )

        plt.fill_between(
            x,
            y_min,
            y_max,
            color=colors[model_key],
            alpha=0.20,
            linewidth=0,
        )

        has_plot = True

    if not has_plot:
        plt.close()
        print(f"[No plot] {csv_path}: no matching columns found.")
        return

    # Use short title or remove it entirely if LaTeX caption already explains.
    plt.title(f"{stage_display[stage]} {metric_display[metric_name]}")

    plt.xlabel("Step")
    plt.ylabel(metric_display[metric_name])
    plt.ylim(0, 1.0)

    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False, loc="lower right")

    plt.tight_layout(pad=0.5)

    out_name = f"{dist}_{setting.lower()}_{stage}_{metric_name}"
    png_path = os.path.join(OUT_DIR, f"{out_name}.png")

    plt.savefig(png_path, dpi=400, bbox_inches="tight", pad_inches=0.03)
    plt.close()

    print(f"[Saved] {png_path}")


def plot_stage1_r2(csv_path, dist, setting):
    r2_display_names = {
        "VaDE": "VaDE",
        "Danru": "Xu et al.\n(2024)",
        "Ours": "ISCA",
    }
    df = pd.read_csv(csv_path)

    values = []
    labels = []
    bar_colors = []

    for model_key in ["VaDE", "Danru", "Ours"]:
        group_name = f"{model_key}_{setting_group_name[setting]}_{dist}"

        if model_key == "VaDE":
            metric_col = f"Group: {group_name} - stage1/eval_r2"
        else:
            metric_col = f"Group: {group_name} - stage1/eval_r2_standard"

        if metric_col not in df.columns:
            print(f"[Skip] {csv_path}: missing column {metric_col}")
            continue

        value = get_last_valid_value(df, metric_col)
        if value is None:
            print(f"[Skip] {csv_path}: no valid value in {metric_col}")
            continue

        values.append(value)
        labels.append(r2_display_names[model_key])
        bar_colors.append(colors[model_key])

    if not values:
        print(f"[No plot] {csv_path}: no valid R2 values found.")
        return

    plt.figure(figsize=(4.2, 3.0))

    plt.barh(labels, values, color=bar_colors)
    plt.axvline(0, color="black", linewidth=1.0, alpha=0.75)

    plt.title(r"Stage 1 $R^2$")
    plt.xlabel(r"$R^2$")
    plt.grid(True, axis="x", alpha=0.3)

    min_value = min(values)
    max_value = max(values)

    x_min = min(0.0, min_value)
    x_max = max(1.0, max_value)

    padding = 0.08 * (x_max - x_min)
    if padding == 0:
        padding = 0.1

    plt.xlim(x_min - padding, x_max + padding)
    plt.tight_layout(pad=0.5)

    out_name = f"{dist}_{setting.lower()}_stage1_r2"
    png_path = os.path.join(OUT_DIR, f"{out_name}.png")

    plt.savefig(png_path, dpi=400, bbox_inches="tight", pad_inches=0.03)
    plt.close()

    print(f"[Saved] {png_path}")


def main():
    for (dist, setting, stage, metric_name), csv_path in csv_files.items():
        if not os.path.exists(csv_path):
            print(f"[Missing file] {csv_path}")
            continue

        plot_one(csv_path, dist, setting, stage, metric_name)

    for (dist, setting), csv_path in r2_csv_files.items():
        if not os.path.exists(csv_path):
            print(f"[Missing file] {csv_path}")
            continue

        plot_stage1_r2(csv_path, dist, setting)


if __name__ == "__main__":
    main()