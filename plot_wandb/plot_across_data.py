import os
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = "plots_across_data"
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# Fonts for small LaTeX subplots
# =========================

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 12,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
})

dataset_colors = {
    "gauss": "#4C72B0",
    "gumbel": "#55A868",
    "cauchy": "#C44E52",
    "trunc_cauchy": "#8172B3",
}

linestyles = {
    "Simple": "-",
    "Complicate": "--",
}

dataset_display = {
    "gauss": "Gaussian",
    "gumbel": "Gumbel",
    "cauchy": "Cauchy",
    "trunc_cauchy": "Trunc. Cauchy",
}

setting_display = {
    "Simple": "Simple",
    "Complicate": "Complex",
}

model_display = {
    "Ours": "ISCA",
    "Danru": "Xu et al. (2024)",
    "VaDE": "VaDE",
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
    ("Ours", "stage1", "mcc_spearman"): "csv/across_ours_stage1_mcc_spearman.csv",
    ("Ours", "stage1", "mcc_pearson"): "csv/across_ours_stage1_mcc_pearson.csv",
    ("Ours", "stage2", "mcc_spearman"): "csv/across_ours_stage2_mcc_spearman.csv",
    ("Ours", "stage2", "mcc_pearson"): "csv/across_ours_stage2_mcc_pearson.csv",

    ("Danru", "stage1", "mcc_spearman"): "csv/across_danru_stage1_mcc_spearman.csv",
    ("Danru", "stage1", "mcc_pearson"): "csv/across_danru_stage1_mcc_pearson.csv",
    ("Danru", "stage2", "mcc_spearman"): "csv/across_danru_stage2_mcc_spearman.csv",
    ("Danru", "stage2", "mcc_pearson"): "csv/across_danru_stage2_mcc_pearson.csv",

    ("VaDE", "stage1", "mcc_spearman"): "csv/across_vade_stage1_mcc_spearman.csv",
    ("VaDE", "stage1", "mcc_pearson"): "csv/across_vade_stage1_mcc_pearson.csv",
    ("VaDE", "stage2", "mcc_spearman"): "csv/across_vade_stage2_mcc_spearman.csv",
    ("VaDE", "stage2", "mcc_pearson"): "csv/across_vade_stage2_mcc_pearson.csv",
}

r2_csv_files = {
    "Ours": "csv/across_ours_stage1_r2.csv",
    "Danru": "csv/across_danru_stage1_r2.csv",
    "VaDE": "csv/across_vade_stage1_r2.csv",
}


def get_datasets_for_model(model_key):
    if model_key == "Ours":
        return ["gauss", "gumbel", "cauchy", "trunc_cauchy"]
    return ["gauss", "gumbel", "cauchy"]


def get_group_name(model_key, setting, dataset):
    if dataset == "trunc_cauchy":
        return f"{model_key}_trunc_{setting}_cauchy"
    return f"{model_key}_{setting}_{dataset}"


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


def plot_mcc_curve(csv_path, model_key, stage, metric_name):
    df = pd.read_csv(csv_path)

    metric = f"{stage}/{metric_name}"
    x_col = find_x_col(df, stage)

    plt.figure(figsize=(4.4, 3.2))

    has_plot = False

    for dataset in get_datasets_for_model(model_key):
        for setting in ["Simple", "Complicate"]:
            group_name = get_group_name(model_key, setting, dataset)
            mean_col, min_col, max_col = find_group_columns(df, group_name, metric)

            if mean_col is None:
                print(f"[Skip] {csv_path}: missing columns for {group_name} - {metric}")
                continue

            x = df[x_col]
            y_mean = df[mean_col]
            y_min = df[min_col]
            y_max = df[max_col]

            label = f"{dataset_display[dataset]} ({setting_display[setting]})"

            plt.plot(
                x,
                y_mean,
                label=label,
                color=dataset_colors[dataset],
                linestyle=linestyles[setting],
                linewidth=2.3,
            )

            plt.fill_between(
                x,
                y_min,
                y_max,
                color=dataset_colors[dataset],
                alpha=0.12,
                linewidth=0,
            )

            has_plot = True

    if not has_plot:
        plt.close()
        print(f"[No plot] {csv_path}: no matching columns found.")
        return

    plt.title(f"{stage_display[stage]} {metric_display[metric_name]}")
    plt.xlabel("Step")
    plt.ylabel(metric_display[metric_name])
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3)

    plt.legend(
        frameon=False,
        fontsize=8.5,
        ncol=2,
        loc="lower right",
        handlelength=1.6,
        columnspacing=0.8,
        labelspacing=0.25,
        borderaxespad=0.2,
    )

    plt.tight_layout(pad=0.5)

    out_name = f"across_{model_key.lower()}_{stage}_{metric_name}"
    png_path = os.path.join(OUT_DIR, f"{out_name}.png")

    plt.savefig(png_path, dpi=400, bbox_inches="tight", pad_inches=0.03)
    plt.close()

    print(f"[Saved] {png_path}")


def plot_r2_bar(csv_path, model_key):
    df = pd.read_csv(csv_path)

    labels = []
    values = []
    bar_colors = []
    hatches = []

    for dataset in get_datasets_for_model(model_key):
        for setting in ["Simple", "Complicate"]:
            group_name = get_group_name(model_key, setting, dataset)

            candidates = [
                f"Group: {group_name} - stage1/eval_r2_standard",
                f"Group: {group_name} - stage1/eval_r2",
            ]

            metric_col = None
            for c in candidates:
                if c in df.columns:
                    metric_col = c
                    break

            if metric_col is None:
                print(f"[Skip] {csv_path}: missing R2 column for {group_name}")
                continue

            value = get_last_valid_value(df, metric_col)
            if value is None:
                print(f"[Skip] {csv_path}: no valid value in {metric_col}")
                continue

            labels.append(f"{dataset_display[dataset]}\n({setting_display[setting]})")
            values.append(value)
            bar_colors.append(dataset_colors[dataset])
            hatches.append("" if setting == "Simple" else "//")

    if not values:
        print(f"[No plot] {csv_path}: no valid R2 values found.")
        return

    plt.figure(figsize=(4.4, 3.2))

    bars = plt.barh(
        labels,
        values,
        color=bar_colors,
        edgecolor="black",
        linewidth=0.6,
    )
    plt.yticks(fontsize=10)

    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

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

    out_name = f"across_{model_key.lower()}_stage1_r2"
    png_path = os.path.join(OUT_DIR, f"{out_name}.png")

    plt.savefig(png_path, dpi=400, bbox_inches="tight", pad_inches=0.03)
    plt.close()

    print(f"[Saved] {png_path}")


def main():
    for (model_key, stage, metric_name), csv_path in csv_files.items():
        if not os.path.exists(csv_path):
            print(f"[Missing file] {csv_path}")
            continue

        plot_mcc_curve(csv_path, model_key, stage, metric_name)

    for model_key, csv_path in r2_csv_files.items():
        if not os.path.exists(csv_path):
            print(f"[Missing file] {csv_path}")
            continue

        plot_r2_bar(csv_path, model_key)


if __name__ == "__main__":
    main()