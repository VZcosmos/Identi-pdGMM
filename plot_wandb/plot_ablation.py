import os
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = "plots_ablation"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 12,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
})

colors = {
    "ISCA": "#E377E3",
    "ISCA + BN": "#6FA3EF",
    "ISCA + MSE": "#5FA76E",
}

linestyles = {
    "Simple": "-",
    "Complicate": "--",
}

setting_display = {
    "Simple": "Simple",
    "Complicate": "Complex",
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
    ("bn", "stage1", "mcc_spearman"): "csv/ablation_bn_stage1_mcc_spearman.csv",
    ("bn", "stage1", "mcc_pearson"): "csv/ablation_bn_stage1_mcc_pearson.csv",
    ("bn", "stage2", "mcc_spearman"): "csv/ablation_bn_stage2_mcc_spearman.csv",
    ("bn", "stage2", "mcc_pearson"): "csv/ablation_bn_stage2_mcc_pearson.csv",

    ("mse", "stage1", "mcc_spearman"): "csv/ablation_mse_stage1_mcc_spearman.csv",
    ("mse", "stage1", "mcc_pearson"): "csv/ablation_mse_stage1_mcc_pearson.csv",
    ("mse", "stage2", "mcc_spearman"): "csv/ablation_mse_stage2_mcc_spearman.csv",
    ("mse", "stage2", "mcc_pearson"): "csv/ablation_mse_stage2_mcc_pearson.csv",
}

r2_csv_files = {
    "bn": "csv/ablation_bn_stage1_r2.csv",
    "mse": "csv/ablation_mse_stage1_r2.csv",
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


def get_ablation_groups(ablation):
    if ablation == "bn":
        return {
            "ISCA": {
                "Simple": "Ours_Simple_cauchy",
                "Complicate": "Ours_Complicate_cauchy",
            },
            "ISCA + BN": {
                "Simple": "Ours_BN_Simple_cauchy",
                "Complicate": "Ours_BN_Complicate_cauchy",
            },
        }

    if ablation == "mse":
        return {
            "ISCA": {
                "Simple": "Ours_Simple_cauchy",
                "Complicate": "Ours_Complicate_cauchy",
            },
            "ISCA + MSE": {
                "Simple": "Ours_MSE_Simple_cauchy",
                "Complicate": "Ours_MSE_Complicate_cauchy",
            },
        }

    raise ValueError(f"Unknown ablation: {ablation}")


def plot_mcc_curve(csv_path, ablation, stage, metric_name):
    df = pd.read_csv(csv_path)

    metric = f"{stage}/{metric_name}"
    x_col = find_x_col(df, stage)

    plt.figure(figsize=(4.2, 3.0))

    has_plot = False
    groups = get_ablation_groups(ablation)

    for method_name, setting_to_group in groups.items():
        for setting, group_name in setting_to_group.items():
            mean_col, min_col, max_col = find_group_columns(df, group_name, metric)

            if mean_col is None:
                print(f"[Skip] {csv_path}: missing columns for {group_name} - {metric}")
                continue

            x = df[x_col]
            y_mean = df[mean_col]
            y_min = df[min_col]
            y_max = df[max_col]

            label = f"{method_name} ({setting_display[setting]})"

            plt.plot(
                x,
                y_mean,
                label=label,
                color=colors[method_name],
                linestyle=linestyles[setting],
                linewidth=2.3,
            )

            plt.fill_between(
                x,
                y_min,
                y_max,
                color=colors[method_name],
                alpha=0.14,
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
        fontsize=9,
        loc="lower right",
        handlelength=1.4,
        labelspacing=0.25,
        borderaxespad=0.2,
    )

    plt.tight_layout(pad=0.5)

    out_name = f"ablation_{ablation}_{stage}_{metric_name}"
    png_path = os.path.join(OUT_DIR, f"{out_name}.png")

    plt.savefig(
        png_path,
        dpi=400,
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close()

    print(f"[Saved] {png_path}")


def plot_r2_bar(csv_path, ablation):
    df = pd.read_csv(csv_path)

    groups = get_ablation_groups(ablation)

    labels = []
    values = []
    bar_colors = []
    hatches = []

    for method_name, setting_to_group in groups.items():
        for setting, group_name in setting_to_group.items():
            candidates = [
                f"Group: {group_name} - stage1/eval_r2_standard",
                f"Group: {group_name} - stage1/eval_r2",
            ]

            metric_col = None
            for col in candidates:
                if col in df.columns:
                    metric_col = col
                    break

            if metric_col is None:
                print(f"[Skip] {csv_path}: missing R2 column for {group_name}")
                continue

            value = get_last_valid_value(df, metric_col)
            if value is None:
                print(f"[Skip] {csv_path}: no valid value in {metric_col}")
                continue

            labels.append(f"{method_name}\n({setting_display[setting]})")
            values.append(value)
            bar_colors.append(colors[method_name])
            hatches.append("" if setting == "Simple" else "//")

    if not values:
        print(f"[No plot] {csv_path}: no valid R2 values found.")
        return

    plt.figure(figsize=(4.2, 3.0))

    bars = plt.barh(
        labels,
        values,
        color=bar_colors,
        edgecolor="black",
        linewidth=0.6,
    )

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

    out_name = f"ablation_{ablation}_stage1_r2"
    png_path = os.path.join(OUT_DIR, f"{out_name}.png")

    plt.savefig(
        png_path,
        dpi=400,
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close()

    print(f"[Saved] {png_path}")


def main():
    for (ablation, stage, metric_name), csv_path in csv_files.items():
        if not os.path.exists(csv_path):
            print(f"[Missing file] {csv_path}")
            continue

        plot_mcc_curve(csv_path, ablation, stage, metric_name)

    for ablation, csv_path in r2_csv_files.items():
        if not os.path.exists(csv_path):
            print(f"[Missing file] {csv_path}")
            continue

        plot_r2_bar(csv_path, ablation)


if __name__ == "__main__":
    main()