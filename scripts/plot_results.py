#!/usr/bin/env python3
"""
Plot overall scores for all models in results directory.
Outputs both light and dark theme versions.
"""

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

COLOR_PALETTES = {
    "openai": ["#98fb98", "#77dd77", "#5bc85b", "#3cb371", "#2e8b57"],
    "anthropic": ["#ffd699", "#ffb347", "#ff9933", "#e68a00", "#cc7a00"],
    "google": ["#b8d4e8", "#89cff0", "#5eb3e4", "#3399cc", "#267399"],
    "deepseek": ["#b8b8e8", "#9999d6", "#7a7ac4", "#5c5cb2", "#4b4ba0"],
    "xai": ["#606060", "#4a4a4a", "#3a3a3a", "#2a2a2a", "#1a1a1a"],
    "kimi": ["#d8b8e8", "#c99bd8", "#ba7ec8", "#ab61b8", "#9c44a8"],
    "meta": ["#b8e8d8", "#8fd9c4", "#66cab0", "#3dbb9c", "#2a9d7f"],
    "mistral": ["#f0b8b8", "#e89999", "#e07a7a", "#d85b5b", "#d03c3c"],
    "moondream": ["#ff6b6b", "#e74c3c", "#c0392b", "#a93226", "#8b0000"],
    "default": ["#d0d0d0", "#b0b0b0", "#909090", "#707070", "#505050"],
}


def get_provider(model_name: str) -> str:
    name_lower = model_name.lower()
    if any(x in name_lower for x in ["gpt", "o3", "o1", "openai"]):
        return "openai"
    if any(x in name_lower for x in ["claude", "anthropic"]):
        return "anthropic"
    if any(x in name_lower for x in ["gemini", "google"]):
        return "google"
    if any(x in name_lower for x in ["deepseek"]):
        return "deepseek"
    if any(x in name_lower for x in ["grok", "xai"]):
        return "xai"
    if any(x in name_lower for x in ["kimi", "moonshot"]):
        return "kimi"
    if any(x in name_lower for x in ["llama", "meta"]):
        return "meta"
    if any(x in name_lower for x in ["mistral", "mixtral"]):
        return "mistral"
    if any(x in name_lower for x in ["moondream"]):
        return "moondream"
    return "default"


def load_results(results_dir: str) -> dict[str, dict]:
    models = {}

    for f in Path(results_dir).glob("*_results.json"):
        with open(f) as fp:
            data = json.load(fp)

        model_name = f.stem.replace("_results", "")

        # Group scores by example, count failures as 0
        example_scores: dict[str, list[float]] = {}
        successful_scores: list[float] = []
        n_failures = 0
        for result in data.get("results", []):
            score = result.get("score")
            example = result.get("example_name", "")
            if score is not None:
                example_scores.setdefault(example, []).append(score)
                successful_scores.append(score)
            elif result.get("error"):
                n_failures += 1
                example_scores.setdefault(example, []).append(0.0)  # Failures count as 0

        all_scores = [s for scores in example_scores.values() for s in scores]
        samples_per_example = len(all_scores) / len(example_scores) if example_scores else 1

        if all_scores:
            n = len(all_scores)
            mean = np.mean(all_scores)

            # Proper SEM for benchmark score (seed-only uncertainty, fixed questions)
            # SE(μ̂) = sqrt((1/n²) * Σᵢ (sᵢ²/k))
            # where sᵢ² is within-question variance, k is samples per question, n is num questions
            if samples_per_example > 1:
                n_examples = len(example_scores)
                k = int(round(samples_per_example))
                # Compute within-question variances
                variances = [np.var(scores, ddof=1) for scores in example_scores.values() if len(scores) > 1]
                # SE formula from CLT over questions
                sem = np.sqrt(np.sum(variances) / k) / n_examples
                std = np.mean([np.sqrt(v) for v in variances])  # for display
            else:
                std = float('nan')
                sem = float('nan')

            # Calculate hypothetical scores if model had failures
            # (failures are already counted as 0 in all_scores)
            if n_failures > 0:
                successful_mean = np.mean(successful_scores) if successful_scores else 0
                # Score if failures were 100% instead of 0
                score_if_perfect = (sum(successful_scores) + n_failures * 1.0) / n
                # Score if failures matched avg of successful queries
                score_if_avg = (sum(successful_scores) + n_failures * successful_mean) / n
            else:
                score_if_perfect = None
                score_if_avg = None

            models[model_name] = {
                "mean": mean,
                "std": std,
                "sem": sem,
                "n": n,
                "n_failures": n_failures,
                "score_if_perfect": score_if_perfect,
                "score_if_avg": score_if_avg,
                "provider": get_provider(model_name),
            }

    return models


def assign_colors(sorted_models: list) -> list[str]:
    provider_indices: dict[str, int] = {}
    colors = []

    for _, stats in sorted_models:
        provider = stats["provider"]
        palette = COLOR_PALETTES.get(provider, COLOR_PALETTES["default"])
        idx = provider_indices.get(provider, 0)
        color = palette[idx % len(palette)]
        provider_indices[provider] = idx + 1
        colors.append(color)

    return colors


def plot_results(models: dict, theme: str, output_path: str):
    if theme == "dark":
        plt.style.use("dark_background")
        edge_color = "#ffffff"
        grid_color = "#404040"
        text_color = "#ffffff"
    else:
        plt.style.use("default")
        edge_color = "#333333"
        grid_color = "#cccccc"
        text_color = "#333333"

    sorted_models = sorted(models.items(), key=lambda x: x[1]["mean"], reverse=True)
    # Add star to names with failures
    names = [f"{m[0]} *" if m[1].get("n_failures", 0) > 0 else m[0] for m in sorted_models]
    means = [m[1]["mean"] for m in sorted_models]
    sems = [m[1]["sem"] for m in sorted_models]
    scores_if_avg = [m[1].get("score_if_avg") for m in sorted_models]
    colors = assign_colors(sorted_models)

    any_failures = any(m[1].get("n_failures", 0) > 0 for m in sorted_models)
    fig, ax = plt.subplots(figsize=(12, max(6, len(names) * 0.5 + (0.5 if any_failures else 0))))

    y_pos = np.arange(len(names))

    # Only show error bars if we have valid SEM values
    xerr = [s if not np.isnan(s) else 0 for s in sems]
    bars = ax.barh(y_pos, means, xerr=xerr if any(x > 0 for x in xerr) else None,
                   capsize=4, color=colors,
                   edgecolor=edge_color, linewidth=0.5, alpha=0.85,
                   error_kw={"ecolor": edge_color, "capthick": 1.5})

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Score", fontsize=12)
    ax.set_xlim(0, 1.05)
    ax.set_title("ScrupulousnessBench - Vision Trick Questions (mean ± SEM)", fontsize=14, fontweight="bold")

    for i, (bar, mean, sem) in enumerate(zip(bars, means, sems)):
        label = f"{mean:.1%}"
        x_pos = min(mean + sem + 0.02, 0.95)
        ax.text(x_pos, bar.get_y() + bar.get_height()/2, label,
                va="center", ha="left", fontsize=9, color=text_color)

    # Plot hypothetical scores for models with failures
    for i, avg in enumerate(scores_if_avg):
        y = y_pos[i]
        if avg is not None:
            ax.plot([avg, avg], [y - 0.3, y + 0.3], '-', color='#e74c3c', linewidth=2, zorder=5)

    ax.xaxis.grid(True, linestyle="--", alpha=0.5, color=grid_color)
    ax.set_axisbelow(True)

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    providers_in_plot = set(m[1]["provider"] for m in sorted_models)
    provider_labels = {
        "openai": "OpenAI", "anthropic": "Anthropic", "google": "Google",
        "deepseek": "DeepSeek", "xai": "xAI", "kimi": "Kimi",
        "meta": "Meta", "mistral": "Mistral", "moondream": "Moondream"
    }
    legend_elements = [
        Patch(facecolor=COLOR_PALETTES[p][1], edgecolor=edge_color, label=provider_labels.get(p, p))
        for p in ["openai", "anthropic", "google", "deepseek", "xai", "kimi", "meta", "mistral", "moondream"]
        if p in providers_in_plot
    ]

    # Add failure marker explanations to legend if any model had failures
    if any_failures:
        legend_elements.append(Line2D([0], [0], color='#e74c3c', linewidth=2,
                                      label='If failures = avg'))

    if legend_elements:
        ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    # Add footnote for starred models
    if any_failures:
        fig.text(0.5, 0.01, '* Model had API failures for some questions',
                ha='center', fontsize=9, style='italic', color='#666666')

    plt.tight_layout(rect=[0, 0.03 if any_failures else 0, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    default_results = Path(__file__).parent.parent / "results"
    default_output = Path(__file__).parent.parent / "outputs"
    results_dir = sys.argv[1] if len(sys.argv) > 1 else str(default_results)
    output_dir = sys.argv[2] if len(sys.argv) > 2 else str(default_output)

    if not os.path.isdir(results_dir):
        print(f"Error: {results_dir} not found")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    models = load_results(results_dir)

    if not models:
        print("No results found")
        sys.exit(1)

    print(f"Found {len(models)} models:")
    for name, stats in sorted(models.items(), key=lambda x: x[1]["mean"], reverse=True):
        sem_str = f"± {stats['sem']:.1%}" if stats['n'] > 1 else "± NaN"
        print(f"  {name}: {stats['mean']:.1%} {sem_str} (n={stats['n']})")

    plot_results(models, "light", os.path.join(output_dir, "scores_light.png"))
    plot_results(models, "dark", os.path.join(output_dir, "scores_dark.png"))


if __name__ == "__main__":
    main()
