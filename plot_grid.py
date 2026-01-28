#!/usr/bin/env python3
"""
Generate a heatmap grid showing model vs question scores.
Helps identify poorly specified questions or model weaknesses.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("outputs")


def load_all_results():
    """Load all results and organize by model and question."""
    data = {}  # model -> question -> list of scores
    has_failures = {}  # model -> bool

    for f in sorted(RESULTS_DIR.glob("*_results.json")):
        model = f.stem.replace("_results", "")
        with open(f) as fp:
            results = json.load(fp)

        data[model] = {}
        has_failures[model] = False
        for r in results.get("results", []):
            q = r.get("example_name", "")
            score = r.get("score")
            if score is not None:
                data[model].setdefault(q, []).append(score)
            elif r.get("error"):
                has_failures[model] = True

    return data, has_failures


def build_matrix(data):
    """Build score matrix and collect question/model names."""
    models = sorted(data.keys())
    questions = sorted(set(q for m in data.values() for q in m.keys()))

    matrix = np.full((len(models), len(questions)), np.nan)

    for i, model in enumerate(models):
        for j, question in enumerate(questions):
            scores = data[model].get(question, [])
            if scores:
                matrix[i, j] = np.mean(scores)

    return matrix, models, questions


def plot_grid(matrix, models, questions, output_path, has_failures):
    """Create heatmap visualization."""
    # Check which models have NaN values in matrix
    has_nans = [np.any(np.isnan(matrix[i, :])) for i in range(len(models))]

    # Build display names with stars
    display_names = []
    for i, model in enumerate(models):
        if has_failures.get(model, False) or has_nans[i]:
            display_names.append(f"{model} *")
        else:
            display_names.append(model)

    any_starred = any(has_failures.get(m, False) or has_nans[i] for i, m in enumerate(models))

    fig, ax = plt.subplots(figsize=(14, max(8, len(models) * 0.5 + (1 if any_starred else 0))))

    # Custom colormap: red (0) -> yellow (0.5) -> green (1)
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['#d73027', '#fee08b', '#1a9850']
    cmap = LinearSegmentedColormap.from_list('score', colors)

    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)

    # Labels
    ax.set_xticks(np.arange(len(questions)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(questions, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(display_names, fontsize=9)

    # Add score text in cells
    for i in range(len(models)):
        for j in range(len(questions)):
            val = matrix[i, j]
            if not np.isnan(val):
                text_color = 'white' if val < 0.4 or val > 0.7 else 'black'
                ax.text(j, i, f'{val:.0%}', ha='center', va='center',
                       fontsize=7, color=text_color)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Score', fontsize=10)

    # Column averages (question difficulty)
    q_avgs = np.nanmean(matrix, axis=0)
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(np.arange(len(questions)))
    ax2.set_xticklabels([f'{a:.0%}' for a in q_avgs], fontsize=8, color='blue')
    ax2.tick_params(axis='x', colors='blue')

    # Row averages (model performance)
    m_avgs = np.nanmean(matrix, axis=1)
    for i, avg in enumerate(m_avgs):
        ax.text(len(questions) + 0.3, i, f'{avg:.0%}', va='center', fontsize=9,
               fontweight='bold', color='#1a9850' if avg > 0.5 else '#d73027')

    ax.set_title('ScrupulousnessBench: Model × Question Scores\n(top: question avg, right: model avg)',
                fontsize=12, fontweight='bold')

    # Add footnote for starred models
    if any_starred:
        fig.text(0.5, 0.01, '* Model had API failures or missing data for some questions',
                ha='center', fontsize=9, style='italic', color='#666666')

    plt.tight_layout(rect=[0, 0.03 if any_starred else 0, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def analyze_questions(data, matrix, models, questions):
    """Analyze questions for potential issues."""
    print("\n" + "="*60)
    print("QUESTION ANALYSIS")
    print("="*60)

    q_avgs = np.nanmean(matrix, axis=0)
    q_stds = np.nanstd(matrix, axis=0)

    # Questions everyone fails (too hard or misspecified?)
    print("\n🔴 Questions with <20% avg (possibly misspecified):")
    for j, q in enumerate(questions):
        if q_avgs[j] < 0.2:
            print(f"  {q}: {q_avgs[j]:.0%} avg")

    # Questions everyone aces (too easy?)
    print("\n🟢 Questions with >80% avg (possibly too easy):")
    for j, q in enumerate(questions):
        if q_avgs[j] > 0.8:
            print(f"  {q}: {q_avgs[j]:.0%} avg")

    # High variance questions (inconsistent scoring?)
    print("\n🟡 Questions with high variance (>0.3 std):")
    for j, q in enumerate(questions):
        if q_stds[j] > 0.3:
            print(f"  {q}: {q_avgs[j]:.0%} avg, {q_stds[j]:.2f} std")

    return q_avgs, q_stds


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    data, has_failures = load_all_results()
    if not data:
        print("No results found")
        return

    print(f"Loaded {len(data)} models")

    matrix, models, questions = build_matrix(data)

    plot_grid(matrix, models, questions, OUTPUT_DIR / "score_grid.png", has_failures)

    analyze_questions(data, matrix, models, questions)

    # Print per-question breakdown
    print("\n" + "="*60)
    print("PER-QUESTION SCORES")
    print("="*60)
    q_avgs = np.nanmean(matrix, axis=0)
    for j, q in enumerate(sorted(zip(q_avgs, questions))):
        avg, name = q
        print(f"  {avg:5.0%}  {name}")


if __name__ == "__main__":
    main()
