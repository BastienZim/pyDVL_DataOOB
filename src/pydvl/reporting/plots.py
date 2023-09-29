from typing import Any, List, Optional, OrderedDict, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
from matplotlib.axes import Axes
from numpy.typing import NDArray
from scipy.stats import norm
from pydvl.reporting.scores import (
    compute_removal_score,
    compute_removal_class_imbal_evol,
)
from pydvl.value.result import ValuationResult
import seaborn as sns


def compute_best_worst_scores(
    utility,
    utility_eval,
    removal_percentages,
    n_iter,
    func,
    kwargs_list,
    f_name="oob",
    return_values=False,
):
    all_best_scores = []
    all_worst_scores = []
    if return_values:
        all_values = {str(list(k.values())[0]): [] for k in kwargs_list}
    for i in range(n_iter):
        for kwargs in kwargs_list:
            # print(kwargs)
            kwarg_val = list(kwargs.values())[0]
            method_name = f"{f_name}_{kwarg_val}"
            values = func(utility, **kwargs)

            best_scores = compute_removal_score(
                u=utility_eval,
                values=values,
                percentages=removal_percentages,
                remove_best=True,
            )
            best_scores["method_name"] = method_name
            all_best_scores.append(best_scores)

            worst_scores = compute_removal_score(
                u=utility_eval,
                values=values,
                percentages=removal_percentages,
                remove_best=False,
            )
            worst_scores["method_name"] = method_name
            all_worst_scores.append(worst_scores)
            if return_values:
                all_values[str(kwarg_val)].append(values)

    best_scores_df = pd.DataFrame(all_best_scores)
    worst_scores_df = pd.DataFrame(all_worst_scores)
    if return_values:
        return (best_scores_df, worst_scores_df, all_values)
    else:
        return (best_scores_df, worst_scores_df)


def plot_best_worst(best_scores_df, worst_scores_df, palette_name="tab10", colors=None):
    _, ax = plt.subplots(nrows=1, ncols=2, figsize=[20, 5])
    if colors is None:
        colors = sns.color_palette(
            palette_name,
            n_colors=best_scores_df.groupby("method_name").count().iloc[:, [0]].size,
        )
    removal_percentages = [
        float(str(x)[:5]) for x in best_scores_df.columns.values if x != "method_name"
    ]
    method_header = "".join(best_scores_df.method_name.iloc[0].split("_")[:1])
    all_params = set(
        [
            "_".join(x.split("_")[1:])
            for x in best_scores_df.loc[:, "method_name"].values
        ]
    )
    if all([x.replace(".", "").isnumeric() for x in all_params]):
        all_params = sorted(all_params, key=lambda x: float(x))
    for i, param in enumerate(all_params):
        method_name = f"{method_header}_{param}"
        shaded_mean_std(
            best_scores_df[best_scores_df["method_name"] == method_name].drop(
                columns=["method_name"]
            ),
            abscissa=removal_percentages,
            mean_color=colors[i % len(colors)],
            shade_color=colors[i % len(colors)],
            xlabel="Percentage removed",
            # ylabel=utility.scorer._name.capitalize(),
            label=method_name.replace("_", ""),
            title="Accuracy as a function of percentage of removed best data points\nThe Lower the Better",
            ax=ax[1],
        )
        shaded_mean_std(
            worst_scores_df[worst_scores_df["method_name"] == method_name].drop(
                columns=["method_name"]
            ),
            abscissa=removal_percentages,
            mean_color=colors[i % len(colors)],
            shade_color=colors[i % len(colors)],
            xlabel="Percentage removed",
            # ylabel=utility.scorer._name.capitalize(),
            label=method_name.replace("_", ""),
            title="Accuracy as a function of percentage of removed worst data points\nThe Higher the Better",
            ax=ax[0],
        )
    ax[0].legend()
    ax[1].legend()
    plt.show()


def plot_best_worst_class_imbalance(
    best_scores_df,
    worst_scores_df,
    all_values,
    ut,
    palette_name="tab10",
    colors=None,
    random_run=False,
):
    _, ax = plt.subplots(nrows=2, ncols=2, figsize=[20, 10])
    if colors is None:
        colors = sns.color_palette(
            palette_name,
            n_colors=best_scores_df.groupby("method_name").count().iloc[:, [0]].size,
        )
    removal_percentages = [
        float(str(x)[:5]) for x in best_scores_df.columns.values if x != "method_name"
    ]
    method_header = "".join(best_scores_df.method_name.iloc[0].split("_")[:1])
    all_params = set(
        [
            "_".join(x.split("_")[1:])
            for x in best_scores_df.loc[:, "method_name"].values
        ]
    )
    if all([x.replace(".", "").isnumeric() for x in all_params]):
        all_params = sorted(all_params, key=lambda x: float(x))
    for i, param in enumerate(all_params):
        method_name = f"{method_header}_{param}"
        shaded_mean_std(
            best_scores_df[best_scores_df["method_name"] == method_name].drop(
                columns=["method_name"]
            ),
            abscissa=removal_percentages,
            mean_color=colors[i % len(colors)],
            shade_color=colors[i % len(colors)],
            xlabel="Percentage removed",
            # ylabel=ut.scorer._name.capitalize(),
            label=method_name.replace("_", ""),
            title="Accuracy as a function of percentage of removed best data points\nThe Lower the Better",
            ax=ax[0, 1],
        )
        shaded_mean_std(
            worst_scores_df[worst_scores_df["method_name"] == method_name].drop(
                columns=["method_name"]
            ),
            abscissa=removal_percentages,
            mean_color=colors[i % len(colors)],
            shade_color=colors[i % len(colors)],
            xlabel="Percentage removed",
            # ylabel=ut.scorer._name.capitalize(),
            label=method_name.replace("_", ""),
            title="Accuracy as a function of percentage of removed worst data points\nThe Higher the Better",
            ax=ax[0, 0],
        )

        for k in range(2):
            is_best = ["WORST", "BEST"][k]
            df_imbal_vals = pd.DataFrame(index=removal_percentages)
            for j, vals in enumerate(all_values[param]):
                imbal_res = compute_removal_class_imbal_evol(
                    u=ut,
                    values=vals,
                    percentages=removal_percentages,
                    remove_best=[False, True][k],
                )
                df_imbal_vals.loc[:, j] = imbal_res.min(axis=1) / imbal_res.max(axis=1)

            df_imbal_vals = df_imbal_vals.dropna().T
            shaded_mean_std(
                df_imbal_vals,
                abscissa=df_imbal_vals.columns,
                mean_color=colors[i % len(colors)],
                shade_color=colors[i % len(colors)],
                xlabel="Percentage removed",
                label=method_name.replace("_", ""),
                title=f"Remove {is_best}\nEvolution of class imbalance (min class count/max class count)",
                ax=ax[1, k],
            )
    if(random_run):
        n_iter=best_scores_df.method_name.value_counts()[0]
        random_best = []
        random_worst = []
        random_vals = []
        for i in range(n_iter):
            vals = ValuationResult.from_random(size=len(ut.data))
            random_vals.append(vals)
            best_score  = compute_removal_score(
                                                    u=ut,
                                                    values=vals,
                                                    percentages=removal_percentages,
                                                    remove_best=True,
                                                )
            random_best.append(best_score)
            worst_score  = compute_removal_score(
                                                    u=ut,
                                                    values=vals,
                                                    percentages=removal_percentages,
                                                    remove_best=False,
                                                )
            random_worst.append(worst_score)

        shaded_mean_std(
            pd.DataFrame(random_best),
            abscissa=removal_percentages,
            mean_color="black",
            shade_color="black",
            xlabel="Percentage removed",
            label="Random",
            title="Accuracy as a function of percentage of removed best data points\nThe Lower the Better",
            ax=ax[0, 1],
        )
        shaded_mean_std(
            pd.DataFrame(random_worst),
            abscissa=removal_percentages,
            mean_color="black",
            shade_color="black",
            xlabel="Percentage removed",
            label="Random",
            title="Accuracy as a function of percentage of removed worst data points\nThe Higher the Better",
            ax=ax[0, 0],
        )
        for k in range(2):
            is_best = ["WORST", "BEST"][k]
            df_imbal_vals = pd.DataFrame(index=removal_percentages)
            for j,vals in enumerate(random_vals):
                imbal_res = compute_removal_class_imbal_evol(
                    u=ut,
                    values=vals,
                    percentages=removal_percentages,
                    remove_best=[False, True][k],
                )
                df_imbal_vals.loc[:, j] = imbal_res.min(axis=1) / imbal_res.max(axis=1)

            df_imbal_vals = df_imbal_vals.dropna().T
            shaded_mean_std(
                df_imbal_vals,
                abscissa=df_imbal_vals.columns,
                mean_color="black",
                shade_color="black",
                xlabel="Percentage removed",
                label="Random",
                title=f"Remove {is_best}\nEvolution of class imbalance (min class count/max class count)",
                ax=ax[1, k],
            )
        



    ax[0, 0].legend()
    ax[0, 1].legend()
    ax[1, 0].legend()
    ax[1, 1].legend()
    ax[1, 0].set_ylim(bottom=0, top=1)
    ax[1, 1].set_ylim(bottom=0, top=1)
    plt.subplots_adjust(hspace=0.3)
    plt.show()


def plot_methods_linreg(best_df, worst_df):
    fig, ax = plt.subplots(
        ncols=2, nrows=best_df.method_name.nunique(), figsize=[15, 10]
    )
    for i, x in enumerate(best_df.method_name.unique()):
        p = sns.regplot(
            x=np.arange(len(best_df.groupby("method_name").mean().values[i])),
            y=best_df.groupby("method_name").mean().loc[x].values,
            ax=ax[i, 1],
        )
        y_data = p.get_lines()[0].get_ydata()
        maxi = best_df.groupby("method_name").mean().loc[x].max()
        ax[i, 1].set_title(f"{x}_{y_data[-1]-y_data[0]:.3f} max:{maxi:.3f}")
        ax[i, 1].set_xticks([])
        p = sns.regplot(
            x=np.arange(len(worst_df.groupby("method_name").mean().values[i])),
            y=worst_df.groupby("method_name").mean().loc[x].values,
            ax=ax[i, 0],
        )
        y_data = p.get_lines()[0].get_ydata()
        maxi = worst_df.groupby("method_name").mean().loc[x].max()
        ax[i, 0].set_title(f"{x}_{y_data[-1]-y_data[0]:.3f} max:{maxi:.3f}")
        ax[i, 0].set_xticks([])
    plt.subplots_adjust(hspace=0.5)
    plt.show()


def shaded_mean_std(
    data: np.ndarray,
    abscissa: Optional[Sequence[Any]] = None,
    num_std: float = 1.0,
    mean_color: Optional[str] = "dodgerblue",
    shade_color: Optional[str] = "lightblue",
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    ax: Optional[Axes] = None,
    **kwargs,
) -> Axes:
    """The usual mean \(\pm\) std deviation plot to aggregate runs of experiments.

    Args:
        data: axis 0 is to be aggregated on (e.g. runs) and axis 1 is the
            data for each run.
        abscissa: values for the x-axis. Leave empty to use increasing integers.
        num_std: number of standard deviations to shade around the mean.
        mean_color: color for the mean
        shade_color: color for the shaded region
        title: Title text. To use mathematics, use LaTeX notation.
        xlabel: Text for the horizontal axis.
        ylabel: Text for the vertical axis
        ax: If passed, axes object into which to insert the figure. Otherwise,
            a new figure is created and returned
        kwargs: these are forwarded to the ax.plot() call for the mean.

    Returns:
        The axes used (or created)
    """
    assert len(data.shape) == 2
    mean = data.mean(axis=0)
    std = num_std * data.std(axis=0)

    if ax is None:
        fig, ax = plt.subplots()
    if abscissa is None:
        abscissa = list(range(data.shape[1]))

    ax.fill_between(abscissa, mean - std, mean + std, alpha=0.3, color=shade_color)
    ax.plot(abscissa, mean, color=mean_color, **kwargs)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return ax


def spearman_correlation(vv: List[OrderedDict], num_values: int, pvalue: float):
    """Simple matrix plots with spearman correlation for each pair in vv.

    Args:
        vv: list of OrderedDicts with index: value. Spearman correlation
            is computed for the keys.
        num_values: Use only these many values from the data (from the start
            of the OrderedDicts)
        pvalue: correlation coefficients for which the p-value is below the
            threshold `pvalue/len(vv)` will be discarded.
    """
    r: np.ndarray = np.ndarray((len(vv), len(vv)))
    p: np.ndarray = np.ndarray((len(vv), len(vv)))
    for i, a in enumerate(vv):
        for j, b in enumerate(vv):
            from scipy.stats._stats_py import SpearmanrResult

            spearman: SpearmanrResult = sp.stats.spearmanr(
                list(a.keys())[:num_values], list(b.keys())[:num_values]
            )
            r[i][j] = (
                spearman.correlation if spearman.pvalue < pvalue / len(vv) else np.nan
            )  # Bonferroni correction
            p[i][j] = spearman.pvalue
    fig, axs = plt.subplots(1, 2, figsize=(16, 7))
    plot1 = axs[0].matshow(r, vmin=-1, vmax=1)
    axs[0].set_title(f"Spearman correlation (top {num_values} values)")
    axs[0].set_xlabel("Runs")
    axs[0].set_ylabel("Runs")
    fig.colorbar(plot1, ax=axs[0])
    plot2 = axs[1].matshow(p, vmin=0, vmax=1)
    axs[1].set_title("p-value")
    axs[1].set_xlabel("Runs")
    axs[1].set_ylabel("Runs")
    fig.colorbar(plot2, ax=axs[1])

    return fig


def plot_shapley(
    df: pd.DataFrame,
    *,
    level: float = 0.05,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
) -> plt.Axes:
    r"""Plots the shapley values, as returned from
    [compute_shapley_values][pydvl.value.shapley.common.compute_shapley_values], with error bars
    corresponding to an $\alpha$-level confidence interval.

    Args:
        df: dataframe with the shapley values
        level: confidence level for the error bars
        ax: axes to plot on or None if a new subplots should be created
        title: string, title of the plot
        xlabel: string, x label of the plot
        ylabel: string, y label of the plot

    Returns:
        The axes created or used
    """
    if ax is None:
        _, ax = plt.subplots()

    yerr = norm.ppf(1 - level / 2) * df["data_value_stderr"]

    ax.errorbar(x=df.index, y=df["data_value"], yerr=yerr, fmt="o", capsize=6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.xticks(rotation=60)
    return ax


def plot_influence_distribution_by_label(
    influences: NDArray[np.float_], labels: NDArray[np.float_], title_extra: str = ""
):
    """Plots the histogram of the influence that all samples in the training set
    have over a single sample index, separated by labels.

    Args:
       influences: array of influences (training samples x test samples)
       labels: labels for the training set.
       title_extra:
    """
    _, ax = plt.subplots()
    unique_labels = np.unique(labels)
    for label in unique_labels:
        ax.hist(influences[labels == label], label=label, alpha=0.7)
    ax.set_xlabel("Influence values")
    ax.set_ylabel("Number of samples")
    ax.set_title(f"Distribution of influences " + title_extra)
    ax.legend()
    plt.show()
