import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Mapping, Sequence, Tuple

import medpy.metric as metric
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from seaborn import JointGrid
from vital.data.camus.config import CamusTags, Label
from vital.results.camus import CamusResultsProcessor
from vital.results.camus.utils.data_struct import InstantResult
from vital.results.camus.utils.itertools import PatientViewInstants
from vital.results.processor import ResultsProcessor

logger = logging.getLogger(__name__)


def _adjust_jointgrid_ylim(grid: JointGrid, ylim: Tuple[float, float]) -> JointGrid:
    """Workaround to fix only min/max bound on ylim in seaborn's `JointGrid`, since setting it with `init` is bugged."""
    grid_ylim = grid.ax_joint.get_ylim()
    ymin, ymax = ylim
    if ymin is not None and ymin > grid_ylim[0]:
        grid_ylim = ymin, grid_ylim[1]
    if ymax is not None and ymax < grid_ylim[1]:
        grid_ylim = grid_ylim[0], ymax
    grid.ax_joint.set_ylim(*grid_ylim)
    grid.ax_marg_y.set_ylim(*grid_ylim)
    return grid


class SegmentationMetricsPlots(ResultsProcessor):
    """Class that plots distributions of the segmentation metrics w.r.t. time."""

    desc = "scores_plots"
    ProcessingOutput = pd.DataFrame
    input_choices = [f"{CamusTags.pred}/{CamusTags.raw}", f"{CamusTags.pred}/{CamusTags.post}"]
    target_choices = [f"{CamusTags.gt}/{CamusTags.raw}"]
    ResultsCollection = PatientViewInstants
    scores = {"dsc": metric.dc}
    distances = {"hd": metric.hd, "assd": metric.assd}
    _metrics_limits = {"dsc": (None, 1), "hd": (0, None), "assd": (0, None)}
    _ylabel_mappings = {"dsc": "Dice score", "hd": "Hausdorff distance (in mm)", "assd": "ASSD (in mm)"}

    def __init__(self, inputs: Sequence[str], target: str, labels: Sequence[Label], **kwargs):
        """Initializes class instance.

        Args:
            inputs: Tag of the different inputs for which to compute metrics.
            target: Tag of the reference data to use as target when computing metrics.
            labels: Labels of the classes included in the segmentations.
            **kwargs: Additional parameters to pass along to ``super().__init__()``.
        """
        formatted_input = "_".join(input_tag.replace("/", "-") for input_tag in inputs)
        super().__init__(output_name=f"{formatted_input}_{self.desc}", **kwargs)
        if any(input_tag not in self.input_choices for input_tag in inputs):
            raise ValueError(
                f"The `input` values should be chosen from one of the supported values: {self.input_choices}. "
                f"You passed '{inputs}' as values for `inputs`."
            )
        if target not in self.target_choices:
            raise ValueError(
                f"The `target` parameter should be chosen from one of the supported values: {self.target_choices}. "
                f"You passed '{target}' as value for `target`."
            )
        self.input_tags = inputs
        self.target_tag = target

        # Compute scores on all labels, except background
        self.labels = {str(label): label.value for label in labels}
        self.labels.pop(str(Label.BG))

        # In the case of the myocardium (EPI) we want to calculate metrics for the entire epicardium
        # Therefore we concatenate ENDO (lumen) and EPI (myocardium)
        if Label.LV in labels and Label.MYO in labels:
            self.labels.pop(str(Label.MYO))
            self.labels["epi"] = (Label.LV.value, Label.MYO.value)

    def process_result(self, result: InstantResult) -> Tuple[str, ProcessingOutput]:
        """Computes metrics on data from an instant.

        Args:
            result: Data structure holding all the relevant information to compute the requested metrics for a single
                instant.

        Returns:
            - Identifier of the result for which the metrics where computed.
            - Dataframe containing the values of each metric for all the different inputs.
        """
        metrics = {}
        for input_tag in self.input_tags:

            pred, gt, voxelspacing = result[input_tag].data, result[self.target_tag].data, result.voxelspacing

            data_metrics = {}
            for label_tag, label in self.labels.items():
                pred_mask, gt_mask = np.isin(pred, label), np.isin(gt, label)

                # Compute the reconstruction accuracy metrics
                data_metrics.update(
                    {f"{label_tag}_{score}": score_fn(pred_mask, gt_mask) for score, score_fn in self.scores.items()}
                )

                # Compute the distance metrics (that require the images' voxelspacing)
                # only if the requested label is present in both result and reference
                if np.any(pred_mask) and np.any(gt_mask):
                    data_metrics.update(
                        {
                            f"{label_tag}_{dist}": dist_fn(pred_mask, gt_mask, voxelspacing=voxelspacing)
                            for dist, dist_fn in self.distances.items()
                        }
                    )
                # Otherwise mark distances as NaN for this item
                else:
                    data_metrics.update({f"{label_tag}_{distance}": np.NaN for distance in self.distances})

            metrics[input_tag] = data_metrics

        return result.id, pd.DataFrame.from_dict(metrics, orient="index")

    def aggregate_outputs(self, outputs: Mapping[str, ProcessingOutput], output_path: Path) -> None:
        """Collects the metrics computed on all the results, and plots their distributions w.r.t. time.

        Args:
            outputs: Mapping between each result in the results collection and their metrics.
            output_path: Root path where to save the plots.
        """
        # 1. Build dataframe containing all the metrics
        metrics_data = pd.concat(outputs).rename_axis(["id", "data"])
        # 1.1. Build multiindex by splitting the IDs into their hierachical levels
        # (w/ each level from the original IDs cast to its appropriate type)
        id_idx = metrics_data.index.levels[0].str.split("/", expand=True).set_names(["patient", "view", "frame"])
        id_idx = id_idx.set_levels(id_idx.levels[-1].astype(int), level="frame")
        # 1.2 Switch ID index level for multiindex based on ID
        num_data = metrics_data.index.get_level_values("data").unique().size
        metrics_data = metrics_data.set_index(id_idx.repeat(num_data), append=True)
        metrics_data = metrics_data.droplevel("id")
        # 1.3 Compute normalized time from index + sequence lengths
        metrics_data["time"] = metrics_data.index.get_level_values("frame")
        sequence_lengths = metrics_data.groupby(by=["patient", "view", "data"]).size()
        metrics_data["time"] /= sequence_lengths - 1

        # 2. Prepare the dataframe for easy plotting w/ seaborn
        # 2.1 Make 'data' a column, rather than an index
        metrics_data = metrics_data.reset_index(level="data")
        # 2.2 Convert the dataframe to long format, for easy plotting with seaborn
        metric_names = metrics_data.columns.difference(["time", "data"])
        metrics_data = metrics_data.melt(
            id_vars=["time", "data"], value_vars=metric_names, var_name="metric", value_name="val"
        )
        # 2.3 Split metric tags into class name (capitalized) and metric name
        metrics_data[["class", "metric"]] = metrics_data["metric"].str.split("_", expand=True)
        metrics_data["class"] = metrics_data["class"].str.upper()

        self._plot_aggregated_metrics(metrics_data, output_path)

    def _plot_aggregated_metrics(self, metrics_data: pd.DataFrame, output_path: Path) -> None:
        """Plots histogram of the metrics w.r.t. time, w/ and w/o y-marginal distributions.

        Args:
            metrics_data: Dataframe of the metrics' data, in long format.
            output_path: Root path where to save the plots.
        """
        # Ensure that matplotlib is using 'agg' backend
        # to avoid possible 'Could not connect to any X display' errors
        # when no X server is available, e.g. in remote terminal
        plt.switch_backend("agg")

        # Plot each score w.r.t. time
        for score in metrics_data.metric.unique():
            logger.info(f"Saving '{score}' plots to '{output_path}'... ")
            score_label = self._ylabel_mappings[score]

            # Distribution plots, w/ 1 figure by metric, 1 column by class
            with sns.axes_style("darkgrid"):
                g = sns.displot(
                    data=metrics_data[metrics_data.metric == score], x="time", y="val", hue="data", col="class"
                )
                g.set_axis_labels(y_var=score_label)
            plt.savefig(output_path / (score + "_dis.png"))
            plt.close()  # Close the figure to avoid contamination between plots

            # Joint plots (w/ y marginal distributions), w/ multiple figures by metric, 1 figure by class
            for seg_class in metrics_data["class"].unique():
                with sns.axes_style("darkgrid"):
                    g = sns.JointGrid(
                        data=metrics_data[(metrics_data.metric == score) & (metrics_data["class"] == seg_class)],
                        x="time",
                        y="val",
                        hue="data",
                        xlim=(0, 1),
                    )
                    g.plot_joint(sns.histplot)
                    g.plot_marginals(sns.kdeplot, clip=self._metrics_limits[score])
                    g.ax_joint.set_ylabel(score_label)
                    g = _adjust_jointgrid_ylim(g, self._metrics_limits[score])
                    g.ax_marg_x.remove()
                plt.savefig(output_path / (score + f"_joint_{seg_class.lower()}.png"), bbox_inches="tight")
                plt.close()  # Close the figure to avoid contamination between plots

    @classmethod
    def build_parser(cls) -> ArgumentParser:
        """Creates parser for scores plot processor.

        Returns:
            Parser object for scores plot processor.
        """
        parser = super().build_parser()
        parser.add_argument(
            "--inputs",
            type=str,
            nargs="+",
            default=cls.input_choices,
            choices=cls.input_choices,
            help="Data for which to analyze scores' evolution w.r.t. time",
        )
        parser.add_argument(
            "--target",
            type=str,
            default=cls.target_choices[0],
            choices=cls.target_choices,
            help="Reference data to use as target when computing scores",
        )
        parser = CamusResultsProcessor.add_labels_args(parser)
        return parser


def main():
    """Run the script."""
    SegmentationMetricsPlots.main()


if __name__ == "__main__":
    main()
