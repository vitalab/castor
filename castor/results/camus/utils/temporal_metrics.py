from argparse import ArgumentParser
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from vital.metrics.camus.temporal.config import attributes_bounds as vital_attributes_bounds
from vital.metrics.camus.temporal.config import thresholds
from vital.metrics.camus.temporal.utils import compute_temporal_consistency_metric
from vital.results.camus.utils.data_struct import ViewResult
from vital.results.camus.utils.itertools import PatientViews
from vital.results.metrics import Metrics
from vital.utils.parsing import StoreDictKeyPair

from castor.metrics.camus.temporal.config import attributes_bounds as castor_attributes_bounds

attributes_bounds = {**vital_attributes_bounds, **castor_attributes_bounds}
"""Merge generic attributes bounds provided by `vital` to model-specific bounds provided by `castor`."""


class TemporalMetrics(Metrics):
    """Class that computes temporal coherence metrics on sequences of attributes' values."""

    desc = "temporal_metrics"
    ResultsCollection = PatientViews
    normalization_cfg_choices: Sequence[str]

    def __init__(
        self,
        normalization_cfg: str = None,
        measure_thresholds: bool = False,
        threshold_margins: Mapping[str, float] = None,
        inconsistent_frames_only: bool = False,
        **kwargs,
    ):
        """Initializes class instance.

        Args:
            normalization_cfg: Tag indicating which set of statistics to use to normalize the attribute values, from
                sets of statistics computed across different domains (e.g. ground truths, AR-VAE encodings, etc.)
            measure_thresholds: Enable the computation of thresholds that should be used for each metric, instead of
                the computation of metrics themselves. Should typically only be used when `input` refers to a
                ground truth (or its derivative).
            threshold_margins: Margin to add to the thresholds for temporal inconsistencies, specific to each attribute.
                Follows the eq. `thresh = thresh * (1 + margin)`.
            inconsistent_frames_only: For metrics apart from the proportion of inconsistent frames, only compute these
                metrics on the inconsistent frames. Only when `measure_thresholds` is `False`.
            **kwargs: Additional parameters to pass along to ``super().__init__()``.
        """
        super().__init__(**kwargs)
        if measure_thresholds and threshold_margins:
            raise ValueError("You should only use one of `measure_thresholds` or `threshold_margins` at a time.")
        if normalization_cfg is None:
            # If `normalization_cfg` was not explicitly provided, default to the first choices in the list
            normalization_cfg = self.normalization_cfg_choices[0]
        self._attrs_bounds = attributes_bounds[normalization_cfg]
        self._measure_thresholds = measure_thresholds
        self._threshold_margins = threshold_margins
        if self._threshold_margins is None:
            self._threshold_margins = {}
        self._inconsistent_frames_only = inconsistent_frames_only

    def process_result(self, result: ViewResult) -> Optional[Tuple[str, "TemporalMetrics.ProcessingOutput"]]:
        """Computes temporal coherence metrics on sequences of attributes' values.

        Args:
            result: Data structure holding all the sequence`s data.

        Returns:
            - Identifier of the result for which the metrics where computed.
            - Mapping between the metrics and their value for the sequence.
        """
        attrs = self._extract_attributes_from_result(result, self.input_tag)
        metrics = {}
        inconsistent_frames_by_attr, err_thresh_ratios_by_attr = {}, {}
        for attr, attr_vals in attrs.items():
            temporal_consistency_abs_err = np.abs(
                compute_temporal_consistency_metric(attr_vals, bounds=self._attrs_bounds[attr])
            )
            if self._measure_thresholds:
                metrics[f"{attr}_neigh_inter_diff"] = np.max(temporal_consistency_abs_err)
            else:
                threshold = thresholds[attr] * (1 + self._threshold_margins.get(attr, 0))
                # Identifies the frames that are temporally inconsistent
                inconsistent_frames_by_attr[attr] = temporal_consistency_abs_err > threshold
                # Computes the ratios between the error and the tolerated threshold
                err_thresh_ratios = temporal_consistency_abs_err / threshold
                if self._inconsistent_frames_only:
                    err_thresh_ratios = err_thresh_ratios[temporal_consistency_abs_err > threshold]
                err_thresh_ratios_by_attr[f"{attr}_error_to_threshold_ratio"] = err_thresh_ratios

        if not self._measure_thresholds:
            # Computes the ratio of frames that are temporally inconsistent, for each of the measured attributes
            metrics.update(
                {
                    f"{attr}_inconsistent_frames_ratio": inconsistent_frames.mean()
                    for attr, inconsistent_frames in inconsistent_frames_by_attr.items()
                }
            )
            # Identifies the frames that are temporally inconsistent w.r.t. any of the measured attributes
            metrics["temporally_inconsistent_frames_ratio"] = (
                np.array(list(inconsistent_frames_by_attr.values())).any(axis=0).mean()
            )
            # Compute the ratio of error to threshold for inconsistent frames, for each of the measured attributes
            metrics.update(
                {attr: err_thresh_ratios.mean() for attr, err_thresh_ratios in err_thresh_ratios_by_attr.items()}
            )
            # Compute the ratio of error to threshold for inconsistent frames, across all the measured attributes
            metrics["error_to_threshold_ratio"] = np.hstack(list(err_thresh_ratios_by_attr.values())).mean()
            # Compute whether the sequence as any temporal inconsistencies, between any instants and for any attributes
            metrics["temporal_consistency_errors"] = bool(metrics["temporally_inconsistent_frames_ratio"])

        return result.id, metrics

    def _aggregate_metrics(self, metrics: pd.DataFrame) -> pd.DataFrame:
        """Computes global statistics for the metrics computed over each result.

        If `measure_thresholds == True`, then the metrics are aggregated differently, since we're more interested in the
        limit cases to determine the thresholds than in the general case.

        Args:
            metrics: Metrics computed over each result.

        Returns:
            Global statistics on the metrics computed over each result.
        """
        if self._measure_thresholds:
            return metrics.agg(["max", "min", "mean", "std"])
        else:
            # Define groups of columns that have to be aggregated differently
            frames_ratio_cols = metrics.columns[metrics.columns.str.contains("frames_ratio")]
            err_thresh_ratio_cols = metrics.columns[metrics.columns.str.contains("error_to_threshold_ratio")]
            temporally_inconsistent_indices = metrics["temporal_consistency_errors"]

            # Aggregate metrics with results reported on all sequences
            frames_ratio_agg_metrics = metrics.loc[temporally_inconsistent_indices, frames_ratio_cols].agg(
                ["mean", "std", "max", "min"]
            )

            # Aggregate metrics with results reported only on sequences w/ temporal inconsistencies
            err_thresh_ratio_agg_metrics = metrics[err_thresh_ratio_cols].agg(["mean", "std", "max", "min"])
            temporal_consistency_agg_metrics = metrics[["temporal_consistency_errors"]].agg(["sum"])

            # Merge aggregations, filling the join with w/ NaNs
            return pd.concat(
                [frames_ratio_agg_metrics, err_thresh_ratio_agg_metrics, temporal_consistency_agg_metrics],
                axis="columns",
            )

    @classmethod
    def build_parser(cls) -> ArgumentParser:
        """Creates parser for temporal metrics processor.

        Returns:
            Parser object for temporal metrics processor.
        """
        parser = super().build_parser()

        threshold_args = parser.add_mutually_exclusive_group()
        threshold_args.add_argument(
            "--measure_thresholds",
            action="store_true",
            help="Enable the computation of thresholds that should be used for each metric, instead of the "
            "computation of metrics themselves. Should typically only be used when `input` refers to a groundtruth "
            "(or its derivative).",
        )
        threshold_args.add_argument(
            "--threshold_margins",
            action=StoreDictKeyPair,
            metavar="THRESH1=MGN1,THRESH2=MGN2...",
            help="Margin to add to the thresholds for temporal inconsistencies, following the eq. "
            "`thresh = thresh * (1 + margin)`",
        )

        parser.add_argument(
            "--inconsistent_frames_only",
            action="store_true",
            help="For metrics apart from the proportion of inconsistent frames, only compute these metrics on the "
            "inconsistent frames",
        )
        parser.add_argument(
            "--normalization_cfg",
            type=str,
            choices=cls.normalization_cfg_choices,
            default=cls.normalization_cfg_choices[0],
            help="Tag indicating which set of statistics to use to normalize the attribute values, from sets of "
            "statistics computed across different domains (e.g. ground truths, AR-VAE encodings, etc.)",
        )
        return parser
