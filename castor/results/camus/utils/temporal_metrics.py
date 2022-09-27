from argparse import ArgumentParser
from pathlib import Path
from typing import Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from vital import get_vital_root
from vital.metrics.evaluate.attribute import compute_temporal_consistency_metric
from vital.results.camus.utils.data_struct import ViewResult
from vital.results.camus.utils.itertools import PatientViews
from vital.results.metrics import Metrics
from vital.utils.parsing import StoreDictKeyPair


class TemporalMetrics(Metrics):
    """Class that computes temporal coherence metrics on sequences of attributes' values."""

    desc = "temporal_metrics"
    ResultsCollection = PatientViews
    default_attribute_statistics_cfg: Path

    def __init__(
        self,
        attribute_statistics_cfg: Union[str, Path],
        thresholds_cfg: Union[str, Path] = None,
        threshold_margins: Mapping[str, float] = None,
        inconsistent_frames_only: bool = False,
        **kwargs,
    ):
        """Initializes class instance.

        Args:
            attribute_statistics_cfg: File containing pre-computed statistics for each attribute, used to normalize
                their values.
            thresholds_cfg: File containing pre-computed thresholds on the acceptable temporal consistency metrics'
                values for each attribute.
            measure_thresholds: Enable the computation of thresholds that should be used for each metric, instead of
                the computation of metrics themselves. Should typically only be used when `input` refers to a
                ground truth (or its derivative).
            threshold_margins: Margin to add to the thresholds for temporal inconsistencies, specific to each attribute.
                Follows the eq. `thresh = thresh * (1 + margin)`. Should only be set when `thresholds_cfg` is provided.
            inconsistent_frames_only: For metrics apart from the proportion of inconsistent frames, only compute these
                metrics on the inconsistent frames. Only when `measure_thresholds` is `False`.
            **kwargs: Additional parameters to pass along to ``super().__init__()``.
        """
        super().__init__(**kwargs)
        if (thresholds_cfg is None) and threshold_margins:
            raise ValueError("You should not set `threshold_margins` if you do not provide `thresholds` as well.")

        # Load statistics on the attributes' distributions thresholds from the config files
        with open(attribute_statistics_cfg) as f:
            self._attrs_bounds = yaml.safe_load(f)
        self._thresholds = None
        if thresholds_cfg:
            with open(thresholds_cfg) as f:
                self._thresholds = yaml.safe_load(f)

        self._threshold_margins = threshold_margins if threshold_margins else {}
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
            if self._thresholds:
                threshold = self._thresholds[attr] * (1 + self._threshold_margins.get(attr, 0))
                # Identifies the frames that are temporally inconsistent
                inconsistent_frames_by_attr[attr] = temporal_consistency_abs_err > threshold
                # Computes the ratios between the error and the tolerated threshold
                err_thresh_ratios = temporal_consistency_abs_err / threshold
                if self._inconsistent_frames_only:
                    err_thresh_ratios = err_thresh_ratios[temporal_consistency_abs_err > threshold]
                err_thresh_ratios_by_attr[f"{attr}_error_to_threshold_ratio"] = err_thresh_ratios
            else:
                metrics[f"{attr}_neigh_inter_diff"] = np.max(temporal_consistency_abs_err)

        if self._thresholds:
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

        If no thresholds are provided, then the metrics are aggregated differently, since we're more interested in the
        limit cases to determine the thresholds than in the general case.

        Args:
            metrics: Metrics computed over each result.

        Returns:
            Global statistics on the metrics computed over each result.
        """
        if self._thresholds:
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
        else:
            return metrics.agg(["max", "min", "mean", "std"])

    @classmethod
    def build_parser(cls) -> ArgumentParser:
        """Creates parser for temporal metrics processor.

        Returns:
            Parser object for temporal metrics processor.
        """
        parser = super().build_parser()
        parser.add_argument(
            "--attribute_statistics_cfg",
            type=Path,
            default=cls.default_attribute_statistics_cfg,
            help="File containing pre-computed statistics for each attribute, used to normalize their values",
        )
        parser.add_argument(
            "--thresholds_cfg",
            type=Path,
            default=get_vital_root() / "data/camus/statistics/attr_thresholds.yaml",
            help="File containing pre-computed thresholds on the acceptable temporal consistency metrics' values for "
            "each attribute",
        )
        parser.add_argument(
            "--threshold_margins",
            action=StoreDictKeyPair,
            metavar="THRESH1=MGN1,THRESH2=MGN2...",
            help="Margin to add to the thresholds for temporal inconsistencies, following the eq. "
            "`thresh = thresh * (1 + margin)`. Should only be set when `thresholds_cfg` is provided.",
        )
        parser.add_argument(
            "--inconsistent_frames_only",
            action="store_true",
            help="For metrics apart from the proportion of inconsistent frames, only compute these metrics on the "
            "inconsistent frames",
        )
        return parser
