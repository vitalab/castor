from abc import abstractmethod
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from vital.results.camus.utils.data_struct import ViewResult
from vital.results.camus.utils.itertools import PatientViews
from vital.results.processor import ResultsProcessor


class AttributesPlots(ResultsProcessor):
    """Abstract class that plots attributes w.r.t. time."""

    desc = "attrs_plots"
    IterableResultT = PatientViews
    input_choices: Sequence[str]  #: Tags of the data on which it is possible to get attributes w.r.t. time

    def __init__(self, inputs: Sequence[str], **kwargs):
        """Initializes class instance.

        Args:
            inputs: Data item for which to analyze attributes' evolution w.r.t. time.
            **kwargs: Additional parameters to pass along to ``super().__init__()``.
        """
        formatted_input = "_".join(input_tag.replace("/", "-") for input_tag in inputs)
        super().__init__(output_name=f"{formatted_input}_{self.desc}", **kwargs)
        if any(input_tag not in self.input_choices for input_tag in inputs):
            raise ValueError(
                f"The `input` values should be chosen from one of the supported values: {self.input_choices}. "
                f"You passed '{inputs}' as values for `inputs`."
            )
        self.input_tags = inputs

    def process_result(self, result: ViewResult) -> None:
        """Plots attributes w.r.t. time.

        Args:
            result: Data structure holding all the sequence`s data.
        """
        # Ensure that matplotlib is using 'agg' backend
        # to avoid possible 'Could not connect to any X display' errors
        # when no X server is available, e.g. in remote terminal
        plt.switch_backend("agg")

        attrs = self._extract_attributes_data(result)
        self._plot_attributes_wrt_time(attrs, self.output_path, result.id)
        plt.close()  # Close the figure to avoid contamination between plots

    def _extract_attributes_data(self, result: ViewResult) -> pd.DataFrame:
        """Queries the attributes' values and metadata for easy downstream lookup/manipulation.

        Args:
            result: Data structure holding all the sequence`s data.

        Returns:
            Structured attribute values and metadata.
        """
        attrs_data = {
            input_tag.replace("/", "_"): pd.DataFrame.from_dict(self._extract_attributes_from_result(result, input_tag))
            for input_tag in self.input_tags
        }
        attrs_data = pd.concat(attrs_data).rename_axis(["data", "frame"]).reset_index()
        attrs_data = attrs_data.melt(
            id_vars=["data", "frame"],
            value_vars=attrs_data.columns.difference(["data", "frame"]),
            var_name="attr",
            value_name="val",
        )
        return attrs_data

    @abstractmethod
    def _extract_attributes_from_result(self, result: ViewResult, item_tag: str) -> Dict[str, np.ndarray]:
        """Extracts the attributes' values over time for one item of the sequence.

        Args:
            result: Data structure holding all the sequence`s data.
            item_tag: Sequence item (e.g. `pred`, `gt`, etc.) for which to extract the attributes' data.

        Returns:
            Attributes' values over time for one item of the sequence.
        """

    @abstractmethod
    def _plot_attributes_wrt_time(self, attrs: pd.DataFrame, plots_root_dir: Path, result_id: str) -> None:
        """Plot the evolution of the attributes' values w.r.t. time.

        Args:
            attrs: Structured attribute values and metadata.
            plots_root_dir: Root output directory in which to save the attributes' plots.
            result_id: Unique ID of the result for which to plot the attributes. Used to name the plot file.
        """

    @classmethod
    def build_parser(cls) -> ArgumentParser:
        """Creates parser for attributes plot processor.

        Returns:
            Parser object for attributes plot processor.
        """
        parser = super().build_parser()
        parser.add_argument(
            "--inputs",
            type=str,
            nargs="+",
            default=cls.input_choices,
            choices=cls.input_choices,
            help="Data item for which to analyze attributes' evolution w.r.t. time",
        )
        return parser
