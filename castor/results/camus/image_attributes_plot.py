import itertools
from argparse import ArgumentParser
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from vital.data.camus.config import CamusTags

from castor.results.camus.utils.attributes_plot import AttributesPlots
from castor.results.camus.utils.image_attributes import ImageAttributesMixin


class ImageAttributesPlots(ImageAttributesMixin, AttributesPlots):
    """Class that plots image attributes w.r.t. time."""

    desc = f"seg_{AttributesPlots.desc}"
    _ylabel_mappings = {
        **dict.fromkeys([CamusTags.lv_area, CamusTags.myo_area, CamusTags.atrium_area, "area"], "number of pixels"),
        **dict.fromkeys([CamusTags.lv_base_width, CamusTags.lv_length], "length (in pixels)"),
        **dict.fromkeys([CamusTags.lv_orientation], "angle (in degrees)"),
        **dict.fromkeys([CamusTags.epi_center_x, CamusTags.epi_center_y, "center"], "pixel index"),
    }

    def __init__(
        self,
        hue: Optional[Literal["attr", "data"]] = "attr",
        style: Optional[Literal["attr", "data"]] = "data",
        **kwargs,
    ):
        """Initializes class instance.

        Args:
            hue: Field of the attributes' data to use to assign the curves' hues.
            style: Field of the attributes' data to use to assign the curves' styles.
            **kwargs: Additional parameters to pass along to ``super().__init__()``.
        """
        super().__init__(**kwargs)
        if hue is None and style is None:
            raise ValueError(
                "You must set at least one grouping variable, i.e. `hue` or `style`, with which to group the image "
                "attributes' data."
            )
        self.hue = hue
        self.style = style

    def _plot_attributes_wrt_time(self, attrs: pd.DataFrame, plots_root_dir: Path, result_id: str) -> None:
        if "attr" in (self.hue, self.style):
            # Only group metrics with comparable attribute values in the same plot
            attr_masks = {"area": attrs.attr.str.contains("area"), "center": attrs.attr.str.contains("center")}
            grouped_attr_mask = np.logical_or.reduce(list(attr_masks.values()))
            attr_masks.update({attr: attrs.attr == attr for attr in attrs[~grouped_attr_mask].attr.unique()})
        else:
            attr_masks = {attr: attrs.attr == attr for attr in attrs.attr.unique()}

        if "data" in (self.hue, self.style):
            data_masks = {None: [True] * len(attrs)}
        else:
            data_masks = {data: attrs.data == data for data in attrs.data.unique()}

        for (attr_tag, attr_mask), (data_tag, data_mask) in itertools.product(attr_masks.items(), data_masks.items()):
            plot_file_parts = [result_id.replace("/", "-")]
            if attr_tag:
                plot_file_parts.append(attr_tag)
            if data_tag:
                plot_file_parts.append(data_tag)
            plot_file = plots_root_dir / f"{'_'.join(plot_file_parts)}.png"
            with sns.axes_style("darkgrid"):
                lineplot = sns.lineplot(
                    data=attrs[attr_mask & data_mask], x="frame", y="val", hue=self.hue, style=self.style
                )
            lineplot.set(ylabel=self._ylabel_mappings.get(attr_tag, "val"))
            plt.savefig(plot_file)
            plt.close()  # Close the figure to avoid contamination between plots

    @classmethod
    def build_parser(cls) -> ArgumentParser:
        """Creates parser for processor that plots image attributes.

        Returns:
            Parser object for processor that plots image attributes.
        """
        parser = super().build_parser()
        parser.add_argument(
            "--hue",
            type=str,
            nargs="?",
            choices=["attr", "data"],
            default="attr",
            help="Field of the attributes' data to use to assign the curves' hues",
        )
        parser.add_argument(
            "--style",
            type=str,
            nargs="?",
            choices=["attr", "data"],
            default="data",
            help="Field of the attributes' data to use to assign the curves' styles",
        )
        return parser


def main():
    """Run the script."""
    ImageAttributesPlots.main()


if __name__ == "__main__":
    main()
