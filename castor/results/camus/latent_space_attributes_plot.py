import math
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from castor.results.camus.utils.attributes_plot import AttributesPlots
from castor.results.camus.utils.latent_space_attributes import LatentSpaceAttributesMixin


class LatentSpaceAttributesPlots(LatentSpaceAttributesMixin, AttributesPlots):
    """Class that plots latent space attributes w.r.t. time."""

    desc = f"z_{AttributesPlots.desc}"

    def __init__(self, residual_dims_group_size: int = None, **kwargs):
        """Initializes class instance.

        Args:
            residual_dims_group_size: How many residual dimensions to include in auxiliary plots. If '0' or 'None', the
                residual dimensions will be ignored.
            **kwargs: Additional parameters to pass along to ``super().__init__()``.
        """
        self._residual_dims_group_size = residual_dims_group_size
        super().__init__(include_residual_dims=bool(residual_dims_group_size), **kwargs)

    def _plot_attributes_wrt_time(self, attrs: pd.DataFrame, plots_root_dir: Path, result_id: str) -> None:
        # Separate interpretable attributes' data from reisual dimensions' data
        attrs_mask = attrs.attr.isin(self.attr_dims)

        # Plot interpretable attributes' data
        with sns.axes_style("darkgrid"):
            sns.lineplot(data=attrs[attrs_mask], x="frame", y="val", hue="attr", style="data")
        plt.savefig(plots_root_dir / f"{result_id.replace('/', '-')}.png")
        plt.close()

        if self._residual_dims_group_size:
            res_dims_output_folder = plots_root_dir / f"{result_id.replace('/', '-')}_residual_dims"
            res_dims_output_folder.mkdir(exist_ok=True)

            # Plot residual dimensions' data
            res_dims_data = attrs[~attrs_mask]
            res_dims = res_dims_data.attr.unique()
            for group_idx in range(math.ceil(len(res_dims) / self._residual_dims_group_size)):
                res_dims_group = res_dims[
                    group_idx * self._residual_dims_group_size : (group_idx + 1) * self._residual_dims_group_size
                ]
                res_dims_group_data = res_dims_data[res_dims_data.attr.isin(res_dims_group)]

                with sns.axes_style("darkgrid"):
                    sns.lineplot(data=res_dims_group_data, x="frame", y="val", hue="attr", style="data")
                plt.savefig(res_dims_output_folder / (",".join(res_dims_group) + ".png"))
                plt.close()

    @classmethod
    def build_parser(cls) -> ArgumentParser:
        """Creates parser for processor that plots latent space attributes.

        Returns:
            Parser object for processor that plots latent space attributes.
        """
        parser = super().build_parser()
        parser.add_argument(
            "--residual_dims_group_size",
            type=int,
            help="How many residual dimensions to include in auxiliary plots. If '0' or 'None', the residual "
            "dimensions will be ignored",
        )
        return parser


def main():
    """Run the script."""
    LatentSpaceAttributesPlots.main()


if __name__ == "__main__":
    main()
