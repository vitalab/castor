import argparse
import logging
import random
import typing
from pathlib import Path
from typing import Dict, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from vital.tasks.autoencoder import SegmentationAutoencoderTask
from vital.tasks.utils.autoencoder import decode, load_encodings, rename_significant_dims
from vital.utils.logging import configure_logging
from vital.utils.saving import load_from_checkpoint


def compute_attributes_sweeps(
    encodings: pd.DataFrame, attrs: Sequence[str], margin: float, num_steps: int
) -> Dict[str, Dict[str, float]]:
    """Computes differences to apply to each attribute's reference value, based on the attribute's prior.

    Args:
        encodings: Latent space encodings, each column matching to a latent space dimension.
        attrs: Attributes for which to compute sweeps over the values.
        margin: Factor that defines the bounds for each sweep around the selected sample. The bounds are defined as
            `value ± margin * stddev`, where stddev is the standard deviation of p(z) for the current dimension.
        num_steps: Number of values to include in the sweep between the selected sample's value and the min/max bounds,
            for each attribute to sweep.

    Returns:
        Differences to the reference value to sweep, with identifying tags, for each specified attribute.
    """
    # Compute differences on each attribute, based on the attributes' prior in the latent space
    stddevs = {attr: stddev for attr, stddev in encodings.std(axis="index").to_dict().items() if attr in attrs}
    step_factors = np.linspace(-margin, margin, num=(num_steps * 2) + 1)
    tags = [f"{step_factor:+.1f}σ" for step_factor in step_factors]
    tags[num_steps] = "original"
    return {attr: dict(zip(tags, step_factors * stddev)) for attr, stddev in stddevs.items()}


def sweep_attributes_around_sample(
    autoencoder: SegmentationAutoencoderTask, sample: pd.Series, attrs_sweeps: Mapping[str, Mapping[str, float]]
) -> Dict[str, Dict[str, np.ndarray]]:
    """Performs sweeps on attribute values from a `sample` reference, decoding each resulting configuration.

    Args:
        autoencoder: Autoencoder model used to decode the results of the attributes' sweeps.
        sample: Reference encoded sample, based on which all sweeps are computed.
        attrs_sweeps: Differences to apply to the reference's value, with identifying tags, for specific attributes.

    Returns:
        Decoded results from the attributes' sweeps, mapping to the values from the sweeps.
    """
    samples_sweep = {attr: {} for attr in attrs_sweeps}
    for attr, sweep in attrs_sweeps.items():
        for tag, diff in sweep.items():
            manipulated_sample = sample.copy()
            manipulated_sample[attr] += diff
            samples_sweep[attr][tag] = decode(autoencoder, manipulated_sample.to_numpy())
    return samples_sweep


def display_images(
    size_by_img: int = 3,
    cmap: str = "gray",
    title: str = None,
    groups_title: str = None,
    display_group_labels: Literal["on", "off"] = "on",
    images_title: str = None,
    display_image_labels: Literal["on", "shared", "off"] = "shared",
    **image_groups: Dict[str, np.ndarray],
) -> None:
    """Display groups of images as rows in a mosaic-like figure, each group with a specific tag.

    Args:
        size_by_img: Factor to use to compute `figsize` as `ncols * size_by_img, nrows * size_by_img`. This is made
            configurable to allow to change the global size of the figure depending on the nature of the images to plot.
        cmap: Color map to use for the images.
        title: Title of the mosaic of images.
        groups_title: Title of the y-axis, describing what groups represent.
        display_group_labels: Whether to enable the display of the groups' labels, at the left of each row.
        images_title: Title of the x-axis, describing what the variation inside a group represents.
        display_image_labels: Whether to enable the display of the images' labels, on top of each image, indicating how
            each image differs from the reference image.
        **image_groups: Groups of images, where the name of the group will be used as its label at the beginning of the
            row.
    """
    num_samples_by_src = {src_name: len(samples) for src_name, samples in image_groups.items()}
    num_samples = random.choice(list(num_samples_by_src.values()))
    if not all(src_num_samples == num_samples for src_num_samples in num_samples_by_src.values()):
        raise ValueError(
            f"`display_data_samples` requires all data sources to provide the same number of (corresponding) samples. "
            f"You provided the following data sources (with the number of samples for each one): {num_samples_by_src}."
        )
    nrows, ncols = len(image_groups), num_samples

    def _display_image(ax: Axes, image: np.ndarray, cmap: str, group_label: str, image_label: str) -> None:
        if image.ndim == 3 and image.shape[0] == 1:
            image = image.squeeze(0)
        elif image.ndim != 2:
            raise RuntimeError(
                "Can't display image that is not 2D or channel+2D. The image you're trying to display has the "
                f"following shape: {image.shape}."
            )
        ax.imshow(image, cmap=cmap)
        if display_group_labels == "on" and ax.get_subplotspec().is_first_col():
            ax.set_ylabel(group_label, size="x-large")
            ax.set(yticks=[], yticklabels=[])
        else:
            ax.yaxis.set_visible(False)
        if display_image_labels == "on" or (display_image_labels == "shared" and ax.get_subplotspec().is_last_row()):
            ax.set_xlabel(image_label, size="x-large")
            ax.set(xticks=[], xticklabels=[])
        else:
            ax.xaxis.set_visible(False)

    ylabel_size, xlabel_size = 0, 0
    if display_group_labels == "on":
        ylabel_size += 0.15
    if groups_title:
        ylabel_size += 0.25
    if display_image_labels == "shared":
        xlabel_size += 0.15
    elif display_image_labels == "on":
        xlabel_size += 0.15 * nrows
    if images_title:
        xlabel_size += 0.25
    fig, axs = plt.subplots(
        nrows,
        ncols,
        squeeze=False,
        figsize=((ncols * size_by_img) + xlabel_size, (nrows * size_by_img) + ylabel_size),
        constrained_layout=True,
    )
    for row, (group_label, images) in enumerate(image_groups.items()):
        for col, (image_label, image) in enumerate(images.items()):
            ax = axs[row, col]
            _display_image(ax, image, cmap, group_label, image_label)

    if title:
        fig.suptitle(title, size="xx-large")
    if groups_title:
        fig.supylabel(groups_title, size="xx-large")
    if images_title:
        fig.supxlabel(images_title, size="xx-large")
    plt.show()


def main():
    """Run the interactive app."""
    configure_logging(log_to_console=True, console_level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "pretrained_ae",
        type=Path,
        help="Path to a model checkpoint, or name of a model from a Comet model registry, of an autoencoder",
    )
    parser.add_argument(
        "results_path", type=Path, help="Path to an HDF5 dataset of results including latent space encodings"
    )
    parser.add_argument(
        "--attrs",
        type=str,
        nargs="+",
        help="Attributes dimension for which to perform the sweep around a reference sample",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=1.5,
        help="Factor that defines the bounds for each sweep around the selected sample. "
        "The bounds are defined as `value ± margin * stddev`, where stddev is the standard deviation of p(z) for the "
        "current dimension.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=1,
        help="Number of steps to sweep between the selected sample's value and the min/max bounds, "
        "for each dimension to sweep",
    )
    parser.add_argument(
        "--size_by_img",
        type=int,
        default=3,
        help="Factor to use to compute `figsize` as `ncols * size_by_img, nrows * size_by_img`. This is made "
        "configurable to allow to change the global size of the figure depending on the nature of the images to plot.",
    )
    parser.add_argument("--cmap", type=str, default="gray", help="Color map to use for the images")
    parser.add_argument(
        "--disable_titles",
        dest="display_titles",
        action="store_false",
        help="Disable the display of titles describing the axes",
    )
    parser.add_argument(
        "--disable_group_labels",
        dest="display_group_labels",
        action="store_false",
        help="Disable the display of the groups' labels, at the left of each row",
    )
    parser.add_argument(
        "--disable_image_labels",
        dest="display_image_labels",
        action="store_false",
        help="Disable the display of the image' labels, on top of each image, indicating how each image differs from "
        "the reference image",
    )
    args = parser.parse_args()

    # Load system
    autoencoder = typing.cast(
        SegmentationAutoencoderTask,
        load_from_checkpoint(args.pretrained_ae, expected_checkpoint_type=SegmentationAutoencoderTask),
    )

    # Load precomputed encodings from the results
    encodings = load_encodings(autoencoder.hparams.choices.data, args.results_path, progress_bar=True)
    encodings = rename_significant_dims(encodings, autoencoder)

    # Select reference latent vector around which to perform sweep
    group_tag = input("Group: ")
    sample_tag = int(input("Sample: "))
    sample = encodings.loc[(group_tag, sample_tag)]

    # Perform sweep
    sweep_params = compute_attributes_sweeps(encodings, args.attrs, args.margin, args.num_steps)
    sweep_res = sweep_attributes_around_sample(autoencoder, sample, sweep_params)

    # Display sweep results as a mosaic of images
    display_images(
        size_by_img=args.size_by_img,
        cmap=args.cmap,
        groups_title="Attribute" if args.display_titles else None,
        display_group_labels="on" if args.display_group_labels else "off",
        images_title="Value w.r.t. reference image" if args.display_titles else None,
        display_image_labels="shared" if args.display_image_labels else "off",
        **sweep_res,
    )


if __name__ == "__main__":
    main()
