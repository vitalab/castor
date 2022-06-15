import argparse
import typing
from pathlib import Path
from typing import Any

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
from panel.layout import Panel
from vital.tasks.autoencoder import SegmentationAutoencoderTask
from vital.tasks.utils.autoencoder import decode, load_encodings, rename_significant_dims
from vital.utils.saving import load_from_checkpoint

hv.extension("bokeh")


def interactive_latent_space_manipulation(
    autoencoder: SegmentationAutoencoderTask, encodings: pd.DataFrame, margin: float, step: float
) -> Panel:
    """Organizes an interactive layout of widgets and images to manipulate samples in the latent space.

    Args:
        autoencoder: Autoencoder model with generative capabilities used to decode the encoded samples.
        encodings: Data set of reference encodings to help navigate the latent space.
             Should contain two level of indices:
                - 1st level: Groups within the data
                - 2nd level: Individual samples within each group
        margin: Factor that defines the bounds for each slider around the selected value. The bounds are defined as
            `value ± margin * stddev`, where stddev is the standard deviation of p(z) for the current dimension.
        step: Factor that defines the size of each slider's step, relative to p(z)'s stddev along that dimension
            according to `step_size = stddev * step`.

    Returns:
        Interactive layout of widgets and images to manipulate samples in the latent space.
    """
    # Make sure the autoencoder model is in 'eval' mode
    autoencoder.eval()

    # Query metadata from the loaded model
    dims = encodings.columns.tolist()

    # Compute bounds on sliders for each latent dimension
    stddevs = encodings.std(axis="index")
    bounds_diff = stddevs * margin

    # Define the widgets to hierarchically select a group and a sample within the group
    group_ids = encodings.index.unique(0).tolist()
    group = pn.widgets.Select(name="group", value=group_ids[0], options=group_ids)
    sample_ids = encodings.loc[group.value].index.tolist()
    sample = pn.widgets.Select(name="sample", value=sample_ids[0], options=sample_ids)

    # Define the widgets to play around with the latent space dimensions
    latent_dim_widgets = {}
    for dim in dims:
        sample_latent_dim_val = encodings.loc[(group.value, sample.value), dim]
        latent_dim_widgets[dim] = pn.widgets.FloatSlider(
            name=dim,
            value=sample_latent_dim_val,
            start=sample_latent_dim_val - bounds_diff[dim],
            end=sample_latent_dim_val + bounds_diff[dim],
            step=step * stddevs[dim],
        )

    # Define functions called when refreshing the dynamic maps' content
    @pn.depends(**latent_dim_widgets)
    def _decode_sample(**encoding_dims: float) -> hv.Image:
        decoded_sample = decode(autoencoder, np.array(list(encoding_dims.values())))
        return hv.Image(decoded_sample).opts(xaxis=None, yaxis=None)

    # Define interactive update functions for the widgets
    @pn.depends(group, watch=True)
    def _update_sample(group: Any) -> None:
        sample_ids = encodings.loc[group].index.tolist()
        sample.options = sample_ids
        sample.value = sample_ids[0]

    @pn.depends(group, sample, watch=True)
    def _update_latent_dims(group: Any, sample: Any) -> None:
        for dim in dims:
            sample_latent_dim_val = encodings.loc[(group, sample), dim]
            latent_dim_widgets[dim].value = sample_latent_dim_val
            latent_dim_widgets[dim].start = sample_latent_dim_val - bounds_diff[dim]
            latent_dim_widgets[dim].end = sample_latent_dim_val + bounds_diff[dim]

    # Configure the interactive data structure
    decoded_sample = hv.DynamicMap(_decode_sample)

    # Configure the widgets in the layout
    widgets = pn.Column(group, sample, *latent_dim_widgets.values())

    # Organize the overall layout and feed starting values
    default_kdims = {
        lat_dim_name: latent_dim_widget.value for lat_dim_name, latent_dim_widget in latent_dim_widgets.items()
    }
    return pn.Row(widgets, decoded_sample.opts(width=800, height=800, title="Decoded sample").select(**default_kdims))


def main():
    """Run the interactive app."""
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
        "--margin",
        type=float,
        default=2,
        help="Factor that defines the bounds for each slider around the selected value. "
        "The bounds are defined as `value ± margin * stddev`, where stddev is the standard deviation of p(z) for the "
        "current dimension.",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.02,
        help="Factor that defines the size of each slider's step, relative to p(z)'s stddev along that dimension "
        "according to `step_size = stddev * step`",
    )
    parser.add_argument("--port", type=int, default=5100, help="Port on which to launch the renderer server")
    args = parser.parse_args()

    # Load system
    autoencoder = typing.cast(SegmentationAutoencoderTask, load_from_checkpoint(args.pretrained_ae))

    # Load precomputed encodings from the results
    encodings = load_encodings(autoencoder.hparams.choices.data, args.results_path, progress_bar=True)
    encodings = rename_significant_dims(encodings, autoencoder)

    # Organize layout
    panel = interactive_latent_space_manipulation(autoencoder, encodings, args.margin, args.step)

    # Launch server for the interactive app
    pn.serve(panel, title="Latent Space Manipulation", port=args.port)


if __name__ == "__main__":
    main()
