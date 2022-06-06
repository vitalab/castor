import argparse
import logging
import typing
from pathlib import Path
from typing import Any, Dict, Sequence

import h5py
import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
import umap
from holoviews import streams
from panel.layout import Panel
from vital.tasks.autoencoder import SegmentationAutoencoderTask
from vital.tasks.utils.autoencoder import decode, encode_dataset
from vital.utils.logging import configure_logging
from vital.utils.parsing import StoreDictKeyPair
from vital.utils.saving import load_from_checkpoint

logger = logging.getLogger(__name__)

hv.extension("bokeh")


def interactive_latent_space_global_distribution(
    autoencoder: SegmentationAutoencoderTask, samples: Dict[str, np.ndarray], embedding_kwargs: Dict[str, Any] = None
) -> Panel:
    """Organizes an interactive layout of widgets and images to visualize the global distribution of a latent space.

    Args:
        autoencoder: Autoencoder model with generative capabilities used to encode/decode samples between the image and
            latent space domains.
        samples: Collection of datasets of samples pre-encoded in the autoencoder's latent space, each dataset mapped to
            a unique identifier.
        embedding_kwargs: Keyword arguments to pass to the UMAP transform's constructor.

    Returns:
        Interactive layout of widgets and images to visualize the global distribution of a latent space.
    """
    if embedding_kwargs is None:
        embedding_kwargs = {}

    # Make sure the autoencoder model is in 'eval' mode
    autoencoder.eval()

    # If the latent space is not already 2D, use the UMAP dimensionality reduction algorithm
    # to learn a projection between the latent space and a 2D space ready to be displayed
    samples_data = np.vstack(list(samples.values()))
    high_dim_latent_space = autoencoder.hparams.latent_dim > 2
    if high_dim_latent_space:
        logger.info("Learning UMAP embedding for latent space vectors...")
        reducer = umap.UMAP(**embedding_kwargs)
        samples_data = reducer.fit_transform(samples_data)
    samples_data = pd.DataFrame(samples_data, columns=["x", "y"])

    # Generate labels for each points based on which dataset the points come from
    targets = []
    for label, samples_group in samples.items():
        targets.extend([label] * len(samples_group))
    samples_data = samples_data.assign(target=targets)

    # Define the widgets to select the data labels to display
    labels = pn.widgets.MultiChoice(
        name="Groups of data to display in the 2D scatter plot",
        value=list(samples.keys()),
        options=list(samples.keys()),
    )

    # Setup callback to provide 2D-embedded data points, w/ their associated labels
    @pn.depends(labels)
    def _plot_samples_embedding(value: Sequence[str]) -> hv.Points:
        # Only keep points whose label matches one of the selected labels
        points = samples_data[samples_data.target.str.fullmatch("|".join(value))]

        # Build a cloud of points with the remaining points
        return hv.Points(points, vdims=["target"]).opts(color="target", cmap="Category10")

    embedded_points = hv.DynamicMap(_plot_samples_embedding)

    # Track the user's pointer in the points cloud
    pointer = streams.PointerXY(x=0.0, y=0.0, source=embedded_points)

    # Setup callback to automatically decode selected points
    def _decode_sample(x: float, y: float) -> hv.Image:
        latent_sample = np.array([x, y])[None]
        if high_dim_latent_space:
            # Project the 2D sample back into the higher-dimensionality latent space
            # using UMAP's learned inverse transform
            latent_sample = reducer.inverse_transform(latent_sample)
        return hv.Image(decode(autoencoder, latent_sample)).opts(xaxis=None, yaxis=None)

    decoded_sample = hv.DynamicMap(_decode_sample, streams=[pointer]).opts(axiswise=True)

    # Common options for the main panels to display
    encodings_title = (
        "Latent space"
        if not high_dim_latent_space
        else f"2D UMAP embedding of the {autoencoder.hparams.latent_dim}D latent space"
    )
    return pn.Row(
        pn.Column(embedded_points.opts(width=800, height=800, title=encodings_title), labels),
        decoded_sample.opts(width=800, height=800, title="Decoded sample"),
    )


def main():
    """Run the interactive app."""
    configure_logging(log_to_console=True, console_level=logging.INFO)

    from vital.data.camus.data_module import CamusDataModule

    # Config mapping between dataset name and datamodule
    datasets = {"camus": CamusDataModule}

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "pretrained_ae",
        type=Path,
        help="Path to a model checkpoint, or name of a model from a Comet model registry, of an autoencoder",
    )
    parser.add_argument("--latent_samples_dataset", type=Path, help="Path to an HDF5 dataset of latent space samples")
    parser.add_argument(
        "--embedding_kwargs",
        action=StoreDictKeyPair,
        default=dict(),
        metavar="ARG1=VAL1,ARG2=VAL2...",
        help="Parameters for Lightning's built-in model checkpoint callback",
    )
    parser.add_argument("--port", type=int, default=5100, help="Port on which to launch the renderer server")

    # Add subparsers for all datasets available
    datasets_subparsers = parser.add_subparsers(
        title="dataset", dest="dataset", description="Dataset to encode and plot in the latent space"
    )
    for dataset, datamodule_cls in datasets.items():
        ds_parser = datasets_subparsers.add_parser(dataset, help=f"{dataset.upper()} dataset")
        datamodule_cls.add_argparse_args(ds_parser)

    args = parser.parse_args()

    if args.latent_samples_dataset is None and args.dataset is None:
        raise ValueError(
            "You must provide at least one data source of samples to encode in the latent space. Data can be provided "
            "either through the `dataset` command, to encode the training and validation subsets of a dataset in the "
            "latent space, or through an HDF5 of samples from the latent space. Both sources can also be provided at "
            "the same time, in which case they will be combined."
        )

    # Load system
    autoencoder = typing.cast(SegmentationAutoencoderTask, load_from_checkpoint(args.pretrained_ae))

    # Encode the dataset's train and val subsets in the latent space, if provided
    latent_samples = {}
    if dataset := args.dataset:
        if dataset != autoencoder.hparams.choices.data:
            logger.warning(
                f"You requested to encode the {dataset.upper()} dataset, but the autoencoder model you provided was "
                f"trained on the {autoencoder.hparams.choices.data.upper()} dataset. Unless you know the two datasets "
                f"are similar, poor reconstruction performances are to be expected."
            )
        datamodule = datasets[dataset](**vars(args))
        latent_samples[args.dataset] = encode_dataset(autoencoder, datamodule, progress_bar=True)

    # Load augmented latent space, if provided
    if args.latent_samples_dataset:
        with h5py.File(args.latent_samples_dataset) as latent_dataset:
            latent_samples.update({key: latent_dataset[key][()] for key in latent_dataset})

    # Organize layout
    panel = interactive_latent_space_global_distribution(
        autoencoder, samples=latent_samples, embedding_kwargs=args.embedding_kwargs
    )

    # Launch server for the interactive app
    pn.serve(panel, title="Latent Space Global Distribution", port=args.port)


if __name__ == "__main__":
    main()
