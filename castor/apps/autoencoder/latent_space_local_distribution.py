import argparse
import typing
from pathlib import Path
from typing import Any, Sequence

import h5py
import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
from panel.layout import Panel
from sklearn.neighbors import NearestNeighbors
from vital.tasks.autoencoder import SegmentationAutoencoderTask
from vital.tasks.utils.autoencoder import load_encodings, rename_significant_dims
from vital.utils.saving import load_from_checkpoint

hv.extension("bokeh")


def interactive_latent_space_local_distribution(
    encodings: pd.DataFrame, samples: np.ndarray = None, attrs: Sequence[str] = None
) -> Panel:
    """Organizes an interactive layout of widgets and plots to visualize the local distribution of a latent space.

    Args:
        encodings: Data set of reference encodings to help navigate the latent space.
             Should contain two level of indices:
                - 1st level: Groups within the data
                - 2nd level: Variation of the samples within each group w.r.t. the independent variable to study
        samples: Additional samples from the autoencoder's latent space, to plot as neighbors along the groups of points
            of interest.
        attrs: If `samples` is provided, `attrs` can be served to select a subset of dimensions (column names from
            `encodings`) to use to search for neighbors. Ignored if `samples` is None.

    Returns:
        Interactive layout of widgets and plots to visualize the local distribution of a latent space.
    """
    # Define the widgets to hierarchically select a group and a latent dimension within the group
    group_ids = encodings.index.unique(0).tolist()
    group = pn.widgets.Select(name="group", value=group_ids[0], options=group_ids)
    dim_ids = encodings.columns.tolist()
    dim = pn.widgets.Select(name="dimension", value=dim_ids[0], options=dim_ids)
    widgets = {"group": group, "dim": dim}

    neighbors_available = samples is not None
    if neighbors_available:
        # Init "global" variables queried by the dynamic plot's callbacks
        # They should be able to be modified inplace, otherwise the callbacks won't have access to updated values
        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(samples)

        # Init new widgets specifically for options related to neighbors
        neighbor_mode_opts = ["all_dims"]
        if attrs:
            neighbor_mode_opts.append("attrs_only")
        neighbor_mode = pn.widgets.Select(name="neighbor_mode", value=neighbor_mode_opts[0], options=neighbor_mode_opts)
        num_neighbors = pn.widgets.DiscreteSlider(name="num_neighbors", options=list(range(6)), value=1)
        widgets.update({"neighbor_mode": neighbor_mode, "num_neighbors": num_neighbors})

        # Precompute nearest neighbor models to use for each option
        # This is done because updating them dynamically is too long (causing the app to lag for multiple seconds)
        neighs = {"all_dims": (NearestNeighbors().fit(samples), dim_ids)}
        if attrs:
            neighs["attrs_only"] = NearestNeighbors().fit(samples[:, [dim_ids.index(dim) for dim in attrs]]), attrs

    # Configure the widgets in the layout
    widgets_layout = pn.Column(*widgets.values())

    # Define functions called when refreshing the dynamic scatter plot's content
    @pn.depends(**widgets)
    def _plot_dim_from_group(
        group: Any, dim: str, neighbor_mode: str = None, num_neighbors: int = int(neighbors_available)
    ) -> hv.Scatter:
        points = encodings.loc[group].reset_index(drop=True).reset_index()
        opts = dict(cmap="Viridis")
        vdims = [dim]  # By default, only plot the latent dimension
        if num_neighbors:
            # Mark all original points as being at 0 distance from themselves,
            points = points.assign(distance=[0] * len(points))
            vdims.append("distance")  # Add the distance to the neighbors as a data to be displayed
            opts.update(dict(color="distance", colorbar=True))

            # Compute original points' nearest neighbors as well as their distance
            neigh, neigh_dims = neighs[neighbor_mode]
            neighbors_dist, neighbors_idx = neigh.kneighbors(points[neigh_dims], n_neighbors=num_neighbors)

            # For every N neighbors of an original point
            for idx, (point_neighbors_idx, point_neighbors_dist) in enumerate(zip(neighbors_idx, neighbors_dist)):
                # Fetch the neighbors' latent vectors
                neighbors_data = samples[point_neighbors_idx]

                # Concatenate the group index and distance to original point to the latent vectors
                neighbors_data = np.hstack(
                    (
                        np.full((num_neighbors, 1), idx),  # (num_neighbors, 1), Repeated index of original point
                        neighbors_data,  # (num_neighbors, latent_dim), Latent vectors
                        point_neighbors_dist[:, None],  # (num_neighbors, 1), Distance to original point
                    )
                )

                # Add the data to the dataframe
                points = points.append(pd.DataFrame(neighbors_data, columns=points.columns))

        return hv.Scatter(points, kdims=["index"], vdims=vdims).opts(**opts)

    # Configure the interactive scatter plot of a latent dimension
    dim_by_group_plot = hv.DynamicMap(_plot_dim_from_group).opts(framewise=True)

    # Organize the overall layout and feed starting values
    return pn.Row(
        widgets_layout,
        dim_by_group_plot.opts(size=7, width=800, height=800, title="Latent dimension w.r.t. index in group").select(
            group=group.value, dim=dim.value
        ),
    )


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
    parser.add_argument("--latent_samples_dataset", type=Path, help="Path to an HDF5 dataset of latent space samples")
    parser.add_argument(
        "--dataset_keys",
        type=str,
        nargs="+",
        default=["no_anatomical_errors"],
        help="Names of the HDF5 dataset objects to load from `latent_samples_dataset`. "
        "Only used if `latent_samples_dataset` is provided.",
    )
    parser.add_argument("--port", type=int, default=5100, help="Port on which to launch the renderer server")
    args = parser.parse_args()

    # Load system
    autoencoder = typing.cast(SegmentationAutoencoderTask, load_from_checkpoint(args.pretrained_ae))

    # Load precomputed encodings from the results
    encodings = load_encodings(autoencoder.hparams.choices.data, args.results_path, progress_bar=True)
    encodings = rename_significant_dims(encodings, autoencoder)

    # Load augmented latent space, if provided
    latent_samples = None
    if args.latent_samples_dataset:
        with h5py.File(args.latent_samples_dataset) as latent_dataset:
            latent_samples = np.vstack([latent_dataset[key][()] for key in args.dataset_keys])

    # Organize layout
    panel = interactive_latent_space_local_distribution(
        encodings, samples=latent_samples, attrs=autoencoder.hparams.get("attrs", None)
    )

    # Launch server for the interactive app
    pn.serve(panel, title="Latent Space Local Distribution", port=args.port)


if __name__ == "__main__":
    main()
