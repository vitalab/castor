## [`autoencoder`](autoencoder): Applications related to autoencoders

### [Latent Space Manipulation](autoencoder/latent_space_manipulation.py)

#### Description
Interactive web page that allows to start from samples in the dataset and to manipulate the dimensions of the encoding
in the latent space, to observe how modifying these dimensions affect the reconstruction of the sample.

#### How to run
The application is installed under the `latent-space-manipulation` command when installing the `castor` python package.
Thus, you can use this alias to launch it when working inside an environment where you have installed `castor`.
```bash
# list all of the application's options
latent-space-manipulation --help

# mininum arguments to provide to launch the application
latent-space-manipulation <AUTOENCODER_CHECKPOINT> <AUTOENCODER_PREDICTIONS_DATASET>
```

### [Latent Space Local Distribution](autoencoder/latent_space_local_distribution.py)

#### Description
Interactive web page that displays the 1D curves of how each latent dimension changes in a segmentation sequence w.r.t. time.
It also supports displaying the where the values of the nearest neighbors for the same dimensions, to investigate how
close the nearest neighbors are compared to other frames in the sequence.

#### How to run
The application is installed under the `latent-space-local-distribution` command when installing the `castor` python package.
Thus, you can use this alias to launch it when working inside an environment where you have installed `castor`.
```bash
# list all of the application's options
latent-space-local-distribution --help

# mininum arguments to provide to launch the application
latent-space-local-distribution <AUTOENCODER_CHECKPOINT> <AUTOENCODER_PREDICTIONS_DATASET>
```

### [Latent Space Global Distribution](autoencoder/latent_space_global_distribution.py)

#### Description
Interactive web page that displays a 2D embedding, obtained using UMAP fitted on the latent space, of all the items in
the dataset.
> WARNING: This application might not work on large datasets, because the computational cost of learning the 2D embedding
> and/or displaying a large number of points interactively might be too much.

#### How to run
The application is installed under the `latent-space-global-distribution` command when installing the `castor` python package.
Thus, you can use this alias to launch it when working inside an environment where you have installed `castor`.
```bash
# list all of the application's options
latent-space-global-distribution --help

# mininum arguments to provide to launch the application
latent-space-global-distribution <AUTOENCODER_CHECKPOINT> <DATASET_PATH>
```

### [Latent Space Attributes Sweeper](autoencoder/latent_space_attributes_sweeper.py)

#### Description
Interactive command line tool that sweeps latent dimensions around a reference sample (specified interactively to the script)
and saves the modified latent vectors' reconstructions as flat images. This application serves a similar purpose to the
[latent space manipulation app](#latent-space-manipulation), only this time instead of providing real-time reconstructions,
it saves the results of the manipulations, thus being more oriented towards producing figures.

#### How to run
The application is installed under the `latent-space-attributes-sweeper` command when installing the `castor` python package.
Thus, you can use this alias to launch it when working inside an environment where you have installed `castor`.
```bash
# list all of the application's options
latent-space-attributes-sweeper --help

# mininum arguments to provide to launch the application
latent-space-attributes-sweeper <AUTOENCODER_CHECKPOINT> <AUTOENCODER_PREDICTIONS_DATASET> --attrs ATTR1 ATTR2
```
