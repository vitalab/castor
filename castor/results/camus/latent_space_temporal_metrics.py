from castor.results.camus.utils.latent_space_attributes import LatentSpaceAttributesMixin
from castor.results.camus.utils.temporal_metrics import TemporalMetrics


class LatentSpaceTemporalMetrics(LatentSpaceAttributesMixin, TemporalMetrics):
    """Class that computes temporal coherence metrics on sequences of latent space attributes' values."""

    desc = f"z_{TemporalMetrics.desc}"
    normalization_cfg_choices = ["ar-vae"]


def main():
    """Run the script."""
    LatentSpaceTemporalMetrics.main()


if __name__ == "__main__":
    main()
