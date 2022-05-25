from castor.results.camus.utils.image_attributes import ImageAttributesMixin
from castor.results.camus.utils.temporal_metrics import TemporalMetrics


class ImageTemporalMetrics(ImageAttributesMixin, TemporalMetrics):
    """Class that computes temporal coherence metrics on sequences of image attributes' values."""

    desc = f"seg_{TemporalMetrics.desc}"
    normalization_cfg_choices = ["gt"]


def main():
    """Run the script."""
    ImageTemporalMetrics.main()


if __name__ == "__main__":
    main()
