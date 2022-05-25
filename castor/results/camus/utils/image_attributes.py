import logging
from argparse import ArgumentParser
from typing import Dict, Sequence

import numpy as np
from vital.data.camus.config import CamusTags
from vital.results.camus.utils.data_struct import ViewResult
from vital.results.processor import ResultsProcessor

logger = logging.getLogger(__name__)


class ImageAttributesMixin(ResultsProcessor):
    """Mixin that bundles various behaviors regarding how to handle images' attribute data.

    - Lists data tags that are associated to images with their attributes;
    - How to access their attributes;
    """

    input_choices = [
        f"{CamusTags.pred}/{CamusTags.raw}",
        f"{CamusTags.pred}/{CamusTags.post_pred}",
        f"{CamusTags.gt}/{CamusTags.raw}",
    ]

    def __init__(self, attrs: Sequence[str], **kwargs):
        """Initializes class instance.

        Args:
            attrs: Labels identifying attributes of interest.
            **kwargs: Additional parameters to pass along to ``super().__init__()``.
        """
        super().__init__(**kwargs)
        self.attrs = attrs

    def _extract_attributes_from_result(self, result: ViewResult, item_tag: str) -> Dict[str, np.ndarray]:
        attrs_data = result[item_tag].attrs
        missing_attrs = [attr for attr in self.attrs if attr not in attrs_data]
        if missing_attrs:
            logger.warning(
                f"Requested attributes {missing_attrs} were not available for '{result.id}'. The attributes listed "
                f"were thus ignored. To avoid this warning, either remove these attributes from those required for "
                f"your task, or run your task on data that provides those attributes."
            )
        return {attr: attrs_data[attr] for attr in self.attrs if attr not in missing_attrs}

    @classmethod
    def build_parser(cls) -> ArgumentParser:
        """Creates parser for processors that need to access image attributes.

        Returns:
            Parser object for processors that need to access image attributes.
        """
        parser = super().build_parser()
        parser.add_argument(
            "--attrs",
            type=str,
            nargs="+",
            default=CamusTags.seg_attrs,
            choices=CamusTags.seg_attrs,
            help="Labels identifying attributes of interest",
        )
        return parser
