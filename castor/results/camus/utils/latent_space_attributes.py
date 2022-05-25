from argparse import ArgumentParser
from typing import Dict, Mapping

import numpy as np
from vital.data.camus.config import CamusTags
from vital.results.camus.utils.data_struct import ViewResult
from vital.results.processor import ResultsProcessor
from vital.utils.parsing import StoreDictKeyPair


class LatentSpaceAttributesMixin(ResultsProcessor):
    """Mixin that bundles various behaviors regarding how to handle latent encodings' attribute data.

    - Lists data tags that are associated to latent encodings with their attributes;
    - How to access their attributes;
    """

    input_choices = [f"{CamusTags.pred}/{CamusTags.encoding}", f"{CamusTags.gt}/{CamusTags.encoding}"]

    def __init__(self, attr_dims: Mapping[str, int], include_residual_dims: bool = False, **kwargs):
        """Initializes class instance.

        Args:
            attr_dims: Mapping between each attribute of interest (keys) and the index of the corresponding dimension in
                the latent space (value).
            include_residual_dims: Whether to also include residual dimensions, i.e. latent dimensions not correlated to
                specific attributes, in the data to extract.
            **kwargs: Additional parameters to pass along to ``super().__init__()``.
        """
        super().__init__(**kwargs)
        self.attr_dims = attr_dims
        self.include_residual_dims = include_residual_dims

    def _extract_attributes_from_result(self, result: ViewResult, item_tag: str) -> Dict[str, np.ndarray]:
        residual_dims = {}
        if self.include_residual_dims:
            num_dims = result[item_tag].data.shape[-1]
            residual_dims = {
                f"{CamusTags.encoding}_{dim}": dim for dim in range(num_dims) if dim not in self.attr_dims.values()
            }
        return {attr: result[item_tag].data[:, dim] for attr, dim in {**self.attr_dims, **residual_dims}.items()}

    @classmethod
    def build_parser(cls) -> ArgumentParser:
        """Creates parser for processors that need to access latent space attributes.

        Returns:
            Parser object for processors that need to access latent space attributes.
        """
        parser = super().build_parser()
        parser.add_argument(
            "--attr_dims",
            required=True,
            action=StoreDictKeyPair,
            metavar="ATTR1=DIM1,ATTR2=DIM2...",
            help="Mapping between each attribute of interest (keys) and the index of the corresponding dimension in "
            "the latent space (value)",
        )
        return parser
