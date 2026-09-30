import numpy as np


class ImageSmoothing:
    """
    Applies spatial smoothing to an extracted spectral image.
    Median and Gaussian are included in the prototype API.
    These methods are not implemented yet and leave the image unchanged.
    "None" keeps the extracted image unchanged.
    """

    SUPPORTED = {"None", "Median", "Gaussian"}

    @staticmethod
    def apply(array, strategy="None"):
        array = np.asarray(array)

        if strategy not in ImageSmoothing.SUPPORTED:
            raise ValueError(
                f"Unsupported image smoothing '{strategy}'. "
                f"Supported: {sorted(ImageSmoothing.SUPPORTED)}"
            )

        if strategy != "None":
            print(
                f"Image smoothing '{strategy}' is not implemented yet "
                "for NRRDSpectrumReader."
            )

        return array
