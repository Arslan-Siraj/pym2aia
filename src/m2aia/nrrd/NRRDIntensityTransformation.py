import numpy as np


class IntensityTransformation:
    """
    Applies a mathematical transformation to spectral intensities.
    SquareRoot, Log2, and Log10 are included in the prototype API.
    These transformations are not implemented yet and leave data unchanged.
    "None" keeps the original intensities unchanged.
    """

    SUPPORTED = {"None", "SquareRoot", "Log2", "Log10"}

    @staticmethod
    def apply(spectra, strategy="None"):
        spectra = np.asarray(spectra, dtype=np.float32)

        if strategy not in IntensityTransformation.SUPPORTED:
            raise ValueError(
                f"Unsupported intensity transformation '{strategy}'. "
                f"Supported: {sorted(IntensityTransformation.SUPPORTED)}"
            )

        if strategy != "None":
            print(
                f"Intensity transformation '{strategy}' is not implemented yet "
                "for NRRDSpectrumReader."
            )

        return spectra
