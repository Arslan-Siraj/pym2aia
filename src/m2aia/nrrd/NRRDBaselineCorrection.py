import numpy as np


class BaselineCorrection:
    """
    Removes slowly varying background intensity from spectra.
    TopHat and Median are included in the prototype API.
    These methods are not implemented yet and leave spectra unchanged.
    "None" disables baseline correction.
    """

    SUPPORTED = {"None", "TopHat", "Median"}

    @staticmethod
    def apply(spectra, strategy="None", half_window_size=50):
        spectra = np.asarray(spectra, dtype=np.float32)

        if strategy not in BaselineCorrection.SUPPORTED:
            raise ValueError(
                f"Unsupported baseline correction '{strategy}'. "
                f"Supported: {sorted(BaselineCorrection.SUPPORTED)}"
            )

        if strategy != "None":
            print(
                f"Baseline correction '{strategy}' is not implemented yet "
                "for NRRDSpectrumReader."
            )

        return spectra
