import numpy as np


class SpectralSmoothing:
    """
    Smooths intensity variations along the spectral axis.
    SavitzkyGolay and Gaussian are included in the prototype API.
    These methods are not implemented yet and leave spectra unchanged.
    "None" keeps the original spectrum unchanged.
    """

    SUPPORTED = {"None", "SavitzkyGolay", "Gaussian"}

    @staticmethod
    def apply(spectra, strategy="None", half_window_size=2):
        spectra = np.asarray(spectra, dtype=np.float32)

        if strategy not in SpectralSmoothing.SUPPORTED:
            raise ValueError(
                f"Unsupported spectral smoothing '{strategy}'. "
                f"Supported: {sorted(SpectralSmoothing.SUPPORTED)}"
            )

        if strategy != "None":
            print(
                f"Spectral smoothing '{strategy}' is not implemented yet "
                "for NRRDSpectrumReader."
            )

        return spectra
