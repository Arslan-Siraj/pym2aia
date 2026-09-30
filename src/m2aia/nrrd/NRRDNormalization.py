import numpy as np


class SpectrumNormalization:
    """
    Applies normalization independently to each pixel spectrum.
    Supported methods include TIC, Sum, Mean, Max, and RMS.
    Normalization changes spectral intensities but not image geometry.
    "None" keeps the original spectrum unchanged.
    """

    SUPPORTED = {"None", "TIC", "Sum", "Mean", "Max", "RMS"}

    @staticmethod
    def factor(spectra, x_axis, strategy="None"):
        spectra = np.asarray(spectra, dtype=np.float32)

        if strategy == "None":
            return np.ones((spectra.shape[0], 1), dtype=np.float32)

        if strategy == "TIC":
            if spectra.shape[-1] == 1:
                factors = spectra[:, 0]
            else:
                dx = np.diff(np.asarray(x_axis, dtype=np.float64))
                factors = np.sum(
                    0.5 * (spectra[:, :-1] + spectra[:, 1:]) * dx[None, :],
                    axis=1,
                )
        elif strategy == "Sum":
            factors = np.sum(spectra, axis=1)
        elif strategy == "Mean":
            factors = np.mean(spectra, axis=1)
        elif strategy == "Max":
            factors = np.max(spectra, axis=1)
        elif strategy == "RMS":
            factors = np.sqrt(
                np.mean(np.square(spectra, dtype=np.float64), axis=1)
            )
        else:
            raise ValueError(
                f"Unsupported normalization '{strategy}'. "
                f"Supported: {sorted(SpectrumNormalization.SUPPORTED)}"
            )

        factors = np.asarray(factors, dtype=np.float32)
        factors[~np.isfinite(factors)] = 1.0
        factors[factors == 0] = 1.0
        return factors[:, None]

    @staticmethod
    def apply(spectra, x_axis, strategy="None"):
        spectra = np.asarray(spectra, dtype=np.float32)
        factors = SpectrumNormalization.factor(spectra, x_axis, strategy)
        return (spectra / factors).astype(np.float32, copy=False)
