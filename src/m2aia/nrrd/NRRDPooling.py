import numpy as np


class SpectralPooling:
    """
    Combines multiple spectral channels selected by a tolerance window.
    Mean, Median, Maximum, or Sum reduce them to one value per pixel.
    "None" selects the channel nearest to the requested center.
    Pooling has no effect when only one spectral channel is selected.
    """

    SUPPORTED = {"None", "Mean", "Median", "Maximum", "Sum"}

    @staticmethod
    def apply(selected, selected_x_axis, center, strategy="Maximum"):
        selected = np.asarray(selected)

        if selected.ndim != 2:
            raise ValueError(
                "selected must have shape [number_of_spectra, number_of_channels]."
            )

        if selected.shape[1] == 1:
            return selected[:, 0]

        if strategy == "Mean":
            return np.mean(selected, axis=1)
        if strategy == "Median":
            return np.median(selected, axis=1)
        if strategy == "Maximum":
            return np.max(selected, axis=1)
        if strategy == "Sum":
            return np.sum(selected, axis=1)
        if strategy == "None":
            selected_x_axis = np.asarray(selected_x_axis, dtype=np.float64)
            nearest = int(np.argmin(np.abs(selected_x_axis - float(center))))
            return selected[:, nearest]

        raise ValueError(
            f"Unsupported pooling '{strategy}'. "
            f"Supported: {sorted(SpectralPooling.SUPPORTED)}"
        )
