import pathlib
from typing import Dict, List

import numpy as np
import SimpleITK as sitk

from .nrrd.NRRDNormalization import SpectrumNormalization
from .nrrd.NRRDImageNormalization import ImageNormalization
from .nrrd.NRRDPooling import SpectralPooling
from .nrrd.NRRDIntensityTransformation import IntensityTransformation
from .nrrd.NRRDSmoothing import SpectralSmoothing
from .nrrd.NRRDBaselineCorrection import BaselineCorrection
from .nrrd.NRRDImageSmoothing import ImageSmoothing


__all__ = ["NRRDSpectrumReader"]


class NRRDSpectrumReader:
    """
    Reads MIR spectral data stored as a vector NRRD image.
    Provides access to spectra, wavenumbers, metadata, and image geometry.
    Spectral channels can be extracted as NumPy or SimpleITK images.
    Processing is delegated to separate normalization and pooling classes.
    """
    _NORMALIZATIONS = SpectrumNormalization.SUPPORTED
    _POOLING = SpectralPooling.SUPPORTED
    _IMAGE_NORMALIZATIONS = ImageNormalization.SUPPORTED
    _INTENSITY_TRANSFORMATIONS = IntensityTransformation.SUPPORTED
    _BASELINE_CORRECTIONS = BaselineCorrection.SUPPORTED
    _SMOOTHINGS = SpectralSmoothing.SUPPORTED
    _IMAGE_SMOOTHINGS = ImageSmoothing.SUPPORTED

    def __init__(
        self,
        nrrd_path,
        x_axis=None,
        normalization="None",
        pooling="Maximum",
        image_normalization="None",
        chunk_size=250000,
    ):
        self.nrrd_path = str(nrrd_path)
        self.image = None
        self.data = None
        self._spectra = None
        self._valid_mask = None
        self._valid_linear_indices = None

        self.normalization = "None"
        self.pooling = "Maximum"
        self.image_normalization = "None"

        self.baseline_correction = "None"
        self.smoothing = "None"
        self.intensity_transformation = "None"
        self.image_smoothing = "None"

        self.tolerance = np.float32(0.0)
        self.chunk_size = int(chunk_size)

        self.Load(x_axis=x_axis)

        self.SetNormalization(normalization)
        self.SetPooling(pooling)
        self.SetImageNormalization(image_normalization)

    @staticmethod
    def _value(value):
        return value.value if hasattr(value, "value") else str(value)

    def Load(self, x_axis=None):
        path = pathlib.Path(self.nrrd_path)

        if not path.exists():
            raise FileNotFoundError(path)

        self.image = sitk.ReadImage(str(path))

        if self.image.GetDimension() != 3:
            raise ValueError(
                f"Expected a 3D vector NRRD, got dimension {self.image.GetDimension()}."
            )

        components = self.image.GetNumberOfComponentsPerPixel()

        if components <= 1:
            raise ValueError(
                "Expected a vector NRRD with more than one spectral component per pixel."
            )

        data = sitk.GetArrayFromImage(self.image)

        if data.ndim != 4:
            raise ValueError(
                "Expected SimpleITK vector-image layout [z, y, x, channels], "
                f"got {data.shape}."
            )

        self.data = np.asarray(data, dtype=np.float32)
        self.depth_z, self.height, self.width, self.depth = self.data.shape
        self._spectra = self.data.reshape(-1, self.depth)

        self._valid_mask = np.all(np.isfinite(self.data), axis=-1)

        if np.all(self._valid_mask):
            self._valid_linear_indices = None
            self.number_of_spectra = self._spectra.shape[0]
        else:
            self._valid_linear_indices = np.flatnonzero(
                self._valid_mask.reshape(-1)
            ).astype(np.int64)
            self.number_of_spectra = len(self._valid_linear_indices)

        if x_axis is None:
            x_axis = self._read_x_axis()

        self.x_axis = np.asarray(x_axis, dtype=np.float64)

        if self.x_axis.ndim != 1:
            raise ValueError("x_axis must be one-dimensional.")

        if self.x_axis.size != self.depth:
            raise ValueError(
                f"x_axis has {self.x_axis.size} values but the NRRD has "
                f"{self.depth} spectral components per pixel."
            )

    def _read_x_axis(self):
        keys = {
            key.lower(): key
            for key in self.image.GetMetaDataKeys()
        }

        for candidate in (
            "m2aia_xaxis",
            "m2aia.xaxis",
            "wavenumbers",
            "wavenumber",
            "type",
        ):
            if candidate not in keys:
                continue

            raw = self.image.GetMetaData(keys[candidate])

            try:
                values = [
                    float(v.strip())
                    for v in raw.replace(";", ",").split(",")
                    if v.strip()
                ]
            except ValueError:
                continue

            if len(values) == self.depth:
                return np.asarray(values, dtype=np.float64)

            if len(values) == 2 and self.depth > 2:
                return np.linspace(
                    values[0],
                    values[1],
                    self.depth,
                    dtype=np.float64,
                )

        raise ValueError(
            "Could not determine the spectral x-axis from NRRD metadata. "
            "Pass x_axis explicitly."
        )

    def path(self) -> pathlib.Path:
        return pathlib.Path(self.nrrd_path)

    def dir(self) -> pathlib.Path:
        return self.path().parent

    def name(self) -> str:
        return self.path().name

    def GetImageName(self) -> str:
        return self.path().stem

    def CheckHandle(self):
        if self.image is None or self.data is None:
            raise ReferenceError(
                "Please initialize the reader with a valid NRRD file."
            )

    def GetModality(self) -> str:
        return "MIR"

    def GetSpectralUnit(self) -> str:
        return "cm^-1"

    def GetSpectrumType(self) -> str:
        return "ContinuousProfile"

    def GetShape(self) -> np.ndarray:
        self.CheckHandle()
        return np.asarray(
            [self.width, self.height, self.depth_z],
            dtype=np.int32,
        )

    def GetSpacing(self) -> np.ndarray:
        self.CheckHandle()
        return np.asarray(self.image.GetSpacing(), dtype=np.float64)

    def GetOrigin(self) -> np.ndarray:
        self.CheckHandle()
        return np.asarray(self.image.GetOrigin(), dtype=np.float64)

    def GetDirection(self) -> np.ndarray:
        self.CheckHandle()
        return np.asarray(self.image.GetDirection(), dtype=np.float64)

    def GetXAxis(self) -> np.ndarray:
        self.CheckHandle()
        return self.x_axis

    def GetXAxisDepth(self) -> int:
        self.CheckHandle()
        return self.depth

    def GetYDataType(self):
        return np.float32

    def GetSizeInBytesOfYAxisType(self) -> int:
        return np.dtype(np.float32).itemsize

    def GetNumberOfSpectra(self) -> int:
        self.CheckHandle()
        return self.number_of_spectra

    def GetSpectrumDepth(self, index) -> int:
        self._linear_indices([index])
        return self.depth

    def _linear_indices(self, ids):
        ids = np.asarray(ids, dtype=np.int64)

        if np.any(ids < 0) or np.any(ids >= self.number_of_spectra):
            raise IndexError("At least one spectrum index is out of range.")

        if self._valid_linear_indices is None:
            return ids

        return self._valid_linear_indices[ids]

    def GetSpectrumPosition(self, index) -> np.ndarray:
        linear = int(self._linear_indices([index])[0])

        z, y, x = np.unravel_index(
            linear,
            (self.depth_z, self.height, self.width),
        )

        return np.asarray([x, y, z], dtype=np.int32)

    def GetMetaData(self) -> Dict[str, str]:
        self.CheckHandle()

        metadata = {
            key: self.image.GetMetaData(key)
            for key in self.image.GetMetaDataKeys()
        }

        shape = self.GetShape()
        spacing = self.GetSpacing()
        origin = self.GetOrigin()
        direction = self.GetDirection()
        xs = self.GetXAxis()

        metadata["Modality"] = "MIR"
        metadata["SpectralUnit"] = "cm^-1"

        metadata["m2aia.modality"] = "MIR"
        metadata["m2aia.spectral_unit"] = "cm^-1"
        metadata["m2aia.xs.n"] = str(len(xs))
        metadata["m2aia.xs.min"] = str(float(np.min(xs)))
        metadata["m2aia.xs.max"] = str(float(np.max(xs)))
        metadata["m2aia.xs"] = ",".join(str(float(x)) for x in xs)

        metadata["m2aia.dim_x"] = str(int(shape[0]))
        metadata["m2aia.dim_y"] = str(int(shape[1]))
        metadata["m2aia.dim_z"] = str(int(shape[2]))

        metadata["m2aia.spacing_x"] = str(float(spacing[0]))
        metadata["m2aia.spacing_y"] = str(float(spacing[1]))
        metadata["m2aia.spacing_z"] = str(float(spacing[2]))

        metadata["m2aia.origin_x"] = str(float(origin[0]))
        metadata["m2aia.origin_y"] = str(float(origin[1]))
        metadata["m2aia.origin_z"] = str(float(origin[2]))

        metadata["m2aia.direction"] = ",".join(
            str(float(v)) for v in direction
        )

        metadata["number of measurements"] = str(
            self.GetNumberOfSpectra()
        )

        return metadata

    def GetMaskArray(self) -> np.ndarray:
        self.CheckHandle()
        return self._valid_mask.astype(np.ushort, copy=True)

    def _copy_geometry(self, image):
        image.SetSpacing(tuple(self.GetSpacing()))
        image.SetOrigin(tuple(self.GetOrigin()))
        image.SetDirection(tuple(self.GetDirection()))
        return image

    def GetMaskImage(self) -> sitk.Image:
        return self._copy_geometry(
            sitk.GetImageFromArray(self.GetMaskArray())
        )

    def GetIndexArray(self) -> np.ndarray:
        self.CheckHandle()

        result = np.zeros(
            self.depth_z * self.height * self.width,
            dtype=np.uint32,
        )

        if self._valid_linear_indices is None:
            result[:] = np.arange(
                self.number_of_spectra,
                dtype=np.uint32,
            )
        else:
            result[self._valid_linear_indices] = np.arange(
                self.number_of_spectra,
                dtype=np.uint32,
            )

        return result.reshape(
            self.depth_z,
            self.height,
            self.width,
        )

    def GetIndexImage(self) -> sitk.Image:
        return self._copy_geometry(
            sitk.GetImageFromArray(self.GetIndexArray())
        )

    def SetNormalization(self, strategy):
        strategy = self._value(strategy)

        if strategy == "MinMax":
            raise ValueError(
                "MinMax is an image-level normalization. "
                "Use SetImageNormalization('MinMax')."
            )

        if strategy not in self._NORMALIZATIONS:
            raise ValueError(
                f"Unsupported normalization '{strategy}'. "
                f"Supported: {sorted(self._NORMALIZATIONS)}"
            )

        self.normalization = strategy

    def SetImageNormalization(self, strategy):
        strategy = self._value(strategy)

        if strategy not in self._IMAGE_NORMALIZATIONS:
            raise NotImplementedError(
                f"Image normalization '{strategy}' is not implemented yet. "
                "Currently supported: None, MinMax."
            )

        self.image_normalization = strategy

    def SetPooling(self, strategy):
        strategy = self._value(strategy)

        if strategy not in self._POOLING:
            raise ValueError(
                f"Unsupported pooling '{strategy}'. "
                f"Supported: {sorted(self._POOLING)}"
            )

        self.pooling = strategy

    def SetTolerance(self, tol):
        self.tolerance = np.float32(tol)

    def GetTolerance(self) -> np.float32:
        return self.tolerance

    def SetBaselineCorrection(self, strategy="None", half_window_size=50):
        strategy = self._value(strategy)

        if strategy not in self._BASELINE_CORRECTIONS:
            raise ValueError(
                f"Unsupported baseline correction '{strategy}'. "
                f"Supported: {sorted(self._BASELINE_CORRECTIONS)}"
            )

        if strategy != "None":
            print(
                f"Baseline correction '{strategy}' is not implemented yet "
                "for NRRDSpectrumReader."
            )
            self.baseline_correction = "None"
            return

        self.baseline_correction = "None"

    def SetSmoothing(self, strategy="None", half_window_size=2):
        strategy = self._value(strategy)

        if strategy not in self._SMOOTHINGS:
            raise ValueError(
                f"Unsupported spectral smoothing '{strategy}'. "
                f"Supported: {sorted(self._SMOOTHINGS)}"
            )

        if strategy != "None":
            print(
                f"Spectral smoothing '{strategy}' is not implemented yet "
                "for NRRDSpectrumReader."
            )
            self.smoothing = "None"
            return

        self.smoothing = "None"

    def SetIntensityTransformation(self, strategy="None"):
        strategy = self._value(strategy)

        if strategy not in self._INTENSITY_TRANSFORMATIONS:
            raise ValueError(
                f"Unsupported intensity transformation '{strategy}'. "
                f"Supported: {sorted(self._INTENSITY_TRANSFORMATIONS)}"
            )

        if strategy != "None":
            print(
                f"Intensity transformation '{strategy}' is not implemented yet "
                "for NRRDSpectrumReader."
            )
            self.intensity_transformation = "None"
            return

        self.intensity_transformation = "None"

    def SetImageSmoothing(self, strategy="None"):
        strategy = self._value(strategy)

        if strategy not in self._IMAGE_SMOOTHINGS:
            raise ValueError(
                f"Unsupported image smoothing '{strategy}'. "
                f"Supported: {sorted(self._IMAGE_SMOOTHINGS)}"
            )

        if strategy != "None":
            print(
                f"Image smoothing '{strategy}' is not implemented yet "
                "for NRRDSpectrumReader."
            )
            self.image_smoothing = "None"
            return

        self.image_smoothing = "None"

    def SetExternalNormalizationArray(self, array):
        print(
            "External normalization is not implemented yet "
            "for NRRDSpectrumReader."
        )
        return

    def _normalization_factor(self, spectra, strategy=None):
        if strategy is None:
            strategy = self.normalization

        strategy = self._value(strategy)

        return SpectrumNormalization.factor(
            spectra,
            self.x_axis,
            strategy,
        )

    def _process_spectra(self, spectra):
        return SpectrumNormalization.apply(
            spectra,
            self.x_axis,
            self.normalization,
        )

    def GetNormalizationArray(self, strategy) -> np.ndarray:
        strategy = self._value(strategy)

        if strategy not in self._NORMALIZATIONS:
            raise NotImplementedError(
                f"Normalization '{strategy}' is not implemented."
            )

        result = np.zeros(
            self.depth_z * self.height * self.width,
            dtype=np.float32,
        )

        for start in range(
            0,
            self.number_of_spectra,
            self.chunk_size,
        ):
            stop = min(
                start + self.chunk_size,
                self.number_of_spectra,
            )

            ids = np.arange(start, stop, dtype=np.int64)
            linear = self._linear_indices(ids)
            spectra = self._spectra[linear]

            factors = self._normalization_factor(
                spectra,
                strategy=strategy,
            )[:, 0]

            result[linear] = factors

        return result.reshape(
            self.depth_z,
            self.height,
            self.width,
        )

    def GetNormalizationImage(self, strategy) -> sitk.Image:
        return self._copy_geometry(
            sitk.GetImageFromArray(
                self.GetNormalizationArray(strategy)
            )
        )

    def _apply_image_normalization(self, array):
        return ImageNormalization.apply(
            array,
            self.image_normalization,
            self._valid_mask,
        )

    def GetSpectrum(self, index) -> List[np.ndarray]:
        linear = int(self._linear_indices([index])[0])

        ys = self._process_spectra(
            self._spectra[linear:linear + 1]
        )[0]

        return [
            self.x_axis.astype(np.float32, copy=False),
            ys.astype(np.float32, copy=False),
        ]

    def GetIntensities(self, index, ys=None) -> np.ndarray:
        values = self.GetSpectrum(index)[1]

        if ys is None:
            return values.copy()

        if ys.dtype != np.float32:
            raise TypeError("ys must have dtype=np.float32.")

        if ys.shape[0] != self.depth:
            ys.resize((self.depth,), refcheck=False)

        ys[:] = values
        return ys

    def GetSpectra(self, indices) -> np.ndarray:
        indices = np.asarray(indices, dtype=np.int64)

        if indices.size == 0:
            return np.zeros(
                (0, self.depth),
                dtype=np.float32,
            )

        linear = self._linear_indices(indices)

        return self._process_spectra(
            self._spectra[linear]
        )

    def GetMeanSpectrum(self) -> np.ndarray:
        self.CheckHandle()

        total = np.zeros(self.depth, dtype=np.float64)
        count = 0

        for start in range(
            0,
            self.number_of_spectra,
            self.chunk_size,
        ):
            stop = min(
                start + self.chunk_size,
                self.number_of_spectra,
            )

            ids = np.arange(start, stop, dtype=np.int64)
            linear = self._linear_indices(ids)
            spectra = self._process_spectra(
                self._spectra[linear]
            )

            total += np.sum(
                spectra,
                axis=0,
                dtype=np.float64,
            )
            count += len(spectra)

        if count == 0:
            return total

        return total / count

    def GetMaxSpectrum(self) -> np.ndarray:
        self.CheckHandle()

        result = np.full(
            self.depth,
            -np.inf,
            dtype=np.float64,
        )

        for start in range(
            0,
            self.number_of_spectra,
            self.chunk_size,
        ):
            stop = min(
                start + self.chunk_size,
                self.number_of_spectra,
            )

            ids = np.arange(start, stop, dtype=np.int64)
            linear = self._linear_indices(ids)
            spectra = self._process_spectra(
                self._spectra[linear]
            )

            result = np.maximum(
                result,
                np.max(spectra, axis=0),
            )

        return result

    def GetArray(
        self,
        center,
        tol,
        dtype=np.float32,
        squeeze=False,
    ) -> np.ndarray:
        self.CheckHandle()

        if dtype not in (np.float32, np.float64):
            raise TypeError(
                "Image pixel type must be np.float32 or np.float64."
            )

        center = float(center)
        tol = float(tol)

        if center < np.min(self.x_axis) or center > np.max(self.x_axis):
            raise ValueError(
                f"Center {center} is outside x-axis range "
                f"[{self.x_axis.min()}, {self.x_axis.max()}]."
            )

        selected_indices = np.flatnonzero(
            (self.x_axis >= center - tol)
            & (self.x_axis <= center + tol)
        )

        if selected_indices.size == 0:
            nearest = int(
                np.argmin(np.abs(self.x_axis - center))
            )
            raise ValueError(
                f"No spectral channel found in [{center - tol}, {center + tol}]. "
                f"Nearest channel is {self.x_axis[nearest]}."
            )

        output = np.zeros(
            self.depth_z * self.height * self.width,
            dtype=np.float32,
        )

        for start in range(
            0,
            self.number_of_spectra,
            self.chunk_size,
        ):
            stop = min(
                start + self.chunk_size,
                self.number_of_spectra,
            )

            ids = np.arange(start, stop, dtype=np.int64)
            linear = self._linear_indices(ids)

            spectra = self._process_spectra(
                self._spectra[linear]
            )
            selected = spectra[:, selected_indices]

            values = SpectralPooling.apply(
                selected,
                self.x_axis[selected_indices],
                center,
                self.pooling,
            )

            output[linear] = values

        output = output.reshape(
            self.depth_z,
            self.height,
            self.width,
        )

        output = self._apply_image_normalization(output)
        output = output.astype(dtype, copy=False)

        if squeeze:
            return np.squeeze(output)

        return output

    def GetImage(
        self,
        center,
        tol,
        dtype=np.float32,
    ) -> sitk.Image:
        image = sitk.GetImageFromArray(
            self.GetArray(
                center,
                tol,
                dtype=dtype,
                squeeze=False,
            )
        )

        return self._copy_geometry(image)

    def SpectrumIterator(self):
        self.CheckHandle()

        for index in range(self.number_of_spectra):
            xs, ys = self.GetSpectrum(index)
            yield index, xs, ys

    def SpectrumRandomBatchIterator(self, batch_size):
        self.CheckHandle()

        while True:
            ids = np.random.randint(
                0,
                self.number_of_spectra,
                int(batch_size),
            )
            yield self.GetSpectra(ids)

    def GetParametersAsFormattedString(self):
        s = ""
        s += f"(normalization {self.normalization})\n"
        s += f"(pooling {self.pooling})\n"
        s += f"(image-normalization {self.image_normalization})\n"
        s += "(baseline-correction NotImplemented)\n"
        s += "(smoothing NotImplemented)\n"
        s += "(intensity-transformation NotImplemented)\n"
        s += "(image-smoothing NotImplemented)\n"
        return s
