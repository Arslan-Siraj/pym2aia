from pathlib import Path

import numpy as np
import SimpleITK as sitk

import m2aia as m2


TEST_FILE = Path(__file__).parent / "data" / "small_mir.nrrd"


def test_write_two_wavenumbers_roundtrip(tmp_path):
    reader = m2.NRRDSpectrumReader(TEST_FILE)
    reader.SetSpectralSelection("Exact")

    output = tmp_path / "selected_1650_1690.nrrd"
    written = reader.WriteNRRD(
        output,
        wavenumbers=[1650, 1690],
    )

    assert written == output
    assert output.is_file()

    result = m2.NRRDSpectrumReader(output)

    assert np.allclose(result.GetXAxis(), [1650.0, 1690.0])
    assert result.GetXAxisDepth() == 2
    assert np.array_equal(result.GetShape(), reader.GetShape())

    assert np.allclose(
        result.GetArray(1650, selection="Exact"),
        reader.GetArray(1650, selection="Exact"),
    )
    assert np.allclose(
        result.GetArray(1690, selection="Exact"),
        reader.GetArray(1690, selection="Exact"),
    )


def test_write_one_wavenumber_roundtrip(tmp_path):
    reader = m2.NRRDSpectrumReader(TEST_FILE)
    reader.SetSpectralSelection("Exact")

    output = tmp_path / "selected_1650.nrrd"
    reader.WriteNRRD(
        output,
        wavenumbers=[1650],
    )

    result = m2.NRRDSpectrumReader(output)

    assert result.GetXAxisDepth() == 1
    assert np.allclose(result.GetXAxis(), [1650.0])
    assert np.array_equal(result.GetShape(), reader.GetShape())
    assert np.allclose(
        result.GetArray(1650, selection="Exact"),
        reader.GetArray(1650, selection="Exact"),
    )


def test_write_preserves_geometry(tmp_path):
    reader = m2.NRRDSpectrumReader(TEST_FILE)

    output = tmp_path / "geometry.nrrd"
    reader.WriteNRRD(
        output,
        wavenumbers=[1650, 1690],
        selection="Exact",
    )

    result = m2.NRRDSpectrumReader(output)

    assert np.allclose(result.GetSpacing(), reader.GetSpacing())
    assert np.allclose(result.GetOrigin(), reader.GetOrigin())
    assert np.allclose(result.GetDirection(), reader.GetDirection())


def test_write_nearest_records_actual_wavenumber(tmp_path):
    reader = m2.NRRDSpectrumReader(TEST_FILE)
    reader.SetSpectralSelection("Nearest")

    output = tmp_path / "nearest_1688.nrrd"
    reader.WriteNRRD(
        output,
        wavenumbers=[1688],
    )

    result = m2.NRRDSpectrumReader(output)

    assert np.allclose(result.GetXAxis(), [1690.0])
    assert np.allclose(
        result.GetArray(1690, selection="Exact"),
        reader.GetArray(1690, selection="Exact"),
    )


def test_write_records_processing_metadata(tmp_path):
    reader = m2.NRRDSpectrumReader(TEST_FILE)
    reader.SetSpectralSelection("Window")
    reader.SetTolerance(25)
    reader.SetPooling("Mean")

    output = tmp_path / "pooled_1670.nrrd"
    reader.WriteNRRD(
        output,
        wavenumbers=[1670],
    )

    image = sitk.ReadImage(str(output))

    assert image.GetMetaData("m2aia.processing.spectral_selection") == "Window"
    assert image.GetMetaData("m2aia.processing.pooling") == "Mean"
    assert float(image.GetMetaData("m2aia.processing.tolerance_cm-1")) == 25.0
    assert image.GetMetaData("m2aia.output.requested_wavenumbers") == "1670"
    assert image.GetMetaData("m2aia.output.selected_wavenumbers") == "1650,1690"


def test_write_window_pooling_roundtrip(tmp_path):
    reader = m2.NRRDSpectrumReader(TEST_FILE)
    reader.SetSpectralSelection("Window")
    reader.SetTolerance(25)
    reader.SetPooling("Mean")

    expected = reader.GetArray(1670, squeeze=False)

    output = tmp_path / "mean_window_1670.nrrd"
    reader.WriteNRRD(
        output,
        wavenumbers=[1670],
    )

    result = m2.NRRDSpectrumReader(output)

    # A pooled Window channel is represented by the requested window center.
    assert np.allclose(result.GetXAxis(), [1670.0])
    assert np.allclose(
        result.GetArray(1670, selection="Exact"),
        expected,
    )
