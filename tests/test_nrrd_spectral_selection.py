from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

import m2aia as m2


TEST_FILE = Path(__file__).parent / "data" / "small_mir.nrrd"


def _raw_data():
    image = sitk.ReadImage(str(TEST_FILE))
    return sitk.GetArrayFromImage(image)


def test_nearest_is_default():
    reader = m2.NRRDSpectrumReader(TEST_FILE)

    assert reader.GetSpectralSelection() == "Nearest"
    assert reader.GetNearestWavenumber(1688) == 1690.0
    assert np.allclose(
        reader.GetSelectedWavenumbers(1688),
        [1690.0],
    )


def test_nearest_extracts_closest_channel():
    reader = m2.NRRDSpectrumReader(TEST_FILE)
    result = reader.GetArray(1688, squeeze=True)
    expected = _raw_data()[0, :, :, 1]

    assert np.allclose(result, expected)


def test_exact_extracts_existing_channel():
    reader = m2.NRRDSpectrumReader(TEST_FILE)
    reader.SetSpectralSelection("Exact")

    result = reader.GetArray(1650, squeeze=True)
    expected = _raw_data()[0, :, :, 0]

    assert np.allclose(result, expected)


def test_exact_rejects_missing_channel():
    reader = m2.NRRDSpectrumReader(TEST_FILE)
    reader.SetSpectralSelection("Exact")

    with pytest.raises(ValueError, match="No exact spectral channel"):
        reader.GetArray(1652, squeeze=True)


def test_window_uses_configured_tolerance_and_pooling():
    reader = m2.NRRDSpectrumReader(TEST_FILE)
    reader.SetSpectralSelection("Window")
    reader.SetTolerance(25.0)
    reader.SetPooling("Mean")

    result = reader.GetArray(1670, squeeze=True)
    expected = np.mean(_raw_data()[0, :, :, :], axis=-1)

    assert np.allclose(
        reader.GetSelectedWavenumbers(1670),
        [1650.0, 1690.0],
    )
    assert np.allclose(result, expected)


def test_explicit_tolerance_keeps_legacy_window_behavior():
    reader = m2.NRRDSpectrumReader(TEST_FILE)
    reader.SetPooling("Maximum")

    result = reader.GetArray(1670, 25.0, squeeze=True)
    expected = np.max(_raw_data()[0, :, :, :], axis=-1)

    assert np.allclose(result, expected)


def test_explicit_selection_overrides_legacy_tolerance_behavior():
    reader = m2.NRRDSpectrumReader(TEST_FILE)

    result = reader.GetArray(
        1688,
        100.0,
        squeeze=True,
        selection="Nearest",
    )
    expected = _raw_data()[0, :, :, 1]

    assert np.allclose(result, expected)


def test_negative_tolerance_is_rejected():
    reader = m2.NRRDSpectrumReader(TEST_FILE)

    with pytest.raises(ValueError, match="Tolerance"):
        reader.SetTolerance(-1.0)
