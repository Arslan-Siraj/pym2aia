from pathlib import Path

import numpy as np
import SimpleITK as sitk

import m2aia as m2


TEST_FILE = (
    Path(__file__).parent
    / "data"
    / "small_mir.nrrd"
)


def test_load_nrrd():
    reader = m2.NRRDSpectrumReader(TEST_FILE)

    assert reader.GetNumberOfSpectra() > 0
    assert reader.GetXAxisDepth() == 2


def test_shape():
    reader = m2.NRRDSpectrumReader(TEST_FILE)

    assert np.array_equal(
        reader.GetShape(),
        [8, 6, 1],
    )


def test_x_axis():
    reader = m2.NRRDSpectrumReader(TEST_FILE)

    assert np.allclose(
        reader.GetXAxis(),
        [1650.0, 1690.0],
    )


def test_number_of_spectra():
    reader = m2.NRRDSpectrumReader(TEST_FILE)

    assert reader.GetNumberOfSpectra() == 48


def test_first_spectrum_matches_nrrd():
    reader = m2.NRRDSpectrumReader(TEST_FILE)

    image = sitk.ReadImage(str(TEST_FILE))
    data = sitk.GetArrayFromImage(image)

    xs, ys = reader.GetSpectrum(0)

    expected = data[0, 0, 0, :]

    assert np.allclose(
        xs,
        [1650.0, 1690.0],
    )

    assert np.allclose(
        ys,
        expected,
    )


def test_extract_1650_image():
    reader = m2.NRRDSpectrumReader(TEST_FILE)

    result = reader.GetArray(
        1650,
        0.1,
        squeeze=True,
    )

    image = sitk.ReadImage(str(TEST_FILE))
    data = sitk.GetArrayFromImage(image)

    expected = data[0, :, :, 0]

    assert result.shape == (6, 8)
    assert np.allclose(result, expected)


def test_extract_1690_image():
    reader = m2.NRRDSpectrumReader(TEST_FILE)

    result = reader.GetArray(
        1690,
        0.1,
        squeeze=True,
    )

    image = sitk.ReadImage(str(TEST_FILE))
    data = sitk.GetArrayFromImage(image)

    expected = data[0, :, :, 1]

    assert result.shape == (6, 8)
    assert np.allclose(result, expected)


def test_geometry():
    reader = m2.NRRDSpectrumReader(TEST_FILE)

    image = sitk.ReadImage(str(TEST_FILE))

    assert np.allclose(
        reader.GetSpacing(),
        image.GetSpacing(),
    )

    assert np.allclose(
        reader.GetOrigin(),
        image.GetOrigin(),
    )

    assert np.allclose(
        reader.GetDirection(),
        image.GetDirection(),
    )


def test_image_minmax_normalization():
    reader = m2.NRRDSpectrumReader(TEST_FILE)

    reader.SetImageNormalization("MinMax")

    result = reader.GetArray(
        1650,
        0.1,
        squeeze=True,
    )

    assert np.isclose(result.min(), 0.0)
    assert np.isclose(result.max(), 1.0)


def test_max_spectrum_normalization():
    reader = m2.NRRDSpectrumReader(TEST_FILE)

    reader.SetNormalization("Max")

    _, ys = reader.GetSpectrum(0)

    assert np.isclose(
        np.max(ys),
        1.0,
    )