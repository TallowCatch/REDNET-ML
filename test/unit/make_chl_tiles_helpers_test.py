from __future__ import annotations

from types import ModuleType

import numpy as np
import pytest
import xarray as xr


@pytest.fixture(scope="module")
def chl_tiles_module(module_loader) -> ModuleType:
    return module_loader("src/tools/make_chl_tiles_from_nc.py")


@pytest.mark.unit
def test_to_uint8_gray_scaling_and_clipping(chl_tiles_module: ModuleType) -> None:
    arr = np.array([[-1.0, 0.0, 5.0, 10.0, 20.0]], dtype=float)
    out = chl_tiles_module.to_uint8_gray(arr, vmin=0.0, vmax=10.0)

    assert out.dtype == np.uint8
    assert out.tolist()[0] == [0, 0, 127, 255, 255]


@pytest.mark.unit
def test_find_chl_var_prefers_chlorophyll_variable(chl_tiles_module: ModuleType) -> None:
    ds = xr.Dataset(
        data_vars={
            "palette": (("lat", "lon"), np.zeros((2, 2))),
            "chlor_a": (("lat", "lon"), np.ones((2, 2))),
        },
        coords={"lat": [1.0, 2.0], "lon": [50.0, 51.0]},
    )

    assert chl_tiles_module.find_chl_var(ds) == "chlor_a"


@pytest.mark.unit
def test_find_chl_var_returns_none_if_not_present(chl_tiles_module: ModuleType) -> None:
    ds = xr.Dataset(
        data_vars={"sst": (("lat", "lon"), np.ones((2, 2)))},
        coords={"lat": [1.0, 2.0], "lon": [50.0, 51.0]},
    )

    assert chl_tiles_module.find_chl_var(ds) is None


@pytest.mark.unit
def test_find_lon_lat_names_for_standard_and_long_names(chl_tiles_module: ModuleType) -> None:
    ds_std = xr.Dataset(
        data_vars={"x": (("lat", "lon"), np.ones((2, 2)))},
        coords={"lat": [1.0, 2.0], "lon": [50.0, 51.0]},
    )
    ds_long = xr.Dataset(
        data_vars={"x": (("latitude", "longitude"), np.ones((2, 2)))},
        coords={"latitude": [1.0, 2.0], "longitude": [50.0, 51.0]},
    )

    assert chl_tiles_module.find_lon_lat_names(ds_std) == ("lon", "lat")
    assert chl_tiles_module.find_lon_lat_names(ds_long) == ("longitude", "latitude")
