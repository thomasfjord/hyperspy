# -*- coding: utf-8 -*-
# Copyright 2007-2022 The HyperSpy developers
#
# This file is part of HyperSpy.
#
# HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HyperSpy. If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import pytest

from hyperspy import signals
from hyperspy.misc.utils import (
    is_hyperspy_signal,
    parse_quantity,
    slugify,
    strlist2enumeration,
    is_binned,
    is_cupy_array,
    to_numpy,
    get_array_module
)
from hyperspy.exceptions import VisibleDeprecationWarning

try:
    import cupy as cp
    CUPY_INSTALLED = True
except ImportError:
    CUPY_INSTALLED = False

skip_cupy = pytest.mark.skipif(not CUPY_INSTALLED, reason="cupy is required")


def test_slugify():
    assert slugify("a") == "a"
    assert slugify("1a") == "1a"
    assert slugify("1") == "1"
    assert slugify("a a") == "a_a"

    assert slugify("a", valid_variable_name=True) == "a"
    assert slugify("1a", valid_variable_name=True) == "Number_1a"
    assert slugify("1", valid_variable_name=True) == "Number_1"

    assert slugify("a", valid_variable_name=False) == "a"
    assert slugify("1a", valid_variable_name=False) == "1a"
    assert slugify("1", valid_variable_name=False) == "1"


def test_parse_quantity():
    # From the metadata specification, the quantity is defined as
    # "name (units)" without backets in the name of the quantity
    assert parse_quantity("a (b)") == ("a", "b")
    assert parse_quantity("a (b/(c))") == ("a", "b/(c)")
    assert parse_quantity("a (c) (b/(c))") == ("a (c)", "b/(c)")
    assert parse_quantity("a [b]") == ("a [b]", "")
    assert parse_quantity("a [b]", opening="[", closing="]") == ("a", "b")


def test_is_hyperspy_signal():
    s = signals.Signal1D(np.zeros((5, 5, 5)))
    p = object()
    assert is_hyperspy_signal(s) is True
    assert is_hyperspy_signal(p) is False


def test_strlist2enumeration():
    assert strlist2enumeration([]) == ""
    assert strlist2enumeration("a") == "a"
    assert strlist2enumeration(["a"]) == "a"
    assert strlist2enumeration(["a", "b"]) == "a and b"
    assert strlist2enumeration(["a", "b", "c"]) == "a, b and c"


# Can be removed in v2.0:
def test_is_binned():
    s = signals.Signal1D(np.zeros((5, 5)))
    assert is_binned(s) == s.axes_manager[-1].is_binned
    with pytest.warns(VisibleDeprecationWarning, match="Use of the `binned`"):
        s.metadata.set_item('Signal.binned', True)
    assert is_binned(s) == s.metadata.Signal.binned


@skip_cupy
def test_is_cupy_array():
    cp_array = cp.array([0, 1, 2])
    np_array = np.array([0, 1, 2])
    assert is_cupy_array(cp_array)
    assert not is_cupy_array(np_array)


@skip_cupy
def test_to_numpy():
    cp_array = cp.array([0, 1, 2])
    np_array = np.array([0, 1, 2])
    np.testing.assert_allclose(to_numpy(cp_array), np_array)
    np.testing.assert_allclose(to_numpy(np_array), np_array)


@skip_cupy
def test_get_array_module():
    cp_array = cp.array([0, 1, 2])
    np_array = np.array([0, 1, 2])
    assert get_array_module(cp_array) == cp
    assert get_array_module(np_array) == np
