"""Setup script for sktime-cython Cython extensions."""

import sys

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

# fast-math is the bulk of the speedup vs numba; flags are platform-specific.
if sys.platform == "win32":
    _fast = ["/O2", "/fp:fast"]
else:
    _fast = ["-O3", "-ffast-math"]

_MODULES = [
    "_minirocket_multivariate_cython",
    "_multirocket_multivariate_cython",
]

extensions = [
    Extension(
        f"sktime_cython.transformations.rocket.{name}",
        sources=[f"sktime_cython/transformations/rocket/{name}.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=_fast,
    )
    for name in _MODULES
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
    ),
)
