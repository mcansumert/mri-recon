from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

setup(
    name="sense_kernel",
    ext_modules=[
        Pybind11Extension(
            "sense_kernel",
            ["sense_kernel.cpp"],
            cxx_std=17,
        )
    ],
    cmdclass={"build_ext": build_ext},
)