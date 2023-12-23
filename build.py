import os
from pybind11.setup_helpers import Pybind11Extension, build_ext


def build(setup_kwargs):
    ext_modules = [
        Pybind11Extension("LeafScore", ["trust/leaf_scores/LeafScore.cpp"]),
    ]
    setup_kwargs.update({
        "ext_modules": ext_modules,
        "cmd_class": {"build_ext": build_ext},
        "zip_safe": False,
    })
