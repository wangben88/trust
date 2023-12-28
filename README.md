# TRUST: Tractable Uncertainty for Structure Learning

## Installation Instructions

Install Poetry (python-poetry.org), then run the following commands from the base directory:

    poetry build
    poetry install

To run code, first activate the environment using

    poetry shell

NEW: There were some compatibility issues with MacOS with the previous dependencies. I've removed the
problematic ones, but you will now need to install them manually (after running poetry shell):

    pip install torch
    pip install torch-scatter