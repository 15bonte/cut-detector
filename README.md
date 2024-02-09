# Cut Detector

[![License BSD-3](https://img.shields.io/pypi/l/cut-detector.svg?color=green)](https://github.com/15bonte/cut-detector/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/cut-detector.svg?color=green)](https://pypi.org/project/cut-detector)
[![Python Version](https://img.shields.io/pypi/pyversions/cut-detector.svg?color=green)](https://python.org)
[![tests](https://github.com/15bonte/cut-detector/workflows/tests/badge.svg)](https://github.com/15bonte/cut-detector/actions)
[![codecov](https://codecov.io/gh/15bonte/cut-detector/branch/main/graph/badge.svg)](https://codecov.io/gh/15bonte/cut-detector)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/cut-detector)](https://napari-hub.org/plugins/cut-detector)

Automatic micro-tubules cut detector.

---

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

## Installation

### Conda environment

It is highly recommended to create a dedicated conda environment, by following these few steps:

1. Install an [Anaconda] distribution of Python. Note you might need to use an anaconda prompt if you did not add anaconda to the path.

2. Open an Anaconda prompt as admin to create a new environment using [conda]. We advice to use python 3.10 and conda 23.10.0, to get conda-libmamba-solver as default solver.

```
conda create --name cut_detector python=3.10 conda=23.10.0
conda activate cut_detector
```

### Package installation

Once in a dedicated environment, our package can be installed via [pip]:

```
pip install cut_detector
```

Alternatively, you can clone the github repo to access to playground scripts.

```
git clone https://github.com/15bonte/cut-detector.git
cd cut-detector
pip install -e .
```

### Fiji

This package relies on [Trackmate] to perform cell tracking. Trackmate is called through [Fiji], which has to be installed independently. Please follow the steps [here](https://imagej.net/software/fiji/downloads) to install it.

Next, install openjdk which is necessary to call Fiji from python.

```
conda install -c conda-forge openjdk=8
```

### GPU

We highly recommend to use GPU to speed up segmentation. To use your NVIDIA GPU, the first step is to download the dedicated driver from [NVIDIA].

Next we need to remove the CPU version of torch:

```
pip uninstall torch
```

The GPU version of torch to be installed can be found [here](https://pytorch.org/get-started/locally/). You may choose the CUDA version supported by your GPU, and install it with conda. This package has been developed with the version 11.6, installed with this command:

```
conda install numpy==1.25 pytorch==1.12.1 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
```

Note that we have added numpy here to prevent conda from installing a version higher than 1.25, which is not supported by numba.

## Update

To update cut-detector to the latest version, open an Anaconda prompt and use the following commands:

```
conda activate cut_detector
pip install cut-detector --upgrade
```

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"cut-detector" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin
[file an issue]: https://github.com/15bonte/cut-detector/issues
[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
[Anaconda]: https://www.anaconda.com/products/distribution
[Trackmate]: https://imagej.net/plugins/trackmate/
[Fiji]: https://imagej.net/software/fiji/
[NVIDIA]: https://www.nvidia.com/Download/index.aspx?lang=en-us
[conda]: https://docs.conda.io/en/latest/
