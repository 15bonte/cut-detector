# cut_detector

[![License BSD-3](https://img.shields.io/pypi/l/cut_detector.svg?color=green)](https://github.com/15bonte/cut_detector/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/cut_detector.svg?color=green)](https://pypi.org/project/cut_detector)
[![Python Version](https://img.shields.io/pypi/pyversions/cut_detector.svg?color=green)](https://python.org)
[![tests](https://github.com/15bonte/cut_detector/workflows/tests/badge.svg)](https://github.com/15bonte/cut_detector/actions)
[![codecov](https://codecov.io/gh/15bonte/cut_detector/branch/main/graph/badge.svg)](https://codecov.io/gh/15bonte/cut_detector)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/cut_detector)](https://napari-hub.org/plugins/cut_detector)

Automatic Cut Detector

---

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

## Installation

You can install `cut_detector` via [pip]:

    pip install cut_detector

To install latest development version :

    pip install git+https://github.com/15bonte/cut_detector.git

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"cut_detector" is free and open source software

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
[file an issue]: https://github.com/15bonte/cut_detector/issues
[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/

conda create -n global python=3.9
pip install cellpose
pip uninstall torch

conda install pytorch==1.12.1 torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
