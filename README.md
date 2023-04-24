# Applied Optimization Methods for Inverse Problems

Welcome to the project for the practical course "Applied Optimization Method
for Inverse Problems" offered during the summer term 2023 at TUM. This is
the basis of the practical work and homework for the course.

The course website can be found [here](https://ciip.in.tum.de/teaching/aom-ip_ss23.html)

## Getting started

#### Poetry

The easiest and recommended way to install is using `poetry` (see
[here](https://python-poetry.org/)). Once you've installed `poetry`, 
run `poetry install` from the root directory.

Next, we miss the dependency on `elsa` (see
[here](https://gitlab.lrz.de/IP/elsa)), our tool for tomographic
reconstruction. First run `poetry shell`, which will activate the virtual
environment created by `poetry`, then clone `elsa` to a new directory,
move into the directory and run `pip install . --verbose` (the `--verbose` is
optional, but then you'll see the progress).

From now you can either active the virtual environment by running `poetry shell`,
or you run `poetry run python myscript`, which will active the environment for
that single command.

#### Classic

If you do not want to use `poetry`, you can use virtual environments.
From the root directory of this repository run `python -m venv /path/to/venv`, active
it using `source /path/to/venv/bin/activate`, and then install everything with
`pip install --editable .` (from the root directory).

Then again you need to install `elsa`. Follow the steps described above in
the `poetry` section.

### Troubleshooting

If you have trouble installing `elsa` (how can that happen :D), see the README
of `elsa`. If you use an Ubuntu based distro and want to use CUDA, you might
need to set `CUDA_HOME`, to wherever CUDA is installed.

Please note, that you do not need CUDA, but it might speed up your
reconstructions quite dramatically.

### Access to CUDA capable hardware

As generally, it might make certain things more pleasant, you can get access
to one of our CUDA capable machines.

Please, send us a public key (see
[this](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)
for instructions to create one), and your RBG username. Then we will arange
access to our machines. To access them you need to be either in a
university WiFi (e.g. eduroam), or use a VPN (see
[here](https://vpn.rbg.tum.de/) for configuration files).

There you need to clone your personal repository, do the same setup steps as
described above.

Please note: You are sharing the resource with all of your students, so please
do not block all GPU's all the time! Check which GPU's are currently free
using `nvidia-smi`, and then use `CUDA_VISIBLE_DEVICES=id` to only
use one (and not the first).

## Getting the data for the Helsinki Tomography Challenge

To get the dataset for the challenge, head over to
[Zenodo](https://doi.org/10.5281/zenodo.7418878) and download the
`htc2022_test_data.zip`. Extract it to a folder and you should be good to go.
In the folder you will see a couple of different files.

`.mat` files contain the actual measurements/sinogram, which will need for
reconstruction. There are the full measurements, one with limited number of
projections and example reconstructions of both. Further, there are segmented
files, which show the required binary thresholding segmentation done to
evaluate the score (again for full data and limited). Finally, there are a
couple of example images.

Then run `main.py` in the `challenge` subfolder, with the required argument
of the path to the folder you just extracted.
