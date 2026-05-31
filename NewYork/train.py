"""Alias for the NewYork live CityFlow SafeGAT training entrypoint."""

import sys

from train_cityflow import main, default_args


if __name__ == "__main__":
    main(default_args() + sys.argv[1:])
