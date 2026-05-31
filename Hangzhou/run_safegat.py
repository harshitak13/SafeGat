"""Alias for the Hangzhou live CityFlow SafeGAT inference entrypoint."""

import sys

from run_safegat_cityflow import main, default_args


if __name__ == "__main__":
    main(default_args() + sys.argv[1:])
