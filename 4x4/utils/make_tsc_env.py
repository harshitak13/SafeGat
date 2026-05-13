"""
utils/make_tsc_env.py

Factory function that builds a single-junction SUMO TSCEnvironment.
Called by make_grid_env() in envs/grid_env_wrapper.py once per TLS.

Install TransSimHub (tshub):
    git clone https://github.com/Traffic-Alpha/TransSimHub.git
    cd TransSimHub && pip install -e .

Source: iLLM-TSC2 (utils/make_tsc_env.py).
"""

from __future__ import annotations

import os
from tshub.tsc.tsc_environment import TSCEnvironment


def make_env(
    tls_id:      str,
    sumo_cfg:    str,
    num_seconds: int  = 3600,
    use_gui:     bool = False,
    log_file:    str  = "./log/",
    obs_dim:     int  = 8,
    trip_info:   str  = None,
) -> TSCEnvironment:
    """
    Create and return a TSCEnvironment for a single traffic light.

    Parameters
    ----------
    tls_id      : junction ID in the SUMO net, e.g. "J1"
    sumo_cfg    : absolute path to the .sumocfg file
    num_seconds : simulation duration in seconds
    use_gui     : True → sumo-gui, False → headless sumo
    log_file    : directory for SUMO log output (unused by TSCEnvironment directly)
    obs_dim     : observation dimension (passed through; obs is built in wrapper)
    trip_info   : optional path for SUMO tripinfo output XML

    Returns
    -------
    TSCEnvironment instance (NOT yet reset — caller calls env.reset())
    """
    net_file = os.path.join(os.path.dirname(sumo_cfg), "4x4.net.xml")

    kwargs = dict(
        sumo_cfg        = sumo_cfg,
        net_file        = net_file,
        num_seconds     = num_seconds,
        tls_id          = tls_id,
        tls_action_type = "choose_next_phase",  # action = phase index 0–3
        use_gui         = use_gui,
    )
    if trip_info is not None:
        kwargs["trip_info"] = trip_info

    return TSCEnvironment(**kwargs)
