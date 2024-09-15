DEFAULT_CUTOFFS = {
    "CHGNetNFF": 6.0,
    "NffScaleMACE": 5.0,
    "PaiNN": 5.0,
}

DEFAULT_SAMPLING_SETTINGS = {
    "canonical": False,
    "total_sweeps": 100,
    "sweep_size": 20,
    "start_temp": 1.0,  # in terms of kT
    "perform_annealing": False,
    "alpha": 1.0,
}
