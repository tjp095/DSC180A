from pathlib import Path
import warnings
warnings.filterwarnings("ignore", message=".*Ignoring cached namespace.*") # Stops annoying library version warnings

import numpy as np
import pandas as pd

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

CACHE_DIR = Path(__file__).resolve().parent / "data" / "ecephys"
DEFAULT_SESSION_ID = 715093703
EDA_SESSION_ID = 750749662


def get_cache():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return EcephysProjectCache.from_warehouse(
        manifest=str(CACHE_DIR / "manifest.json")
    )


def get_session(session_id=DEFAULT_SESSION_ID):
    return get_cache().get_session_data(session_id)


def get_units(session, stim="drifting_gratings", region="VISp"):
	stim_presentations = session.get_stimulus_table(stim)
	units = session.units[session.units["ecephys_structure_acronym"] == region]
	return stim_presentations, units

def get_spikes(session, stim_presentations, units):
	spikes = session.presentationwise_spike_times(
		stimulus_presentation_ids=stim_presentations.index.values,
		unit_ids=units.index.values[:]
	)
	spikes = spikes.reset_index()
	return spikes

def clean_grating_spike_data(grating_spikes, grating_presentations, bin_size=0.2):
	# Combine spike information with grating orientation table
	grating_angles = grating_presentations["orientation"].reset_index()
	grate_angle_spikes = grating_spikes.merge(grating_angles, on="stimulus_presentation_id")
	spikes_clean = grate_angle_spikes[grate_angle_spikes["orientation"] != "null"]
	# Clean out any null values (not actually missing, just want actual grating presentations instead of null trials)
	spikes_clean = spikes_clean.dropna().reset_index(drop=True)

	# Add bin column
	bins = np.arange(0, 2.2, bin_size)
	spikes_clean['time_bin'] = pd.cut(
		spikes_clean['time_since_stimulus_presentation_onset'],
		bins=bins,
		right=False,
		include_lowest=True
	)

	return spikes_clean


if __name__ == "__main__":
    print(f"AllenSDK cache: {CACHE_DIR}", flush=True)
    cache = get_cache()
    cache.get_session_table()

    for session_id in (DEFAULT_SESSION_ID, EDA_SESSION_ID):
        print(f"Preparing session {session_id}...", flush=True)
        session = cache.get_session_data(session_id)
        print(f"Session {session_id} ready ({len(session.units)} units)", flush=True)
