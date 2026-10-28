import os
import warnings
warnings.filterwarnings("ignore", message=".*Ignoring cached namespace.*") # Stops annoying library version warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

DEFAULT_SESSION_ID = 715093703
def get_session(session_id=DEFAULT_SESSION_ID):
	output_dir = '/Users/lyn/Documents/UCSD Stuffs/DSC 180A/allensdk-data' # must be updated to a valid directory in your filesystem
	DOWNLOAD_COMPLETE_DATASET = True
	
	manifest_path = os.path.join(output_dir, "manifest.json")
	cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
	session_data = cache.get_session_data(session_id)
	return session_data

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