# Pre-sentence baseline correction
# - Two main functions
# -- 1) generate_baseline_dict
# -- 2) apply_baseline

# (1) Explanation:
# - A folder exists with all the mne.Epochs files for each session
# - Loop through this list, each time:
# - Extract all the trials where the previous PoS tag id indicated
# - as "SENT_START", i.e. the first trial of each sentence. 
# - Collect these trials and concatenate them and extract corpus-wide
# - metadata. Loop over each sentence and store, for that sent_ident
# - value, the average of the prior 200 ms before the sentence began.
# - Store / save this value to be applied in the apply_baseline step.

# (2) Explanation:
# - Using the generated baseline dict


import numpy as np
import pandas as pd
import mne
import pickle
from pathlib import Path
from collections import Counter
import random

# Examples
epoch_dir = './extended_epochs_minICA'
write_filename = 'baseline_dict_minICA.pkl'

def generate_baseline_dict(epoch_dir, baseline_filename) -> None:

    epoch_folder = Path(epoch_dir)
    epoch_files = list(epoch_folder.iterdir())
    np.random.shuffle(epoch_files)
    to_concat = []

    for epoch_file in epoch_files:
        tmp_epochs = mne.read_epochs(str(epoch_file), preload=False)
        
        query = 'prev_pos == "SENT_START"'
        res = tmp_epochs[query]
        to_concat.append(res)

    to_concat_filtered = [e for e in to_concat if len(e) > 0]
    epochs_concat = mne.concatenate_epochs(to_concat_filtered)
    md = epochs_concat.metadata
    # print(epochs_concat)

    baseline_dict = {}

    # Each sentence in the corpus is associated with a unique identifier 'sent_ident'.
    for sent_id in list(set(md.sent_ident)):
        # Find the epoch for the first word of each sentence
        tmp = epochs_concat[f'sent_ident == "{sent_id}"']
        # print(sent_id)
        tmp_data = tmp.get_data()
        assert tmp_data.shape[0] == 1, f"Should have only found 1 corresponding sentence, \
            instead found multiple for {sent_id} with shape[0] == {tmp_data.shape[0]}"
        baseline_period = tmp_data[0,:,0:50] # prior 200 ms before word onset
        baseline_avg = np.mean(baseline_period, axis=1) # along temporal axis
        assert len(baseline_avg) == 64, "Need 64 dim baseline average"
        baseline_dict[sent_id] = baseline_avg
    
    # Write out results
    pickle.dump(baseline_dict, open(baseline_filename, 'wb'))


def apply_baseline(epoch_dir, baseline_filename) -> None:

    baseline = pickle.load(open(baseline_filename, 'rb'))

    epoch_folder = Path(epoch_dir)
    epoch_files = list(epoch_folder.iterdir())
    np.random.shuffle(epoch_files)

    for epoch_file in epoch_files:

        #if "13-20" not in str(epoch_file):
        #    continue

        tmp_epochs = mne.read_epochs(str(epoch_file), preload=True)
        tmp_sent_ids = tmp_epochs.metadata.sent_ident
        tmp_set_sent_ids = list(set(tmp_sent_ids))

        idx_checker = []
        error_counter = []

        for tmp_sent_id in tmp_set_sent_ids:
            # These are the elements of the corresponding sentence
            idx = tmp_sent_ids[tmp_sent_ids == tmp_sent_id].index
            # For indexing into base data
            start, end = idx[0], idx[-1]+1
            # Track how many idx are processed
            idx_checker.extend(idx)
            # Pull out baseline correction for this sent_id
            #assert tmp_sent_id in baseline.keys(), f"Not found baseline for {tmp_sent_id}"
            if tmp_sent_id not in baseline.keys():
                error_counter.append([tmp_sent_id])
                continue
            else:        
                tmp_baseline = baseline[tmp_sent_id]

            # Loop through EEG channels and baseline correct in-place
            for i in range(len(tmp_epochs.ch_names)):
                x = tmp_epochs._data
                assert id(x) == id(tmp_epochs._data), "id mismatch"
                x[start:end,i,:] = x[start:end,i,:] - tmp_baseline[i]

            # assert len(idx_checker) == len(tmp_epochs), "Not all events baseline-corrected"

        new_filename = str(epoch_file)[:-10] + str('bc_epo.fif')
        tmp_epochs.save(new_filename, overwrite=True)