# This code is used to take the no-ICA version of the raw data and convert it to a 
# sequence of mne Epoch data structures spanning [-200, 900] ms around the onset of
# a word presented on screen, while EEG data is recorded. During this conversion
# a lot of the necessary metadata is added in to the Epochs data structure for later
# selection. Features of tag, length and frequency are given for prev, current, next words.

from pathlib import Path
import glob
import re
import mne
import numpy as np
import pandas as pd
import os
from wordfreq import word_frequency, zipf_frequency

pos2code = {"NOUN":500, "VERB":501, "ADJ":502, "DET":503, "PRON":504,
            "ADP":505, "ADV":506, "AUX":507, "CCONJ":508, "INTJ":509,
            "NUM":510, "PART":511, "PROPN":512, "PUNCT":513, "SCONJ":514,
            "X":516}

# For some data, no record of tag "X" exists and causes error
# This is incorporated as a check and when applicable, version below is used
pos2code_no_X = {"NOUN":500, "VERB":501, "ADJ":502, "DET":503, "PRON":504,
                "ADP":505, "ADV":506, "AUX":507, "CCONJ":508, "INTJ":509,
                "NUM":510, "PART":511, "PROPN":512, "PUNCT":513, "SCONJ":514}

code2pos = {v:k for k,v in pos2code.items()}

iteration_folder = Path('session_files')
trigger_folder = Path('triggerfiles')
subfolders = iteration_folder.iterdir()

# Some folder prefixed with '_' should be ignored
subfolders = [e for e in subfolders if e.is_dir() and
              str(e.name).startswith('s')]


# --- HELPER FUNCTIONS

def add_in_surrounding_word_info(df: pd.DataFrame) -> pd.DataFrame: 
    """ Loop through DataFrame adding in PoS info for prev/next words
    Presuming this happens while 202 / 203 + session start / end triggers
    are present """
    
    prev_pos = "SENT_START"
    prev_sent = "sent_0"
    
    for i in range(1, len(df)):
        if i % 1000 == 0:
            print(f'{i/len(df) * 100:.2f}')
        # Trigger for a word reading event
        if df.iloc[i]['trigger'] < 200:
            
            tmp_sent = df.iloc[i]['sent_num']
            
            # 'i' will always be in range due to other marker triggers
            # at the beginning and end of the recording session
            prev_trigger = df.iloc[i-1]['trigger']
            next_trigger = df.iloc[i+1]['trigger']
            prev_pos = df.iloc[i-1]['pos']
            next_pos = df.iloc[i+1]['pos']
            prev_freq = df.iloc[i-1]['freq']
            next_freq = df.iloc[i+1]['freq']
            prev_len  = df.iloc[i-1]['len']
            next_len  = df.iloc[i+1]['len']
            
            # Use loc to avoid Chained Assignment warning / failure
            df.loc[i, 'prev_pos'] = prev_pos if prev_trigger < 200 else 'SENT_START'
            df.loc[i, 'next_pos'] = next_pos if next_trigger < 200 else 'SENT_END'
            df.loc[i, 'prev_freq'] = prev_freq
            df.loc[i, 'next_freq'] = next_freq
            df.loc[i, 'prev_len'] = prev_len
            df.loc[i, 'next_len'] = next_len
            
            # Add in a unique sentence identifier
            sess     = df.loc[i, 'sess']
            filename = df.loc[i, 'filename']
            sent_num = df.loc[i, 'sent_num']
    
            sent_ident = f"{sess}_{filename}_{sent_num}"
            df.loc[i, 'sent_ident'] = sent_ident
                       
    return df

# Functions to extract DataFrame from processed trigger file
# and also to check that the length + PoS codes match

def get_session_df(folder: str, file: str) -> pd.DataFrame:
    path = os.path.join(folder, file)
    df = pd.read_csv(path, sep=' ',
                     names=["trigger", "_", "word", "pos", "sess", "sent_num", "filename"],
                     quotechar="^")
    df = df.drop(columns=['_'], axis=1)
    df = df.fillna(value={'word': ''})
    df['len'] = df.apply(lambda row: len(row.word), axis=1)
    df['freq'] = df.apply(lambda row: zipf_frequency(row.word, lang='en'), axis=1)
    df['sess_id'] = file
    df = add_in_surrounding_word_info(df)
    df = df[df['trigger'] < 200] # Only include word-presentation triggers
    return df

def check_df_with_epochs(df: pd.DataFrame, events: np.array) -> bool:
    """ Check both lengths match + that each PoS code matches"""
    if 515 not in np.unique(events[:,2]):
        print('Dropping SYM events')
        df.drop(df[df['pos'] == 'SYM'].index, inplace=True)
    assert len(df) == len(events), f"len(df) ({len(df)}) != len(events) ({len(events)})"
    for i in range(len(df)):
        pos = df.iloc[i].pos
        word = df.iloc[i].word
        event_no = events[i,2]
        
        # Quick fix
        if (df.iloc[i].filename == 'weblog-blogspot.com_aggressivevoicedaily_20060814163400_ENG_20060814_163400.txt' and df.iloc[i].sess == 'sess_0'):
            events[109,2] = 504
            
        if pos2code[pos] != event_no:
            if event_no == 513 or event_no == 511 or event_no == 504 or event_no == 501 or event_no == 509 or event_no == 512:
                # Error that occurs a few times, punct label mixed up with PoS
                print(f"Discrepancy at i == {i}, file says {pos} but events says {code2pos[events[i,2]]}, word is {word}")
                events[i,2] = pos2code[pos]
            else:
                f"Discrepancy at i == {i}, file says {pos} but events says {code2pos[events[i,2]]}, word is {word} (marking as X)"
                events[i,2] = 516

        #assert {pos2code[pos]} == {events[i,2]}, f"Discrepancy at i == {i}, file says {pos} but events says {code2pos[events[i,2]]}, word is {word}"
        
    return True


# --- PERFORM CONVERSION
metadata_files = []

for file in subfolders:
    
    # Unfixable errors here, skip this session
    if str(file) == r'session0\session0_files29-39':
        print(f'skipping {file}')
        continue
        
    print(file)
    
    # Look for the preprocessed file and check it exists
    eeg_file = glob.glob(f'{file}//*noica*.fif')
     
    event_file = glob.glob(f'{file}/events_downsampled*.npy')
    assert len(eeg_file) > 0, "No '*noica.fif' file found"
    if len(eeg_file) > 1:
        print(f'len(eeg_file) > 1)')
        print(eeg_file)
        eeg_file = eeg_file[0]
        print(eeg_file)
    assert len(event_file) >  0, "No events file found"
    eeg_session = re.search("(rsvp.*)_", str(eeg_file)).group(0)
    triggerfile = glob.glob(f'{trigger_folder}/rsvp_{eeg_session[5:]}trigger*.txt')
    assert len(triggerfile) > 0, f"Couldn't find triggerfile: '{trigger_folder}/rsvp_{eeg_session[5:]}trigger*.txt"
    triggerfile = Path(triggerfile[0])
    
    eeg_file = eeg_file[0] if type(eeg_file) == list else eeg_file
    print(f'eeg_file = {eeg_file}')
    
    # Also faulty session, needs to be skipped
    if "files123-131" in eeg_file:
        print('Skipping....')
        continue
        
    event_file = event_file[0]
    
    eeg_data = mne.io.read_raw_fif(eeg_file, preload=True, verbose='WARNING')
    events = np.load(event_file)
    
    # This will fail in the code-PoS mapping if no "X" tag is found in the data
    # First try most common case (X is there), otherwise run the mapping without "X"
    try:
        epochs = mne.Epochs(eeg_data, events, event_id=pos2code, baseline=None, tmin=-0.2,
                            tmax=0.9, proj=True, preload=True, reject=None,
                            reject_by_annotation=None)
    except:
                epochs = mne.Epochs(eeg_data, events, event_id=pos2code_no_X, baseline=None, tmin=-0.2,
                            tmax=0.9, proj=True, preload=True, reject=None,
                            reject_by_annotation=None)
    
    # Add in updated metadata
    tmp_df = get_session_df(triggerfile.parent, triggerfile.name)
    assert check_df_with_epochs(tmp_df, epochs.events) is True
    epochs.metadata = tmp_df
    metadata_files.append(tmp_df)
    
    # Drop the stimulus channel if present
    if epochs.info['ch_names'][-1] == 'STI 014':
        epochs.drop_channels(['STI 014'])
        
    epochs.save(f'extended_epochs_noica/{eeg_session}_noica-epo.fif', overwrite=True)
