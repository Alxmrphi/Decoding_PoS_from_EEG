import numpy as np
import pandas as pd
import mne
import pickle
from pathlib import Path
from collections import Counter

lex_description = 'pos == "NOUN" or pos == "VERB" or pos == "ADJ" or pos == "PROPN" or pos == "ADV"'
gram_description = 'pos == "ADP" or pos == "DET" or pos == "PRON" or pos == "AUX" or pos == "SCONJ" or pos == "CCONJ"'

def round_to_half(x):
    return np.round(x*2) / 2

def round_to_quarter(x):
    return np.round(x*4) / 4

def apply_class(x):
    if x in ['NOUN', 'VERB', 'ADJ', 'PROPN', 'ADV']: return 'LEX'
    elif x in ['ADP', 'DET', 'PRON', 'AUX', 'SCONJ', 'CCONJ']: return 'GRAM'
    else: return 'OTHER'

epoch_folder = Path('extended_epochs')
epoch_files = list(epoch_folder.iterdir())
np.random.shuffle(epoch_files)

n_epoch_files = len(epoch_files)

metadata_list = []
for epoch_file in epoch_files:
    
    print(f'Currently on: {str(epoch_file)}')
    tmp_epochs = mne.read_epochs(str(epoch_file), preload=False)
    metadata_list.append(tmp_epochs.metadata.copy())

dev_files = pickle.load(open('dev_files.pkl','rb'))
test_files = pickle.load(open('test_files.pkl','rb'))

def set_data_label(row):
    
    # Check mapping dictionaries are loaded / accessible
    assert 'dev_files' in globals(), "dev_files not loaded"
    assert 'test_files' in globals(), "test_files not loaded"
    
    # sess here is iteration of corpus acquisition
    row_filename = row['filename']
    row_sess = row['sess']
    
    if row_filename in dev_files and dev_files[row_filename] == row_sess:
        return 'dev'
    
    elif row_filename in test_files and test_files[row_filename] == row_sess:
        return 'test'
    
    else:
        return 'train'
    
def set_sent_pos_level(row):

    if row['prev_pos'] == "SENT_START": return 'init'
    elif row['next_pos'] == "SENT_END": return 'end'
    elif row['prev_pos'] != "SENT_START" and row['next_pos'] != "SENT_END":
        return "middle"
    else: return 'other'
    
def set_freq_level(row):
    return "LF" if int(row['freq']) <= 5.91 else "HF"

def set_sess_sent_num_level(row):
    return str(row['word']) + str(row['sess']) + str(row['filename']) + str(row['sent_num'])

new_metadata = pd.concat(metadata_list, axis=0)
new_metadata['freq2'] = new_metadata['freq'].apply(np.round)
new_metadata['wordclass'] = new_metadata['pos'].apply(apply_class)
new_metadata['train_dev_test'] = new_metadata.apply(set_data_label, axis=1)
new_metadata['freq_level'] = new_metadata.apply(set_freq_level, axis=1)
new_metadata['sent_pos'] = new_metadata.apply(set_sent_pos_level, axis=1)
new_metadata['sess_sent_num'] = new_metadata.apply(set_sess_sent_num_level, axis=1)
train_meta = new_metadata.query('train_dev_test == "train"')
dev_meta = new_metadata.query('train_dev_test == "dev"')
test_meta = new_metadata.query('train_dev_test == "test"')

total = 0
train_counter = Counter()

train_metadata = new_metadata.query('train_dev_test == "train"')

for sent_pos in ['init', 'middle', 'end']:
    for tmp_len in range(1,9,1):
        for wordclass in ['LEX', 'GRAM']:
            
            if sent_pos == 'init':
                position = train_metadata.query('prev_pos == "SENT_START"')
            elif sent_pos == 'end':
                position = train_metadata.query('next_pos == "SENT_END"')
            else:
                position = train_metadata.query('prev_pos != "SENT_START"')
                position = position.query('next_pos != "SENT_END"')
            
            tmp_res = position.query(f'wordclass == "{wordclass}" and len == {tmp_len}')
            tmp_hf = tmp_res.query(f'freq > 5.91')
            tmp_lf = tmp_res.query(f'freq <= 5.91')

            min_val = min(len(tmp_lf), len(tmp_hf))

            train_counter[(tmp_len, wordclass, sent_pos)] = min_val
            print(f'counter[({tmp_len},{wordclass}, {sent_pos})] = {min_val}')

            total += min_val

print(f'total = {total*2}')

total = 0
dev_counter = Counter()

dev_metadata = new_metadata.query('train_dev_test == "dev"')

for sent_pos in ['init', 'middle', 'end']:
    for tmp_len in range(1,9,1):
        for wordclass in ['LEX', 'GRAM']:
            
            if sent_pos == 'init':
                position = dev_metadata.query('prev_pos == "SENT_START"')
            elif sent_pos == 'end':
                position = dev_metadata.query('next_pos == "SENT_END"')
            else:
                position = dev_metadata.query('prev_pos != "SENT_START"')
                position = position.query('next_pos != "SENT_END"')

            tmp_res = position.query(f'wordclass == "{wordclass}" and len == {tmp_len}')
            tmp_hf = tmp_res.query(f'freq > 5.91')
            tmp_lf = tmp_res.query(f'freq <= 5.91')

            min_val = min(len(tmp_lf), len(tmp_hf))

            dev_counter[(tmp_len, wordclass, sent_pos)] = min_val
            print(f'counter[({tmp_len},{wordclass}, {sent_pos})] = {min_val}')

            total += min_val
print(f'total = {total*2}')

total = 0
test_counter = Counter()

test_metadata = new_metadata.query('train_dev_test == "test"')

for sent_pos in ['init', 'middle', 'end']:
    for tmp_len in range(1,9,1):
        for wordclass in ['LEX', 'GRAM']:
            
            if sent_pos == 'init':
                position = test_metadata.query('prev_pos == "SENT_START"')
            elif sent_pos == 'end':
                position = test_metadata.query('next_pos == "SENT_END"')
            else:
                position = test_metadata.query('prev_pos != "SENT_START"')
                position = position.query('next_pos != "SENT_END"')
     
            tmp_res = position.query(f'wordclass == "{wordclass}" and len == {tmp_len}')
            tmp_hf = tmp_res.query(f'freq > 5.91')
            tmp_lf = tmp_res.query(f'freq <= 5.91')

            min_val = min(len(tmp_lf), len(tmp_hf))

            test_counter[(tmp_len, wordclass, sent_pos)] = min_val
            print(f'counter[({tmp_len},{wordclass}, {sent_pos})] = {min_val}')

            total += min_val

print(f'total = {total*2}')

# This tracking counter is aware of the number of elemenbts in a (len, freq) pairing
# that have already been extracted so we don't go over and select too many

for subset, counter in [('test', test_counter) , ('dev', dev_counter), ('train', train_counter)]:

    print('subset = ', subset)
    
    tracking_counter_hf = Counter()
    tracking_counter_lf = Counter()

    to_concat = []

    epoch_files = list(epoch_folder.iterdir())
    assert len(epoch_files) == n_epoch_files, "Error: Iterating through different number of files"
    np.random.shuffle(epoch_files)

    for epoch_file in epoch_files:
        
        print(f'Loading: {str(epoch_file)}')
        tmp_epochs = mne.read_epochs(str(epoch_file), preload=False)
        print('len(tmp_epochs) = ', len(tmp_epochs))
        #tmp_epochs.metadata['freq2'] = tmp_epochs.metadata['freq'].apply(round_to_quarter)
        tmp_epochs.metadata['wordclass'] = tmp_epochs.metadata['pos'].apply(apply_class)
        tmp_epochs.metadata['train_dev_test'] = tmp_epochs.metadata.apply(set_data_label, axis=1)
        #tmp_epochs.metadata['sent_pos'] = tmp_epochs.metadata.apply(set_sent_pos_level, axis=1)
        #tmp_epochs.metadata['freq_level'] = tmp_epochs.metadata.apply(set_freq_level, axis=1)
        
        tmp_epochs = tmp_epochs[f'train_dev_test == "{subset}"']
        tmp_epochs.metadata.reindex(np.random.permutation(tmp_epochs.metadata.index))

        for (tmp_len, wordclass, sent_pos) in list(counter.keys()):
            print(f'tmp_len = {tmp_len}, wordclass == {wordclass}, sent_pos = {sent_pos}')
            min_val = counter[(tmp_len, wordclass, sent_pos)]
            print(f'I want to extract {min_val} elements here')
            
            if sent_pos == "init":
                lf = tmp_epochs[f'wordclass == "{wordclass}" and len == {tmp_len} and freq <= 5.91 and prev_pos == "SENT_START"']
                hf = tmp_epochs[f'wordclass == "{wordclass}" and len == {tmp_len} and freq > 5.91 and prev_pos == "SENT_START"']
            elif sent_pos == "end":
                lf = tmp_epochs[f'wordclass == "{wordclass}" and len == {tmp_len} and freq <= 5.91 and next_pos == "SENT_END"']
                hf = tmp_epochs[f'wordclass == "{wordclass}" and len == {tmp_len} and freq > 5.91 and next_pos == "SENT_END"']
            else:
                lf = tmp_epochs[f'wordclass == "{wordclass}" and len == {tmp_len} and freq <= 5.91 and prev_pos != "SENT_START"']
                lf = lf['next_pos != "SENT_END"']
                hf = tmp_epochs[f'wordclass == "{wordclass}" and len == {tmp_len} and freq > 5.91 and prev_pos != "SENT_START"']
                hf = hf['next_pos != "SENT_END"']
            
            #hf = tmp_epochs[f'freq > 5.91']
            #lf = tmp_epochs[f'freq <= 5.91']
                
            print(f'len(hf) = {len(hf)}')
            print(f'len(lf) = {len(lf)}')

            # In this case, don't restrict the amount of data for freq because it is regression
            # But need to make sure lex / gram is balanced

            # See how many needed to make it to min_val based on what we have already
            how_many_hf_already = tracking_counter_hf[(tmp_len, wordclass, sent_pos)]
            how_many_lf_already = tracking_counter_lf[(tmp_len, wordclass, sent_pos)]

            print(f'how many hf already collected? : {how_many_hf_already}')
            print(f'how many lf already collected? : {how_many_lf_already}')


            if how_many_hf_already + len(hf) <= min_val:
                how_many_hf_to_extract = len(hf)
            else:
                how_many_hf_to_extract = (min_val - how_many_hf_already)

            print(f'how many hf to extract? : {how_many_hf_to_extract}')

            if (min_val - how_many_hf_already) == 0:
                print('Already have enough hf!')

            if how_many_lf_already + len(lf) <= min_val:
                how_many_lf_to_extract = len(lf)
            else:
                how_many_lf_to_extract = (min_val - how_many_lf_already)

            print(f'how many lf to extract? : {how_many_lf_to_extract}')

            if (min_val - how_many_lf_already) == 0:
                print('Already have enough lf')

            if len(hf) > 0:
                hf = hf[:how_many_hf_to_extract]
                tracking_counter_hf[(tmp_len, wordclass, sent_pos)] += how_many_hf_to_extract
                print(f'updated hf counter to: {tracking_counter_hf[(tmp_len, wordclass, sent_pos)]}')
                to_concat.append(hf)

            if len(lf) > 0:
                lf = lf[:how_many_lf_to_extract]
                tracking_counter_lf[(tmp_len, wordclass, sent_pos)] += how_many_lf_to_extract
                print(f'updated lf counter to: {tracking_counter_lf[(tmp_len, wordclass, sent_pos)]}')
                to_concat.append(lf)  

    to_concat_filtered = [e for e in to_concat if len(e) > 0]
    epochs_concat = mne.concatenate_epochs(to_concat_filtered)
    epochs_concat.crop(0,0.7)
    epochs_concat.baseline = (0, 0)
    epochs_concat.save(f'freq_matched_epochs_{subset}_noica-epo.fif', overwrite=True)
