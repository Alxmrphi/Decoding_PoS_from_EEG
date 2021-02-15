import tensorflow as tf
import numpy as np

# -- Data Selection & Conversion Settings
n_avg = 'avg10'
n_classes = 6
category = "bigram_6class"
save_misc = "_noica"
misc = ''
wr = '' # or "_wr" if NumPy data were averaged without-replacement


def _float_feature_seq(eeg_channel):
    """ Convert sequence of EEG values to tf.train.FeatureList """
    feature_list = tf.train.FeatureList(feature=[
        tf.train.Feature(float_list=tf.train.FloatList(
            value=eeg_channel))])
    
    return feature_list

def _int_feature_scalar(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def encode_single_example(eeg_data: np.array, label: int):
    """ TFRecords: process single EEG trial as a SequenceExample"""
    #print(eeg_data.shape)
    eeg_data_flat = eeg_data.copy().reshape((-1, 1))
    
    fl_dict = {'eeg_data':_float_feature_seq(eeg_data_flat)}
    label_dict = {'label':  _int_feature_scalar(label)}
    
    label_features = tf.train.Features(feature=label_dict)

    feature_lists = tf.train.FeatureLists(feature_list=fl_dict)
    protobuff = tf.train.SequenceExample(context=label_features,
                                         feature_lists=feature_lists)

    protobuff_serialised = protobuff.SerializeToString()
    
    return protobuff_serialised

# Mapping from PoS to numerical trigger code in EEG data
pos2code = {"NOUN":500, "VERB":501, "ADJ":502, "DET":503, "PRON":504,
            "ADP":505, "ADV":506, "AUX":507, "CCONJ":508, "INTJ":509,
            "NUM":510, "PART":511, "PROPN":512, "PUNCT":513, "SCONJ":514, "X":516}


def convert(data, labels, n_per_tf, category, n_avg, wr, save_misc) -> None:
    """ Convert NumPY EEG data to TFRecords
    
    Args:
        data:      NumPy array of EEG data (shape = (n_batch, n_chan, n_time))
        labels:    Accompanying labels for each row in `data`
        n_per_tf:  How many EEG samples to include per TFRecord
        category:  The experiment category for correct filename saving
        n_avg:     The number of single-trial averages per pseudotrial
        wr:        Whether data are averaged with/without replacement
        save_misc: Any other adhoc addition to filename while saving
    """
    
    max_range = (len(data) // n_per_tf) + 1
    idx = [n_per_tf * i for i in range(max_range+1)]
    loop_idx = list(zip(idx, idx[1:]))

    for i, (start, stop) in enumerate(loop_idx):
        print(f'Iteration {i}/{len(loop_idx)}, start={start}, stop={stop}')

        X = data[start:stop]
        y = labels[start:stop]

        file_path = f'{category}_{n_avg}_train{wr}{save_misc}_file{i}.tfrecords'

        with tf.io.TFRecordWriter(file_path) as writer:
            for sample, label in zip(data, labels):
                serialised_example = encode_single_example(sample, label)                              
                writer.write(serialised_example)

            writer.close()

# -- Training Data
train_data = np.load(f'{category}_{n_avg}_train{wr}{misc}_data.npy')
train_labels = np.load(f'{category}_{n_avg}_train{wr}{misc}_labels.npy')

convert(data=train_data, labels=train_labels, n_per_tf=4_000,
        category=category, n_avg=n_avg, wr=wr, save_misc=save_misc)

# -- Dev Data
dev_data = np.load(f'{category}_{n_avg}_dev{wr}{misc}_data.npy')
dev_labels = np.load(f'{category}_{n_avg}_dev{wr}{misc}_labels.npy')

convert(data=dev_data, labels=dev_labels, n_per_tf=4_000,
        category=category, n_avg=n_avg, wr=wr, save_misc=save_misc)

# -- Test Data
test_data = np.load(f'{category}_{n_avg}_test{wr}{misc}_data.npy')
test_labels = np.load(f'{category}_{n_avg}_test{wr}{misc}_labels.npy')

convert(data=test_data, labels=test_labels, n_per_tf=4_000,
        category=category, n_avg=n_avg, wr=wr, save_misc=save_misc)