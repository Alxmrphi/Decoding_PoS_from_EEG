import glob
import numpy as np
import os
import pathlib
import pickle
import tensorflow as tf
from functools import partial


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
    
    #print(f'label_features : {label_features}')
    
    feature_lists = tf.train.FeatureLists(feature_list=fl_dict)
    protobuff = tf.train.SequenceExample(context=label_features,
                                         feature_lists=feature_lists)

    protobuff_serialised = protobuff.SerializeToString()
    
    return protobuff_serialised

def decode_single_example(serialised_example, start_window, end_window,
                           total_timepoints, n_electrodes):

    data_dim = total_timepoints * n_electrodes

    context_desc = {'label': tf.io.FixedLenFeature([], dtype=tf.int64)}
    feature_desc = {
        'eeg_data': tf.io.FixedLenSequenceFeature([data_dim], dtype=tf.float32)
    }

    context, data = tf.io.parse_single_sequence_example(
        serialized=serialised_example,
        context_features=context_desc,
        sequence_features=feature_desc,
        name='parsing_single_seq_example')

    data = data['eeg_data']
    data = tf.reshape(data, (n_electrodes, total_timepoints))
    #data = tf.transpose(data, (1,0))
    
    print(f'data shape: {data.shape}')
    data = data[:, start_window:end_window] # Extract (potential sub-window)
    WINDOW_LENGTH = end_window - start_window
    data = tf.reshape(data, (1, WINDOW_LENGTH * n_electrodes))

    label = context['label']
    label = tf.cast(label, tf.int32)

    return data, label

def get_dataset(file_regex, batch_size, repeat,
               start_window, end_window, total_timepoints,
               n_electrodes):
    
    decode_single_example_fn = partial(decode_single_example,
                                       start_window=start_window,
                                       end_window=end_window,
                                       total_timepoints=total_timepoints,
                                       n_electrodes=n_electrodes)
    
    files = list(pathlib.Path(".").glob(file_regex))
    assert len(files) > 0, f"No files found for: {file_regex}"
    files = [str(x) for x in files]
    print(f'Found files: {files}')
    
    dataset = tf.data.TFRecordDataset(files, num_parallel_reads=1)
    dataset = dataset.map(decode_single_example_fn, num_parallel_calls=1)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.cache()
    dataset = dataset.shuffle(BATCH_SIZE)
    dataset = dataset.repeat(repeat)

    return dataset


def convert(data, labels, n_per_tfr, fname=None):
    """ Convert NumPY EEG data to TFRecords
    
    Args:
        data:      NumPy array of EEG data (shape = (n_batch, n_chan, n_time))
        labels:    Accompanying labels for each row in `data`
        n_per_tf:  How many EEG samples to include per TFRecord
        fname:     Filename for saved TFRecords
        
    Returns:
        None
    """
    
    max_range = (len(data) // n_per_tfr) + 1
    idx = [n_per_tfr * i for i in range(max_range+1)]
    loop_idx = list(zip(idx, idx[1:]))

    for i, (start, stop) in enumerate(loop_idx, start=1):
        print(f'Iteration {i}/{len(loop_idx)}, start={start}, stop={stop}')

        X = data[start:stop]
        y = labels[start:stop]

        file_path = f'{fname}_file{i}.tfrecords'

        with tf.io.TFRecordWriter(file_path) as writer:
            for sample, label in zip(X, y):
                serialised_example = encode_single_example(sample, label)                              
                writer.write(serialised_example)

            writer.close()