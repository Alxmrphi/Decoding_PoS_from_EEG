from collections import defaultdict
from psychopy import visual, core, event, gui
import numpy as np
from numpy import random
import os
import re
import io, codecs
import pickle
import datetime
import time
import sys

print("Using Python {}".format(sys.version))

# Experiment-specific params
useEEG = True
labjack_address = 6751
wpm = 350.
labjack_address = 6751
global_trigger_counter = 1
full_screen = True
redicle_map = defaultdict(lambda: 4) # Anything else defaults to position 4
redicle_map.update({1:1, 2:2, 3:2, 4:2, 5:2, 6:3, 7:3, 8:3, 9:3})
pixel_width = 0.05
post_sentence_wait_time = 0.5 # 500 ms gap between sentences

# Session-specific params
# NB: start/end determines how many files are presented and thus
# determine how long the recording session will run for. This
# is determined prior to the session when a desired running time
# is available by consulting the random presentation order of files.
session = 0
subj_name = 'name'
start_file = 0 
end_file = 15 

if useEEG:
    from psychopy.hardware.labjacks import U3
    p_port = U3()

texts_path = 'C:\\Users\\locadmin\\Documents\\MATLAB\\Alex_M\\SingleFiles2019'
file_perms = 'C:\\Users\\locadmin\\Documents\\MATLAB\\Alex_M\\file_permutations20.npy'

# These trigger codes have special status in the EEG to mark major events in the experiment
# such as end of file, when recall episodes occur and when the experiment begins / ends
trigger_map = {'session_start': 200,
               'session_end':   202,
               'sentence_start':202,
               'sentence_end':  203,
               'recall_start':  204,
               'recall_end':    205,
               'start_new_file':206,
               'end_new_file':  207}

# -- Helper Functions -- 

def get_file_order(session):
    """ Read the permutation file and return relevant file ordering """
    all_file_permutations = np.load(file_perms)
    current_files = all_file_permutations[session]
    return current_files

def get_redicle_pos(word):
    """ Return, for a given word, the character that sits at central position """
    return redicle_map[len(word)]

def get_keypress():
    keys = event.getKeys()
    return keys[0] if keys else None

def get_recall_indices(text):
    """ This function takes in a text and provides indices for sentences when
    the word recall occurs """
    num_sents = len(text)
    num_recall_tests = int(np.floor((num_sents / 100.) * 20.)) # 20% of sentences recalled
    #print("num recall tests: {}".format(num_recall_tests))
    sent_idx = [i for (i, sent) in enumerate(text) if len(sent.split()) > 15]
    recall_idx = np.random.choice(sent_idx, size=num_recall_tests, replace=False)
    return np.sort(recall_idx)

def shutdown(name=None, useEEG=True):
    """ Close down labjack and PsychoPy correctly """
    if useEEG:
        send_eeg_trigger('session_end', useEEG)
        p_port.close()
    win.close()
    core.quit()

def plotline():
    """ Plot a small line under redicle position for presented word """
    line = visual.Line(win=win, lineColor=[1, 0, 0], start=[0.00,-0.15],
                       end=[0.00, 0.15], opacity=0.25).draw()

def display_word(word, plot_line=False, cwr=0.2, useEEG=True, sendTrigger=True):
    """ Function to present a word on the screen and send corresponding trigger to EEG system """

    redicle_pos = get_redicle_pos(word)
    pos = (-(redicle_pos * pixel_width - 0.03),0) # Reposition according to OVP
    visual.TextStim(win, alignHoriz='left', text=word, font='Courier New', pos=pos).draw()

    if plot_line: plotline()
    win.flip()

    if useEEG and sendTrigger:
        send_eeg_trigger(global_trigger_counter, useEEG)

    core.wait(cwr)

def load_texts(file_order):
    """ Load texts for presentation into memory, return dict object """
    texts = {}
    for file in file_order:
        path = os.path.join(texts_path, file)
        # with open(path, 'r', encoding='utf-8', errors='ignore') as tmp:
        #with codecs.open(path, 'rb', encoding="utf-8") as tmp:
        with io.open(path,'r') as tmp:
            tmp_text = tmp.readlines()
        texts[file] = tmp_text
    return texts

def send_eeg_trigger(code, useEEG):
    """ Send a trigger to the EEG system """
    if useEEG:
        if type(code) == str:
            code = trigger_map[code]
        print("Sending trigger: {}".format(code))
        p_port.setData(int(code), address=labjack_address)
        time.sleep(0.001)
        p_port.setData(0, address=labjack_address) # Helps minimise lost triggers

def ask_for_words():
    """ Pause experiment while subject reads back previously read sentence.
        Wait for space bar to indicate completion """
    paused = True
    while paused:
        msg = visual.TextStim(win, text="Repeat back previous sentence to experimenter. \
            Press space bar to continue").draw()
        core.wait(0.01)
        if get_keypress() == 'space':
            paused = False
    return
# -- End helper functions -- 

if full_screen:
    win = visual.Window(size=(1920,1080), fullscr=True)
else:  
    win = visual.Window(size=(1000,700))



msg = visual.TextStim(win, text="Press spacebar to begin...").draw()
win.flip()
event.waitKeys(keyList=["space"])  # wait for a spacebar press before continuing
event.clearEvents()


# -- File Permutations --
# The corpus of text corresponding to our data subset consists of 151 different text files.
# These were presented in random order for each recording session
# The entire list of files was randomly shuffled 20 times (way more than we would need)
# Using `session`, `start_file` and `end_file` variables defined at the start, this indexes
# the file permutations list and selects the text files for presentation per recording session.

file_order = get_file_order(session) # Pick out random ordering of text file presentations
files_this_session = file_order[start_file:end_file] # Take subset for this EEG session
texts = load_texts(files_this_session) # Load into memory for later iteration (nb: type(texts) == dict)

# -- Main loop --
send_eeg_trigger('session_start', useEEG)

for text_file in texts:

    text = texts[text_file] # Extract full text, each element in list a sentence string
    recall_idx = get_recall_indices(text)
    print("Current file : {}".format(text_file))
    print("Recall indices for {} : {}".format(text_file, recall_idx))
    txt = "Next text file : {}. \n Press Space to begin".format(text_file)
    visual.TextStim(win, alignHoriz='left', text=txt, font='Courier New', pos=(-0.7,0)).draw()
    win.flip()
    event.waitKeys(keyList=["space"]) # Wait for subject to start experiment

    send_eeg_trigger('start_new_file', useEEG) 

    for sent_idx, sentence in enumerate(text):
        send_eeg_trigger('sentence_start', useEEG)
        global_trigger_counter = 1

        for word in sentence.split():

            button_pressed = get_keypress()

            if button_pressed == 'escape':
                shutdown(name=subj_name, useEEG=useEEG)

            elif button_pressed == 'p': # detected 'p' key press to pause
                paused = True
                while paused:
                    core.wait(0.01)
                    if get_keypress() == 'p': # 'p' to resume
                        paused = False

            display_word(word, useEEG=useEEG)
            global_trigger_counter += 1
            
        # End sentence
        send_eeg_trigger('sentence_end', useEEG)

        # Perform recall check
        if sent_idx in recall_idx:
            send_eeg_trigger('recall_start', useEEG)
            ask_for_words()
            send_eeg_trigger('recall_end', useEEG)

        # Sentences are split by displaying a fixation cross, don't include
        # as a trigger as it doesn't exist in the corpus data
        display_word('+', useEEG=useEEG, sendTrigger=False)

# Shutdown function takes care of closing windows and writing to files etc.
shutdown(name=subj_name, useEEG=useEEG)