
import os
import numpy as np
import pickle

FRAME_PER_BAR = 16
NUM_BAR = 8
NUM_TIME_STEP = FRAME_PER_BAR * NUM_BAR

ONSET_TOKEN = 2
NB_PITCH_CLASS = 12

# --------------------------------- RHYTHM CONSTRAINT ---------------------------------

def check_alldifferent_notes_bar(path, nb_bars_group, on_test_dataset=False):
    """
    Checks samples have all different number of notes per bar in the first span of nb_bars_group bars

    :path:                  folder path where all the .pkl files are for test or generated samples
    :nb_bars_group:         number of bars that are constrained
    :on_test_dataset:       set to True to check on the test samples

    return                  prints files not respecting the constraint and their total number
    """
    bad_file_count = 0
    for root, _ , files in os.walk(path):
        for file in files:
            if '.pkl' in file and ('groundtruth' if on_test_dataset else 'sample') in file:
                with open(os.path.join(root, file), 'rb') as f:
                    data = pickle.load(f)
                    rhythm_data = data['rhythm']

                    counts = []
                    for i in range(nb_bars_group):
                        curr_bar = rhythm_data[i * FRAME_PER_BAR:(i + 1) * FRAME_PER_BAR]
                        onset_count = np.count_nonzero(curr_bar == ONSET_TOKEN)
                        if (onset_count in counts):
                            print(f'{file}')
                            bad_file_count += 1
                            break
                        counts.append(onset_count)

    print(bad_file_count)

def check_at_least_k_notes(path, nb_bars_group, k, on_test_dataset=False):
    """
    Checks samples have at least k notes in the first span of nb_bars_group bars

    :path:                  folder path where all the .pkl files are for test or generated samples
    :nb_bars_group:         number of bars that are constrained
    :on_test_dataset:       set to True to check on the test samples

    return                  prints files not respecting the constraint and their total number
    """
    bad_file_count = 0
    for root, _ , files in os.walk(path):
        for file in files:
            if '.pkl' in file and ('groundtruth' if on_test_dataset else 'sample') in file:
                with open(os.path.join(root, file), 'rb') as f:
                    data = pickle.load(f)
                    rhythm_data = data['rhythm']
                    rhythm_data = rhythm_data[0:nb_bars_group * FRAME_PER_BAR]
                    onset_count = np.count_nonzero(rhythm_data == ONSET_TOKEN)
                    if (onset_count < k):
                        print(f'{file}')
                        bad_file_count += 1

    print(bad_file_count)

# --------------------------------- PITCH CONSTRAINT ---------------------------------

def check_occurrence_C_major(path, nb_bars_group, on_test_dataset=False):
    """
    Checks samples have at least one occurrence of every note of the C major scale in the first span of nb_bars_group bars

    :path:                  folder path where all the .pkl files are for test or generated samples
    :nb_bars_group:         number of bars that are constrained
    :on_test_dataset:       set to True to check on the test samples

    return                  prints files not respecting the constraint and their total number
    """
    C_major = [0, 2, 4, 5, 7, 9, 11]
    bad_file_count = 0
    for root, _ , files in os.walk(path):
        for file in files:
            if '.pkl' in file and ('groundtruth' if on_test_dataset else 'sample') in file:
                with open(os.path.join(root, file), 'rb') as f:
                    data = pickle.load(f)
                    pitch_data = data['pitch']

                    group_bar = pitch_data[0:nb_bars_group * FRAME_PER_BAR]
                    for note in C_major:
                        pitch_idx = [(i * NB_PITCH_CLASS + note) for i in range(4)]
                        count = 0
                        for idx in pitch_idx:
                            count += np.count_nonzero(group_bar == idx)
                        if count < 1:
                            bad_file_count += 1
                            print(f'{file}')
                            break

    print(bad_file_count)

#---------------------------------------------- CHECK CONSTRAINTS --------------------------------------------------------

sample_path = r'path\path\path\generated_folder_containing_pkl_files'
nb_bars_group = 8

# CHECK ON GENERATED DATASET
check_alldifferent_notes_bar(sample_path, nb_bars_group, False)
check_occurrence_C_major(sample_path, nb_bars_group, False)
check_at_least_k_notes(sample_path, nb_bars_group, 8, False)


# CHECK ON TEST DATASET
check_alldifferent_notes_bar(sample_path, nb_bars_group, True)
check_occurrence_C_major(sample_path, nb_bars_group, True)
check_at_least_k_notes(sample_path, nb_bars_group, 8, True)
