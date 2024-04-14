import numpy as np
import mne


def extract_emg_data(raw_dataset: mne.io.Raw, inverse: bool = False) -> np.ndarray:
    """ Extract the raw EMG dataset

    Args:
        raw_dataset (mne.io.Raw): The raw dataset object of dimension: [Channel*Time]
        inverse (bool): Whether the channels have been inversed during data collection. Default is False.

    Returns:
        np.ndarray: The EMG data
    """
    emg_data = np.zeros(shape=(2, raw_dataset.get_data().shape[-1]))
    # process emg_data only in motor tasks

    if not inverse:
        right_ch1 = 66; right_ch2 = 67
        left_ch1 = 68; left_ch2 = 69
    else:
        right_ch1 = 68; right_ch2 = 69
        left_ch1 = 66; left_ch2 = 67
        
    # Compute the difference
    emg_left = raw_dataset.get_data()[left_ch2, :] - raw_dataset.get_data()[left_ch1, :]
    emg_right = raw_dataset.get_data()[right_ch2, :] - raw_dataset.get_data()[right_ch1, :]

    # compute rectification and absolute values. 
    centred_left = (emg_left) - np.mean(emg_left)
    centred_right = (emg_right) - np.mean(emg_right)

    # Concatenate the data (Left and Right )
    emg_data = np.concatenate((centred_left[None, :], centred_right[None, :]))

    return emg_data


def extract_ecg_data(raw_dataset: mne.io.Raw, inverse : bool = False) -> np.ndarray:
    """ Extract the raw ECG dataset

    Args:
        raw_dataset (mne.io.Raw): The raw dataset object of dimension: [Channel*Time]
        inverse (bool): . Whether channels have been inverted. Default is False.

    Returns:
        np.ndarray: The array of dimension [Channel*Time] containing the difference 
                    between the bipolar ECG channels
    """
    # Extract ECG data, compute the difference and store in new array
    ch_idx = 70
    ecg_data = raw_dataset.get_data()[ch_idx:, :]  # EXG7-EXG8

    #  compute the difference to obtain monopolar recording
    if not inverse:
        ecg_data = ecg_data[1, :] - ecg_data[0, :]
    else:
        ecg_data = ecg_data[0, :] - ecg_data[1, :]



    return ecg_data[None, :]


def smooth_emg_signal(emg_data: np.ndarray, window_size: int):
    """
    Smooth the EMG signal using the Root Mean Square (RMS) formula.

    Args:
        emg_data (np.ndarray): One-dimensional EMG dataset with shape (2, n_time),
                               where the first row represents the left EMG and
                               the second row represents the right EMG.
        window_size (int): The size of the sliding window used for smoothing.
        
    Note:
        This function reduces the size of the data by `window_size`. Zero-padding is added at the end of recordings.
    """

    n_time = emg_data.shape[-1]
    left_emg = emg_data[0, :]
    right_emg = emg_data[1, :]

    # Initialize arrays to store RMS values
    left_rms = np.zeros(shape=(left_emg.shape[-1]-window_size))
    right_rms = np.zeros(shape=(right_emg.shape[-1]-window_size))

    # Compute this only if Motoreal and MotorImagery
    for index in range(0, n_time):
        try:
            # Get sliding window data
            left_windowed_data = get_sliding_window(data=left_emg, time_index=index, window_size=window_size)
            right_windowed_data = get_sliding_window(data=right_emg,time_index=index, window_size=window_size)

            # Compute RMS for the sliding window
            left_rms_windowed_data = np.sqrt(np.mean(left_windowed_data ** 2))
            right_rms_windowed_data = np.sqrt(np.mean(right_windowed_data ** 2))
            
            # Store RMS values
            left_rms[index] = left_rms_windowed_data
            right_rms[index] = right_rms_windowed_data
        except Exception as e:
            pass
    
    # ZERO-PADDING at the end of recordings
    z = np.zeros(shape=(window_size))
    left_rms = np.concatenate((z, left_rms))
    right_rms = np.concatenate((z, right_rms))
    emg_data = np.concatenate((left_rms[None, :], right_rms[None, :]))

    return emg_data
    

def create_new_raw_dataset(eeg_data: mne.io.Raw, ecg_data: np.ndarray, emg_data: np.ndarray) -> mne.io.RawArray:
    """ Create a new raw EEG dataset by combining EEG and ECG dataset (can be modified if EMG data is also needed)

    Args:
        raw_dataset (mne.io.Raw): The raw dataset object of dimension: [Channel*Time]
        ecg_data (np.ndarray): The array of dimension [Channel*Time] containing the difference 
                                  between the bipolar ECG channels

    Returns:
        mne.io.RawArray: The new raw EEG dataset containing the processed EEG and ECG data
    """
    eeg_array = eeg_data.get_data() # Extract the EEG data into numpy array

    # Create new info
    ch_names = eeg_data.info['ch_names'] 

    ch_names.append('EXG7d') # 'ECG channel

    ch_names.append('EMGl') # EMG channels 
    ch_names.append('EMGr')

    sfreq = eeg_data.info['sfreq']
    new_info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')

    # Concatenate the EEG and ECG array
    new_data = np.concatenate((eeg_array, ecg_data, emg_data)) if emg_data is not None else np.concatenate((eeg_array, ecg_data))

    # Create a new RawArray object with both EEG and ECG data using new info
    new_raw_dataset = mne.io.RawArray(data=new_data, info=new_info)

    # set external channels
    mapping = {'EXG7d': 'ecg', 'EMGl': 'emg', 'EMGr': 'emg'}
    new_raw_dataset.set_channel_types(mapping=mapping)

    return new_raw_dataset


def get_sliding_window(data: np.ndarray, time_index: int, window_size: int) -> np.ndarray:
    """ Obtain a sliding window that moves of one time_index at the time. This method is used to smooth 
    the EMG signal using the RMS formula.
        
    Args:
        data (np.ndarray): The one-dimensional array of len nTimes
        time_index (int): The first index of the sliding-window. It increases iteratively in the loop.
        window_size (int): The size of the sliding window.

    Returns:
        np.ndarray: The one-dimensional windowed data of len window_size
    """
    data = data.T
    window = np.arange(time_index, time_index + window_size)
    return data[window].T