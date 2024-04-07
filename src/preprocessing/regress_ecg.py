from os.path import join
from os import listdir
import mne
import numpy as np
import matplotlib.pyplot as plt
from mne.baseline import rescale
import neurokit2 as nk



def regress_ecg(ecg_data_raw: np.ndarray, epochs: mne.Epochs) -> np.ndarray:
    """ Use a regression model to remove the heartbeat artefact from the EEG data. 

    Args:
        ecg_data_raw (np.ndarray): The 1D ECG raw signal obtained from the whole recording 
        epochs (mne.Epochs): The mne Epochs object of dimension nEpochs*nChannels*nTime

    Returns:
        np.ndarray: a Numpy array of dimension nEpochs*nChannels*nTime containing the epochs without the heartbeat artefact
    """

    # extract the R-peaks events from the whole recording (make sure the data is preprocessed)
    r_peaks = nk.ecg.ecg_findpeaks(ecg_data_raw, sampling_rate=epochs.info['sfreq'], show=False)['ECG_R_Peaks']

    # Get ECG and EEG epochs data
    ecg_data = epochs.get_data('ecg') # the ECG epochs
    eeg_data = epochs.get_data('eeg') # the EEG epochs

    # Get dimensions of the data
    n_epochs = epochs.get_data().shape[0] # number of epochs
    n_channels = epochs.get_data('eeg').shape[1] #  number of channels
    n_time = epochs.get_data().shape[-1] # number of time points

    # Create an empty numpy array for regressed data
    regressed_eeg_epochs = np.zeros(shape=(n_epochs, n_channels, n_time))

    # Initialize arrays for error calculation
    mean_errors = np.zeros(shape=(n_epochs, n_channels))
    errors = np.zeros(shape=(n_epochs, n_channels, n_time))

    # Get time vector and events array
    time = epochs.times # the 1D time vector (given by mne.Epochs object)
    events = epochs.events # the event array contaning the timing (col1) and event id (col3)

    rpeaks_events = []
    for n_epoch in range(0, n_epochs):

        # Get the relevant R-peaks timing for the current epoch
        r_peaks_temp = r_peaks[np.where(r_peaks >= events[n_epoch, 0])[0][0]] 
        rpeaks_events.append(r_peaks_temp) # Store the R-peaks events

        tzero = np.where(time == 0.0)[0][0] # get index of time zero (cue presentation)
        # Calculate distance from cue presentation to R-peak
        distance_from_cue = r_peaks_temp - events[n_epoch, 0] 

        # Calculate indices before and after the R-peak
        idx1 = (tzero + distance_from_cue) - 29  # 29 indices before R-peak (~30 ms)
        idx2 = (tzero + distance_from_cue) + 29  # 29 indices after R-peak (~30 ms)

        # Extract ECG and EEG data around the R-peak
        eegc_ecg_ch = ecg_data[:, :, idx1:idx2]
        eegc_all_ch = eeg_data[:, :, idx1:idx2]

        # Build the least-square model as intercept and ECG for each trial
        intercept_ecg = np.ones(shape=(ecg_data.shape[-1])) 
        ecg = ecg_data[n_epoch, 0, :] 

        # Create an array of ones for the intercept term in the regression model
        intercept_eegc = np.ones(shape=(eegc_ecg_ch.shape[-1])) 
        eegc_ecg_ch = eegc_ecg_ch[n_epoch, 0, :] # Select the ECG data for the current epoch

        Xeeg = np.concatenate( (intercept_ecg[None, :], ecg[None, :]) ).T # Construct the design matrix Xeeg for the regression model
        X = np.concatenate( (intercept_eegc[None, :], eegc_ecg_ch[None, :]) ).T # Construct the design matrix X for the regression model using ECG data

        b_1 = np.dot(X.T, X) # Compute the matrix product of X transpose and X (X'X)
        b_2 = np.dot(X.T, eegc_all_ch[n_epoch, :, :].T) # Compute the matrix product of X transpose and the EEG data (X'Y) in the current epoch

        # Perform least squares regression to estimate the coefficients (b) of the model
        b, resid, rank, s = np.linalg.lstsq(b_1, b_2)
        # Predict the EEG data without the ECG artifact using the regression coefficients (b)
        y_hat = np.dot(Xeeg, b)

        # Remove ECG artifact from EEG data
        temp_eeg_data = eeg_data[n_epoch, :, :] - y_hat.T
        regressed_eeg_epochs[n_epoch, :, :] = temp_eeg_data

        # Calculate the error as the squared distance between Y and Yhat
        error = (eeg_data[n_epoch, :, :] - temp_eeg_data)

        # calculate where, for each channel, is the max error
        mean_errors[n_epoch, :] =  np.mean(error, axis=1)
        errors[n_epoch, :, :] = error

    return regressed_eeg_epochs