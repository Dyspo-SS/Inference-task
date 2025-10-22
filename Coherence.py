"""
LFP Coherence Analysis Between Brain Regions
Analyzes theta band (4-12 Hz) coherence between dCA1 and ACC from Plexon SDK exported data
"""

import pandas as pd
import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, coherence
import matplotlib.pyplot as plt
import os
import glob


class LFPCoherenceAnalyzer:
    def __init__(self, sampling_rate=40000, target_fs=1000):
        """
        Initialize the LFP coherence analyzer

        Parameters:
        -----------
        sampling_rate : int
            Original sampling rate (default: 40000 Hz)
        target_fs : int
            Target sampling rate after downsampling (default: 1000 Hz)
        """
        self.sampling_rate = sampling_rate
        self.target_fs = target_fs

    def load_lfp_data(self, filepath):
        """
        Load LFP data from Excel file

        Parameters:
        -----------
        filepath : str
            Path to the Excel file

        Returns:
        --------
        timestamps : numpy array
            Timestamps array
        values : numpy array
            LFP values array
        """
        df = pd.read_excel(filepath, sheet_name='Sheet1')
        timestamps = df.iloc[:, 0].values
        values = df.iloc[:, 1].values
        return timestamps, values

    def downsample_data(self, data, original_fs, target_fs):
        """
        Downsample data to target sampling rate

        Parameters:
        -----------
        data : numpy array
            Original data
        original_fs : int
            Original sampling frequency
        target_fs : int
            Target sampling frequency

        Returns:
        --------
        downsampled_data : numpy array
            Downsampled data
        """
        downsample_factor = int(original_fs / target_fs)
        downsampled_data = signal.decimate(data, downsample_factor, ftype='iir', zero_phase=True)
        return downsampled_data

    def apply_lowpass_filter(self, data, cutoff=250, fs=1000, order=4):
        """
        Apply lowpass filter to remove high frequency noise

        Parameters:
        -----------
        data : numpy array
            Input signal
        cutoff : float
            Cutoff frequency (default: 250 Hz)
        fs : int
            Sampling frequency (default: 1000 Hz)
        order : int
            Filter order (default: 4)

        Returns:
        --------
        filtered_data : numpy array
            Filtered signal
        """
        nyquist = fs / 2
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        filtered_data = filtfilt(b, a, data)
        return filtered_data

    def notch_filter(self, data, freq=50, fs=1000, Q=30):
        """
        Apply notch filter to remove power line interference and harmonics

        Parameters:
        -----------
        data : numpy array
            Input signal
        freq : float
            Frequency to remove (default: 50 Hz)
        fs : int
            Sampling frequency (default: 1000 Hz)
        Q : float
            Quality factor (default: 30)

        Returns:
        --------
        filtered_data : numpy array
            Filtered signal
        """
        filtered_data = data.copy()
        # Remove 50 Hz and its harmonics (50, 100, 150, 200, 250 Hz)
        harmonics = [50, 100, 150, 200, 250]
        for f in harmonics:
            if f < fs / 2:  # Only filter if below Nyquist frequency
                b, a = signal.iirnotch(f, Q, fs)
                filtered_data = filtfilt(b, a, filtered_data)
        return filtered_data

    def preprocess_signal(self, data):
        """
        Complete preprocessing pipeline: downsample -> lowpass -> notch filter

        Parameters:
        -----------
        data : numpy array
            Raw LFP signal

        Returns:
        --------
        processed_data : numpy array
            Preprocessed signal
        """
        # Downsample from 40kHz to 1kHz
        downsampled = self.downsample_data(data, self.sampling_rate, self.target_fs)

        # Apply 250 Hz lowpass filter
        lowpass_filtered = self.apply_lowpass_filter(downsampled, cutoff=250, fs=self.target_fs)

        # Remove 50 Hz power line interference and harmonics
        notch_filtered = self.notch_filter(lowpass_filtered, freq=50, fs=self.target_fs)

        return notch_filtered

    def load_and_preprocess_channels(self, file_pattern, channel_indices):
        """
        Load and preprocess multiple channels from specified files

        Parameters:
        -----------
        file_pattern : str
            File path pattern (e.g., '/path/to/data/channel{}_LFP.xlsx')
        channel_indices : list
            List of channel numbers to load (e.g., [1, 2, 3, 4])

        Returns:
        --------
        processed_signals : list
            List of preprocessed signals for each channel
        """
        processed_signals = []

        for idx in channel_indices:
            filepath = file_pattern.format(idx)
            print(f"Loading and preprocessing: {filepath}")

            # Load data
            timestamps, values = self.load_lfp_data(filepath)

            # Preprocess
            processed = self.preprocess_signal(values)
            processed_signals.append(processed)

        return processed_signals

    def average_signals(self, signals):
        """
        Average multiple signals to get regional activity

        Parameters:
        -----------
        signals : list
            List of numpy arrays (signals from different channels)

        Returns:
        --------
        averaged_signal : numpy array
            Averaged signal across channels
        """
        # Ensure all signals have the same length
        min_length = min([len(s) for s in signals])
        truncated_signals = [s[:min_length] for s in signals]

        # Average across channels
        averaged_signal = np.mean(truncated_signals, axis=0)
        return averaged_signal

    def calculate_theta_coherence(self, signal1, signal2, fs=1000, theta_band=(4, 12)):
        """
        Calculate coherence between two signals in theta frequency band

        Parameters:
        -----------
        signal1 : numpy array
            First signal (e.g., brain region 1 average)
        signal2 : numpy array
            Second signal (e.g., brain region 2 average)
        fs : int
            Sampling frequency (default: 1000 Hz)
        theta_band : tuple
            Theta frequency band range (default: (4, 12) Hz)

        Returns:
        --------
        frequencies : numpy array
            Frequency values in theta band
        coherence_values : numpy array
            Coherence values corresponding to frequencies
        """
        # Calculate coherence using Welch's method
        nperseg = min(2 * fs, len(signal1), len(signal2))  # 2-second windows
        f, Cxy = coherence(signal1, signal2, fs=fs, nperseg=nperseg)

        # Extract theta band
        theta_mask = (f >= theta_band[0]) & (f <= theta_band[1])
        theta_frequencies = f[theta_mask]
        theta_coherence = Cxy[theta_mask]

        return theta_frequencies, theta_coherence

    def plot_coherence(self, frequencies, coherence_values, save_path=None):
        """
        Plot coherence as a line plot

        Parameters:
        -----------
        frequencies : numpy array
            Frequency values
        coherence_values : numpy array
            Coherence values
        save_path : str, optional
            Path to save the figure
        """
        plt.figure(figsize=(10, 6))
        plt.plot(frequencies, coherence_values, 'b-', linewidth=2)
        plt.xlabel('Frequency (Hz)', fontsize=12)
        plt.ylabel('Coherence', fontsize=12)
        plt.title('Theta Band Coherence Between Brain Regions', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xlim([frequencies[0], frequencies[-1]])
        plt.ylim([0, 1])

        # Add mean coherence as text
        mean_coherence = np.mean(coherence_values)
        plt.text(0.7, 0.95, f'Mean Coherence: {mean_coherence:.3f}',
                 transform=plt.gca().transAxes,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                 fontsize=11)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")

        plt.show()

    def analyze_coherence(self, data_folder, output_folder=None):
        """
        Complete analysis pipeline

        Parameters:
        -----------
        data_folder : str
            Folder containing the Excel files
        output_folder : str, optional
            Folder to save results (default: same as data_folder)
        """
        if output_folder is None:
            output_folder = data_folder

        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Define file pattern
        file_pattern = os.path.join(data_folder, 'channel{}_LFP.xlsx')

        # Define brain regions
        region1_channels = [1, 2, 3, 4]  # Channels 01-04
        region2_channels = [5, 6, 7, 8]  # Channels 05-08

        print("=" * 60)
        print("Loading and preprocessing Brain Region 1 (Channels 01-04)")
        print("=" * 60)
        region1_signals = self.load_and_preprocess_channels(file_pattern, region1_channels)

        print("\n" + "=" * 60)
        print("Loading and preprocessing Brain Region 2 (Channels 05-08)")
        print("=" * 60)
        region2_signals = self.load_and_preprocess_channels(file_pattern, region2_channels)

        print("\n" + "=" * 60)
        print("Averaging signals within each brain region")
        print("=" * 60)
        region1_avg = self.average_signals(region1_signals)
        region2_avg = self.average_signals(region2_signals)
        print(f"Region 1 averaged signal length: {len(region1_avg)} samples")
        print(f"Region 2 averaged signal length: {len(region2_avg)} samples")

        print("\n" + "=" * 60)
        print("Calculating theta band (4-12 Hz) coherence")
        print("=" * 60)
        frequencies, coherence_values = self.calculate_theta_coherence(
            region1_avg, region2_avg, fs=self.target_fs, theta_band=(4, 12)
        )

        # Calculate statistics
        mean_coh = np.mean(coherence_values)
        max_coh = np.max(coherence_values)
        peak_freq = frequencies[np.argmax(coherence_values)]

        print(f"Mean theta coherence: {mean_coh:.4f}")
        print(f"Maximum coherence: {max_coh:.4f} at {peak_freq:.2f} Hz")

        # Save results to CSV
        results_df = pd.DataFrame({
            'Frequency_Hz': frequencies,
            'Coherence': coherence_values
        })
        csv_path = os.path.join(output_folder, 'theta_coherence_results.csv')
        results_df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")

        # Plot and save figure
        print("\n" + "=" * 60)
        print("Generating coherence plot")
        print("=" * 60)
        fig_path = os.path.join(output_folder, 'theta_coherence_plot.png')
        self.plot_coherence(frequencies, coherence_values, save_path=fig_path)

        return frequencies, coherence_values


def main():
    """
    Main function to run the analysis
    """
    # Example usage
    # Modify these paths according to your data location
    data_folder = '/path/to/your/data/folder'  # Change this to your data folder
    output_folder = '/path/to/output/folder'  # Change this to your output folder

    # Initialize analyzer
    analyzer = LFPCoherenceAnalyzer(sampling_rate=40000, target_fs=1000)

    # Run analysis
    frequencies, coherence_values = analyzer.analyze_coherence(data_folder, output_folder)

    print("\n" + "=" * 60)
    print("Analysis completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    # For testing, you can modify the paths below
    print("LFP Coherence Analysis Script")
    print("Please modify the data_folder and output_folder paths in the main() function")
    print("\nOr use the analyzer programmatically:")
    print("  analyzer = LFPCoherenceAnalyzer()")
    print("  analyzer.analyze_coherence('/path/to/data', '/path/to/output')")