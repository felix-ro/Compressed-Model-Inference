from speech_dataset import SpeechDataset


def getDataset():
    dataset = SpeechDataset(
                    words=['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes'],
                    upper_band_limit=5000.0,  # ~ human voice range
                    lower_band_limit=125.0,
                    feature_bin_count=25,
                    window_size_ms=40.0,
                    window_stride=20.0,
                    silence_percentage=5, unknown_percentage=5
                )
    return dataset
