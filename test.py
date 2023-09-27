import numpy as np



def normalize_features(features):
    if not isinstance(features, (list, np.ndarray)):
        raise ValueError("Input 'features' must be a list or numpy array.")
    features = np.array(features)
    mean = np.mean(features)
    std_dev = np.std(features)
    if std_dev == 0:
        return features
    normalized_features = (features - mean) / std_dev
    return normalized_features

# Example of usage
input_features = [2, 4, 6, 8, 10]
normalized = normalize_features(input_features)
print("Normalized Features:", normalized)
