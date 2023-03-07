import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from scipy import signal

window_size = 1000
input_dim = 270

with open("/home/joanna/collected_data_preprocessing/noisy_data_1.pk1", "rb") as file:
    data = pickle.load(file)
x_full = data[0]
y_full = data[1]

x_full_resampled = []
for i in range(len(x_full)):
    x_original = x_full[i]
    x_original = np.reshape(x_original, (-1, input_dim))
    x_resampled = signal.resample(x_original, window_size)
    x_resampled = np.reshape(x_resampled, -1)
    x_full_resampled.append(x_resampled)

x_full_resampled = np.vstack(x_full_resampled)
y_full = np.vstack(y_full)
# print(x_full_resampled.shape)
# print(y_full.shape)

# Split into training and testing data (80/20)
x_train, x_test, y_train, y_test = train_test_split(x_full_resampled, y_full, test_size=0.20, random_state=1000)

data = [x_train, x_test, y_train, y_test]
with open("noisy_data_1_fixed_length.pk1", "wb") as file:
    pickle.dump(data, file)