import numpy as np
import pandas as pd
import scipy.io


if __name__ == '__main__':
    full_dataset = np.empty((0, 1601))
    for i in range(1, 33):
        print(f'Loading {i}.mat')
        matfile = scipy.io.loadmat(f'./data/affnist_data/transformed/training_and_validation_batches/{i}.mat')
        labels = matfile['affNISTdata']['label_int'][0][0][0]
        images = matfile['affNISTdata']['image'][0][0]
        batch = np.vstack([images, labels]).T # each row is a case
        full_dataset = np.vstack([full_dataset, batch])
    n_cols = full_dataset.shape[1]
    col_names = [str(i) for i in range(1, n_cols)] + ['label']
    frame = pd.DataFrame(full_dataset, columns=col_names)
    frame.to_csv('./data/affnist.csv', index=False)
