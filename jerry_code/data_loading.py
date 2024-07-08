"""
Functions for loading data
I put all data files in batches (of currently 500 events) and load each batch
All data batches specified by prefix and batch number
"""

import numpy as np
import csv

import uproot


def load_csv(csv_path, has_headers=True, has_y=True):
    with open(csv_path, 'r') as file:
        # Create a CSV reader object
        csv_reader = csv.reader(file)

        if has_headers:
            headers = next(csv_reader)

        x_data = []
        y_data = []
        # Iterate over each row in the CSV file
        for row in csv_reader:
            # Each row is a list where each element represents a column value
            row = [float(value) for value in row]
            if has_y:
                x_data.append(row[:-1])
                y_data.append(row[-1])
            else:
                x_data.append(row)

    x_data, y_data = np.array(x_data), np.array(y_data)

    return x_data, y_data


def load_data(file_path, batch_range=(0, 8), flatten_singular_dimension=True):
    # Loads all events
    x_data_lst, y_data_lst = [], []
    for i in range(batch_range[0], batch_range[1]):
        cur_x, cur_y = load_csv(file_path + str(i) + '.csv')
        print(f"Done batch {i} from '{file_path}' files")
        x_data_lst.append(cur_x)
        y_data_lst.append(cur_y)
    x_data = np.vstack(x_data_lst)
    y_data = np.concatenate(y_data_lst)
    if flatten_singular_dimension and (x_data.shape[0] == 1 or x_data.shape[1] == 1):
        x_data = x_data.reshape(-1,)
    return x_data, y_data


def load_train_test(file_path, train_range, test_range):
    # Return training data, x_test, y_test
    x_train, y_train = load_data(file_path, train_range)
    training_data = x_train[y_train == 0]
    x_test, y_test = load_data(file_path, test_range)
    return training_data, x_test, y_test


def load_truth_hs_z(root_file_path, start, end):
    """
    Temporary solution for extracting HS vertex z-coordinates.
    May instead store these coords in csv in future
    :param root_file_path:
    :param start: start index of event
    :param end: index after final event to load
    :return:
    """
    with uproot.open(root_file_path) as file:
        tree_name = "EventTree;1"
        tree = file[tree_name]

    truth_z_array = tree['truthvertex_z'].array()
    hs_z_coords = []
    for i in range(start, end):
        hs_z_coords.append(truth_z_array[i][0])

    return np.array(hs_z_coords)
