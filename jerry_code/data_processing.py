"""
Reads data from root file and stores it in csv
This is where I do all my data processing. I included a fraction of the files I processed already
so you don't have to process them.

Notes
noticed artifact in data: last vertex has zero pt, same coordinates as HS vertex
removing last vertex in data from data processing
"""

import numpy as np
import uproot
import csv


# Insert root file path
ROOT_PATH = ""


def compute_track_pt(tree_file, out_path, event_range):
    with uproot.open(tree_file) as file:
        tree = file['EventTree;1']
        qOverP_array = tree['track_qOverP'].array()
        theta_array = tree['track_theta'].array()

    track_pt_data = []
    if event_range[0] < 0 or event_range[0] >= event_range[1] or event_range[1] > len(theta_array):
        raise ValueError("Event range is invalid!")
    print(f"Starting track pt calculations for events {event_range[0]} to {event_range[1]-1}")
    for i in range(event_range[0], event_range[1]):
        thetas = np.array(theta_array[i])
        qOverP_vals = np.array(qOverP_array[i])
        pts = np.sin(thetas) / np.abs(qOverP_vals) / 1000
        track_pt_data.append(pts)

    print(f"Completed pt calculations for {event_range[1] - event_range[0]} events.")

    with open(out_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(track_pt_data)



def load_pt(pt_path):
    track_pts = []
    with open(pt_path, mode='r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            track_pts.append([float(entry) for entry in row])

    # track_deltaRs = []
    # with open(r_path, mode='r') as file:
    #     csv_reader = csv.reader(file)
    #     for row in csv_reader:
    #         track_deltaRs.append([float(entry) for entry in row])

    print("Done loading csv files")
    return track_pts


def read_pt_to_csv(tree, track_pts, outfile_path, N_TRACKS=50, batch_range=None):
    """
    Reads the pts to a csv-file. Top N_TRACKS pt
    :param track_pts: list of computed track pts
    :param outfile_path: Name of output file
    :param N_TRACKS: number of pt tracks to store
    :param batch_range: two-tuple specifying event indices of start and end of range to process
    :return:
    """

    idx_array = tree['recovertex_tracks_idx'].array()
    weight_array = tree['recovertex_tracks_weight'].array()
    isHS_array = tree['recovertex_isHS'].array()
    vertex_z_array = tree['recovertex_z'].array()

    pt_data = []
    z_coords = []
    labels = []
    N_events = len(idx_array)
    if batch_range is None:
        batch_range = (0, N_events)
    if batch_range[1] > N_events or batch_range[0] < 0:
        raise ValueError("Invalid range request for processing batch!")
    print(f"Processing Batch from event {batch_range[0]} to event {batch_range[1]}")
    for i in range(batch_range[0], batch_range[1]):
        # Ignoring final vertex for labels and z-coordinate
        labels.extend(isHS_array[i][:-1])
        # Collect the data on the current event
        N_vertices = len(idx_array[i])
        event_pts = np.array(track_pts[i])
        event_weight_arrs = weight_array[i]
        event_idx_arrs = idx_array[i]
        # Note: we know the last vertex is an artifact so we ignore it
        for j in range(N_vertices-1):
            # Get proper tracks (i.e. weight > 0.75)
            vertex_weights = event_weight_arrs[j]
            vertex_idxs = event_idx_arrs[j]
            weight_mask = vertex_weights >= 0.75
            vertex_idxs = vertex_idxs[weight_mask]
            vertex_pts = event_pts[vertex_idxs]
            pt_mask = vertex_pts <= 50
            # Sort tracks by pt, in descending order
            vertex_pts = vertex_pts[pt_mask]
            vertex_pts = np.sort(vertex_pts)[::-1]
            # Pad with zeros
            if len(vertex_pts) >= N_TRACKS:
                vertex_pts = vertex_pts[:N_TRACKS]
            else:
                vertex_pts = np.pad(vertex_pts, (0, N_TRACKS - len(vertex_pts)), mode='constant')
            pt_data.append(vertex_pts)
        if i % 100 == 99:
            print(f"Done event {i}.")

    # Create headers
    headers = ["pt_" + str(i) for i in range(N_TRACKS)]
    headers.append("y")
    feat_data = np.column_stack((np.array(pt_data), labels))
    feat_data = np.vstack((headers, feat_data))

    np.savetxt(outfile_path, feat_data, delimiter=',', fmt='%s')
    print(f"Successfully saved data batch to '{outfile_path}'")


def read_z_coord_to_csv(tree, outfile_path, batch_range=None):
    isHS_array = tree['recovertex_isHS'].array()
    vertex_z_array = tree['recovertex_z'].array()

    z_coords = []
    labels = []
    N_events = len(vertex_z_array)
    if batch_range is None:
        batch_range = (0, N_events)
    if batch_range[1] > N_events or batch_range[0] < 0:
        raise ValueError("Invalid range request for processing batch!")
    print(f"Processing Batch from event {batch_range[0]} to event {batch_range[1]}")
    for i in range(batch_range[0], batch_range[1]):
        # Ignoring final vertex for labels and z-coordinate
        labels.extend(isHS_array[i][:-1])
        z_coords.extend(vertex_z_array[i][:-1])

    z_coord_data = np.column_stack((z_coords, labels))
    z_coord_data = np.vstack((["z_coord", "y"], z_coord_data))

    np.savetxt(outfile_path, z_coord_data, delimiter=',', fmt='%s')
    print(f"Successfully saved data batch to '{outfile_path}'")


def read_sum_pt2_to_csv(tree, track_pts, outfile_path, batch_range=None):
    """
    Reads sum of pt2 for each vertex to a csv file, along with y_label.
    :param track_pts:
    :return:
    """
    idx_array = tree['recovertex_tracks_idx'].array()
    weight_array = tree['recovertex_tracks_weight'].array()
    isHS_array = tree['recovertex_isHS'].array()

    vertex_data = []
    labels = []
    N_events = len(idx_array)
    if batch_range is None:
        batch_range = (0, N_events)
    if batch_range[1] > N_events or batch_range[0] < 0:
        raise ValueError("Invalid range request for processing batch!")
    print(f"Processing Batch from event {batch_range[0]} to event {batch_range[1]}")
    for i in range(batch_range[0], batch_range[1]):
        labels.extend(isHS_array[i][:-1])
        N_vertices = len(idx_array[i])
        event_pts = np.array(track_pts[i])
        event_weight_arrs = weight_array[i]
        event_idx_arrs = idx_array[i]
        # We ignore the last vertex as it is irrelevant
        for j in range(N_vertices-1):
            # Get proper tracks (i.e. weight > 0.75)
            vertex_weights = event_weight_arrs[j]
            vertex_idxs = event_idx_arrs[j]
            weight_mask = vertex_weights >= 0.75
            vertex_idxs = vertex_idxs[weight_mask]
            vertex_pts = event_pts[vertex_idxs]
            # get tracks with proper pt
            vertex_pts = vertex_pts[vertex_pts <= 50]
            sum_pt2 = np.sum(np.multiply(vertex_pts, vertex_pts))
            vertex_data.append(sum_pt2)
        if i % 100 == 0:
            print(f"Done event {i}.")

    headers = ["sum-pt2", 'y']
    final_data = np.column_stack((np.array(vertex_data), labels))
    final_data = np.vstack((headers, final_data))

    np.savetxt(outfile_path, final_data, delimiter=',', fmt='%s')
    print(f"Successfully saved data batch to '{outfile_path}'")


if __name__ == "__main__":
    # compute_track_pt(VBF_ROOT_PATH, "other_data_files/track_pt_vbf_8000-16000.csv", (8000, 16000))
    BATCH_SIZE = 500
    N_TRACKS = 50
    track_pts = load_pt("other_data_files/track_pt_vbf.csv")
    new_track_pts = load_pt("other_data_files/track_pt_vbf_8000-16000.csv")
    track_pts.extend(new_track_pts)
    print("Successfully loaded all track pts.")
    with uproot.open(ROOT_PATH) as file:
        tree = file["EventTree;1"]
        print("Successfully loaded tree")
    for k in range(16, 25):
        batch_range = (k*BATCH_SIZE, k*BATCH_SIZE + BATCH_SIZE)
        read_sum_pt2_to_csv(tree, track_pts, f"sum_pt2_batches/vbf_sum_pt2_{k}.csv", batch_range=batch_range)
        read_pt_to_csv(tree, track_pts, f"{N_TRACKS}_track_batches/{N_TRACKS}_track_vbf_pt_{k}.csv", N_TRACKS=50, batch_range=batch_range)
        read_z_coord_to_csv(tree, f"{N_TRACKS}_track_batches/{N_TRACKS}_track_vbf_z_{k}.csv", batch_range=batch_range)