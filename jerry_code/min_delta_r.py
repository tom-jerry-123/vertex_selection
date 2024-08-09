"""
Computations for min dr features here
"""


import numpy as np


def compute_track_eta(tree, event_range):
    theta_array = tree['track_theta'].array()
    eta_lst = []
    for i in range(event_range[0], event_range[1]):
        thetas = np.array(theta_array[i])
        etas = - np.log(np.tan(thetas/2))
        eta_lst.append(etas)
    print(f"Completed Eta Computations from event {event_range[0]} to event {event_range[1]}")
    return eta_lst


def compute_delta_R(tree, event_range):
    # First, compute track etas as we need them to find min delta r
    eta_lst = compute_track_eta(tree, event_range)
    # Collect other necessary data from the tree
    jet_pt_array = tree['jet_pt'].array()
    jet_phi_array = tree['jet_phi'].array()
    jet_eta_array = tree['jet_eta'].array()
    track_phi_array = tree['track_phi'].array()

    delta_R_data = []
    for i in range(event_range[0], event_range[1]):
        mask = jet_pt_array[i] > 30
        jet_phis = np.array(jet_phi_array[i][mask])
        jet_etas = np.array(jet_eta_array[i][mask])
        track_phis = np.array(track_phi_array[i])
        track_etas = np.array(eta_lst[i - event_range[0]])
        # Compute min r values
        Dphi_matrix = track_phis[np.newaxis, :] - jet_phis[:, np.newaxis]
        Dphi_matrix[Dphi_matrix > np.pi] -= 2*np.pi
        Dphi_matrix[Dphi_matrix < -np.pi] += 2*np.pi
        R_matrix = Dphi_matrix ** 2 + (track_etas[np.newaxis, :] - jet_etas[:, np.newaxis])**2
        R_matrix = R_matrix ** 0.5
        R_matrix = np.vstack((R_matrix, np.full((R_matrix.shape[1],), 5)))
        delta_Rs = np.min(R_matrix, axis=0)
        delta_R_data.append(delta_Rs)

    print(f"Completed delta-R computations from event {event_range[0]} to event {event_range[1]}")

    return delta_R_data


def read_features_to_csv(tree, out_path, n_tracks, event_range):
    """
    Reads the pts and Rs to a csv-file. Top N_tracks pt and their corresponding delta-R
    :param out_path: Name of output file
    :param batch_range: two-tuple specifying event indices of start and end of range to process
    :return:
    """
    idx_array = tree['recovertex_tracks_idx'].array()
    weight_array = tree['recovertex_tracks_weight'].array()
    isHS_array = tree['recovertex_isHS'].array()

    track_Drs = compute_delta_R(tree, event_range)
    track_pts = []
    vertex_data = []
    labels = []
    N_events = len(idx_array)
    if event_range is None:
        event_range = (0, N_events)
    if event_range[1] > N_events or event_range[0] < 0:
        raise ValueError("Invalid range request for processing event!")
    print(f"Processing event from event {event_range[0]} to event {event_range[1]}")
    for i in range(event_range[0], event_range[1]):
        labels.extend(isHS_array[i][:-1])
        N_vertices = len(idx_array[i])
        event_pts = np.array(track_pts[i])
        event_Drs = np.array(track_Drs[i - event_range[0]])
        event_weight_arrs = weight_array[i]
        event_idx_arrs = idx_array[i]
        for j in range(N_vertices-1):
            # Get proper tracks (i.e. weight > 0.75)
            vertex_weights = event_weight_arrs[j]
            vertex_idxs = event_idx_arrs[j]
            weight_mask = vertex_weights >= 0.75
            vertex_idxs = vertex_idxs[weight_mask]
            vertex_pts = event_pts[vertex_idxs]
            vertex_dRs = event_Drs[vertex_idxs]
            # get tracks with proper pt
            pt_mask = vertex_pts <= 50
            vertex_pt_R = np.column_stack((vertex_pts[pt_mask], vertex_dRs[pt_mask]))
            # sort tracks by pt in descending order. Each min Dr immediately follows associated pt
            sorted_indices = np.argsort(vertex_pt_R[:, 0])[::-1]
            vertex_pt_R = vertex_pt_R[sorted_indices]
            vertex_pt_R = vertex_pt_R.flatten()
            if len(vertex_pt_R) >= n_tracks*2:
                vertex_pt_R = vertex_pt_R[:n_tracks*2]
            else:
                vertex_pt_R = np.pad(vertex_pt_R, (0, n_tracks*2 - len(vertex_pt_R)), mode='constant')
            vertex_data.append(vertex_pt_R)
        if i % 100 == 0:
            print(f"Done event {i}.")

    headers = [("pt_" if i % 2 == 0 else "Dr_") + str(i // 2) for i in range(n_tracks*2)]
    headers.append("y")
    final_data = np.column_stack((np.array(vertex_data), labels))
    final_data = np.vstack((headers, final_data))

    np.savetxt(out_path, final_data, delimiter=',', fmt='%s')
    print(f"Successfully saved data batch to '{out_path}'")