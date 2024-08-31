##############################################################
######## Event display code for vertices ################
######### Written by Wasikul Islam (Wasikul.islam@cern.ch) ########
##############################################################

import uproot
import matplotlib.pyplot as plt
import numpy as np
import math

root_file = uproot.open("C:/Users/jerry/Documents/Phys_Summer_Research/root_files/ttbar-full-data.root")
# root_file = uproot.open("C:/Users/jerry/Documents/Phys_Summer_Research/root_files/VBF_hist-Rel21sample_199files.root")
tree = root_file["EventTree;1"]

# Get the branches of interest
my_branches = tree.arrays(['truthvertex_z', 'track_prob', 'truthvertex_isHS','truthvertex_isPU','recovertex_isHS', 'recovertex_isPU', 'recovertex_sumPt2', 'recovertex_tracks_idx', 'recovertex_z', 'track_z0', 'track_status', 'track_qOverP', 'track_theta', 'track_phi'])

# Define the event number and vertex ID to focus on
event_num = 6631 #4147 #631
vtxID = 10  #1

print("isHS",my_branches.recovertex_isHS[event_num][vtxID] )
print("isPU",my_branches.recovertex_isPU[event_num][vtxID] )

vtx_z = my_branches.recovertex_z[event_num][vtxID]
sumpt = my_branches.recovertex_sumPt2[event_num][vtxID]
truth_z = my_branches.truthvertex_z[event_num][0]

selected_HS_vtx_id = None
max_sumpt = float('-inf')


# Iterate through the vertex z-values and sumpt values
for idx, z in enumerate(my_branches.recovertex_z[event_num]):
    if abs(z - truth_z) <= 2:
        sumpt_value = my_branches.recovertex_sumPt2[event_num][idx]
        print("reco vertex within 2 mm of truth z : ", z, "vtx id:", idx, "sumpt :", sumpt_value)
        # If the sumpt value is higher, update the selected vertex
        if  sumpt_value > max_sumpt:
            selected_HS_vtx_id = idx
            max_sumpt = my_branches.recovertex_sumPt2[event_num][idx]

# Print the selected vertex and its sumpt value
if selected_HS_vtx_id is not None:
    print(f"Selected Vertex ID: {selected_HS_vtx_id}")
    print(f"Sumpt of Selected Vertex: {max_sumpt}")
else:
    print("No vertices found within the specified range.")

if (vtxID == selected_HS_vtx_id):
    print("This is HS vertex")
else :
    print("This is PU vertex")

# Get the vertex z-coordinate

closest_vertices = []
# Get the recovertex_z values within the specified range
reco_vertices = []
for idx, z in enumerate(my_branches.recovertex_z[event_num]):
    if abs(z - vtx_z) <= 5:
        reco_vertices.append((z, idx))
        diff = abs(z - vtx_z)
        closest_vertices.append((idx, z, diff))

# Sort the closest_vertices list based on the differences
closest_vertices.sort(key=lambda x: x[2])
# Display the closest vertices
for idx, z, diff in closest_vertices:
    print(f"Vertex ID: {idx}, Z: {z}, Difference from vtx_z: {diff}")
    if (idx==selected_HS_vtx_id):
        print("closest HS vertex# :", idx)
    
truth_vertices = []
for idx, z in enumerate(my_branches.truthvertex_z[event_num]):
    if abs(z - vtx_z) <= 5:
        truth_vertices.append((z, idx))

# Get the tracks connected to the selected vertex
connected_tracks = my_branches.recovertex_tracks_idx[event_num][vtxID]

# Initialize lists to store track information
track_info = []

# Loop over the connected tracks
for idx in connected_tracks:
    track_z0 = my_branches.track_z0[event_num][idx]
    p = abs(1 / (my_branches.track_qOverP[event_num][idx]))
    track_eta = -np.log(math.tan((my_branches.track_theta[event_num][idx]) / 2))
    track_pT = (p / (np.cosh(track_eta))) / 1000
    track_phi = my_branches.track_phi[event_num][idx]
    z0 = track_z0 - vtx_z
    
    #print("track_pT :", track_pT, "track prob", my_branches.track_prob[event_num][idx], "track status", my_branches.track_status[event_num][idx])
    print("track_pT :", track_pT, "track eta", track_eta, "track status", my_branches.track_status[event_num][idx])

    pz = track_pT * math.sinh(track_eta)
    signX = track_eta / abs(track_eta)
    signY = math.sin(track_phi) / abs(math.sin(track_phi))
    theta = math.atan(track_pT / abs(pz))
    x = (track_pT / 2) * math.cos(theta) * signX
    y = (track_pT / 2) * math.sin(theta) * signY

    status = my_branches.track_status[event_num][idx]
    
    track_info.append([vtx_z, z0, x, y, status])

# Plot the graph
plt.figure(figsize=(12, 6))

for track in track_info:
    Z = track[0]
    z0 = track[1]
    x = track[2]
    y = track[3]
    status = track[4]
    if status == 0:
        color = 'blue'  # Set color to blue for hard-scatter tracks
    elif status == 1:
        color = 'red'
    elif status == 2:        
        color = 'green'  # Set color to red for pile-up tracks
    elif status == 3:        
        color = 'cyan' 
    else:
        color = 'black' 
    plt.plot([Z + z0, Z + z0 + x], [0, y], color=color) #, label='HS tracks' if isHS == 0 else 'PU tracks')
    
# Determine the x-coordinate based on the Z coordinate range
x_coord = vtx_z-4.5  # Set it to the minimum x-coordinate of the line_positions
y_coord = 0.9  # Set the y-coordinate for the text annotations
plt.text(x_coord, y_coord, f"Reco z = {vtx_z:.2f}", weight='bold', fontsize=12)
plt.text(x_coord, y_coord - 0.1, f"Truth z = {truth_z:.2f}", weight='bold', fontsize=12)
plt.text(x_coord, y_coord - 0.2, f"Sum $p_T^2$ = {sumpt:.2f}", weight='bold', fontsize=12)

#plt.text(x_coord, y_coord - 0.2, r"Sum $p_T^2$ = " + r"$/square$", weight='bold', fontsize=12)

# Plot the line for the recovertex_z values
reco_vertices_z = [z for z, _ in reco_vertices]
marker_colors_reco = ['red' if z == vtx_z else 'black' for z in reco_vertices_z]
plt.scatter(reco_vertices_z, [0] * len(reco_vertices_z), color=marker_colors_reco, marker='o', s=100)  # Set marker size to 100
plt.axhline(y=0.0, color='black', linestyle='--')

# Add a text annotation for the label
label_x = vtx_z + 3.5  # Adjust the x-coordinate for the label
label_y = 0.05  # Adjust the y-coordinate for the label
plt.text(label_x, label_y, 'Reco vertices', fontsize=12)
plt.text(label_x, label_y-0.75, 'Truth vertices', fontsize=12)
plt.text(label_x-3, 0.85, 'ATLAS Simulation Preliminary', fontsize=16, weight='bold', style='italic')

truth_vertices_z = [z for z, _ in truth_vertices]
marker_colors = ['red' if z == truth_z else 'black' for z in truth_vertices_z]
plt.scatter(truth_vertices_z, [-0.75] * len(truth_vertices_z), color=marker_colors, marker='|', s=100)  # Set marker size to 100
plt.axhline(y=-0.75, color='black', linestyle='--')

# Add legend for HS and PU tracks
hs_legend = plt.legend(handles=[plt.Line2D([], [], color='blue', label='HS tracks'),
                                plt.Line2D([], [], color='red', label='PU tracks')],
                                #plt.Line2D([], [], color='red', label='PU tracks'),
                                #plt.Line2D([], [], color='green', label='GEANT4=2 tracks')],
                                #GEANT4=2, Unlinked=3, Other=-1
                       loc='upper right', title='Track Types', bbox_to_anchor=(1.0, 0.9))
plt.gca().add_artist(hs_legend)  # Add the HS and PU tracks legend back to the axes

plt.ylim(-1.0, 1.0)
plt.xlim(vtx_z-5.0, vtx_z+5.0)
plt.title(f'Tracks connected to vertex {vtxID} in event {event_num}')
plt.xlabel('Z [mm]')
plt.ylabel('R [mm]')
plt.legend()
#plt.grid(True)
plt.savefig(f'fig_{event_num}_{vtxID}.png')
plt.show()
