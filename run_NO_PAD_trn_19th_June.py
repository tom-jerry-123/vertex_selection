import pickle
import torch
import torch.nn as nn
import time
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split

'''
train_data = [torch.tensor([ 9.1211e-01,  1.0653e+00,  2.3662e+00,  2.9390e+00,  2.3239e+00, 1.0859e+00,  1.0609e+00,  1.3664e+00,  9.2449e-01,  2.0988e+01, 2.9190e+00,  5.3468e+00,  4.4847e+01,  1.8029e+00,  3.4218e+00,1.0059e+00,  1.9215e+00,  1.0949e+00,  1.6624e+01,  1.5062e+00,-3.4613e+00,  1.8925e+00, -1.8588e+00,  1.5234e+00,  1.6553e+00,-1.5568e+00, -1.5160e+00, -1.5056e+00, -2.2351e+00,  1.2099e+00,1.4728e+00,  3.6967e-01,  1.4616e-01, -7.7985e-01,  3.0095e-01, 1.7931e+00, -5.9868e-01,  3.0733e+00,  1.1920e+00,  1.3879e+00, 1.0000e+00,  1.0000e+00,  1.0000e+00,  1.0000e+00,  1.0000e+00, 3.5913e-01,  3.6201e-01,  3.1567e-01,  1.0000e+00,  3.8095e-02, 1.0000e+00,  1.4121e-01,  9.2469e-02,  1.0000e+00,  6.7676e-02, 1.0000e+00,  7.9343e-01,  1.0000e+00,  5.3731e-02,  1.0000e+00]), 
torch.tensor([ 1.1221,  0.9610,  1.5986,  2.2752,  1.5112,  1.8867,  1.0636,  0.9275, 1.5850,  2.6268,  1.0660,  3.9252,  3.9860,  1.5659,  1.5081, 10.5770, 3.5393,  1.1358,  3.0528,  1.0030, 10.9143,  1.3539,  1.0590, 16.3011, 2.8047,  3.8725,  2.7263,  3.8176,  2.6596,  1.4966,  1.5429,  0.9380,10.6840,  7.6728,  7.4733,  4.2017,  1.5724,  2.9604,  1.8346,  1.0043, 3.6985,  2.7000,  2.4642,  2.3090,  1.1966,  2.8055,  2.4208,  2.4668, -1.8623, -1.8621,  1.7798,  1.8768,  2.1788,  1.5923,  1.7435,  3.0928, -3.5497, -1.1040, -2.1631,  1.4008, -0.4276, -0.8002, -0.8532,  0.7783, 1.0712,  0.7560, -0.9714,  0.7924, -0.5874,  0.5347,  1.1983, -0.8726, 0.8157, -0.4620,  0.4733,  0.8412, -0.6789, -0.3736,  0.8278,  0.7407, 0.7115,  1.9356, -0.8418, -1.0891, -0.8440, -0.7764,  0.8576,  1.0677,-0.4163,  1.1559,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000, 1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,0.6887,  1.0000,  1.0000,  1.0000,  0.1833,  1.0000,  0.0693,  1.0000,0.2301,  0.0651,  1.0000,  1.0000,  1.0000,  0.3025,  1.0000,  1.0000, 0.1254,  1.0000,  1.0000,  0.0987,  0.1296,  1.0000,  1.0000,  1.0000,1.0000,  1.0000,  1.0000,  0.1046,  0.6370,  1.0000,  1.0000]), 
torch.tensor([ 1.6733e+00,  1.5321e+00,  3.1321e+00,  6.8539e+00,  1.9093e+00, 1.3707e+00,  3.0404e+01,  1.8074e+00,  5.2639e+00,  5.8103e+00, 3.9200e+00,  2.2174e+00,  7.2093e+00,  1.2937e+00,  1.6291e+01, 1.0943e+00,  1.1175e+01,  9.1676e-01,  3.1437e+00,  4.4250e+00, 9.9428e-01,  2.6958e+00,  3.0217e+00,  1.1811e+00,  2.1589e+00, 1.2341e+00,  3.0380e+00,  1.2815e+00,  9.5159e-01,  1.1008e+00, 1.7640e+00,  9.8324e-01, -2.7238e+00,  3.4199e+00,  2.9563e+00, 2.8446e+00, -3.2631e+00, -3.4695e+00,  1.8440e+00,  2.3375e+00, 2.2759e+00,  2.2357e+00,  1.8915e+00,  2.0437e+00,  2.0014e+00, 1.5947e+00,  2.0133e+00, -1.8515e+00,  2.1333e+00, -3.6764e+00,-1.3037e+00,  2.7787e+00, -1.5277e+00,  1.5836e+00,  2.6547e+00,-1.5483e+00,  2.7647e+00, -3.8859e-01,  2.2358e+00,  3.6983e+00, 1.1255e+00, -9.6912e-01, -1.1792e+00,  1.8775e+00,  1.0000e+00, 1.0000e+00,  6.9692e-01,  5.8570e-01,  1.0000e+00,  1.0000e+00, 1.2406e-01,  1.5223e-01,  9.9534e-02,  3.0614e-02,  3.1748e-02, 2.2531e-01,  1.0972e-01,  1.0000e+00,  9.6486e-02,  1.0000e+00, 5.8653e-02,  1.0000e+00,  1.0000e+00,  6.9285e-01,  1.0000e+00, 3.5387e-01,  7.4728e-01,  1.0000e+00,  5.6075e-01,  1.0000e+00, 2.4731e-02,  1.0000e+00,  1.0000e+00,  1.0000e+00,  1.0000e+00, 5.0346e-02]), 
torch.tensor([ 9.1211e-01,  1.0653e+00,  2.3662e+00,  2.9390e+00,  2.3239e+00, 1.0859e+00,  1.0609e+00,  1.3664e+00,  9.2449e-01,  2.0988e+01, 2.9190e+00,  5.3468e+00,  4.4847e+01,  1.8029e+00,  3.4218e+00,1.0059e+00,  1.9215e+00,  1.0949e+00,  1.6624e+01,  1.5062e+00,-3.4613e+00,  1.8925e+00, -1.8588e+00,  1.5234e+00,  1.6553e+00,-1.5568e+00, -1.5160e+00, -1.5056e+00, -2.2351e+00,  1.2099e+00,1.4728e+00,  3.6967e-01,  1.4616e-01, -7.7985e-01,  3.0095e-01, 1.7931e+00, -5.9868e-01,  3.0733e+00,  1.1920e+00,  1.3879e+00, 1.0000e+00,  1.0000e+00,  1.0000e+00,  1.0000e+00,  1.0000e+00, 3.5913e-01,  3.6201e-01,  3.1567e-01,  1.0000e+00,  3.8095e-02, 1.0000e+00,  1.4121e-01,  9.2469e-02,  1.0000e+00,  6.7676e-02, 1.0000e+00,  7.9343e-01,  1.0000e+00,  5.3731e-02,  1.0000e+00]), 
torch.tensor([ 1.1221,  0.9610,  1.5986,  2.2752,  1.5112,  1.8867,  1.0636,  0.9275, 1.5850,  2.6268,  1.0660,  3.9252,  3.9860,  1.5659,  1.5081, 10.5770, 3.5393,  1.1358,  3.0528,  1.0030, 10.9143,  1.3539,  1.0590, 16.3011, 2.8047,  3.8725,  2.7263,  3.8176,  2.6596,  1.4966,  1.5429,  0.9380,10.6840,  7.6728,  7.4733,  4.2017,  1.5724,  2.9604,  1.8346,  1.0043, 3.6985,  2.7000,  2.4642,  2.3090,  1.1966,  2.8055,  2.4208,  2.4668, -1.8623, -1.8621,  1.7798,  1.8768,  2.1788,  1.5923,  1.7435,  3.0928, -3.5497, -1.1040, -2.1631,  1.4008, -0.4276, -0.8002, -0.8532,  0.7783, 1.0712,  0.7560, -0.9714,  0.7924, -0.5874,  0.5347,  1.1983, -0.8726, 0.8157, -0.4620,  0.4733,  0.8412, -0.6789, -0.3736,  0.8278,  0.7407, 0.7115,  1.9356, -0.8418, -1.0891, -0.8440, -0.7764,  0.8576,  1.0677,-0.4163,  1.1559,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000, 1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,0.6887,  1.0000,  1.0000,  1.0000,  0.1833,  1.0000,  0.0693,  1.0000,0.2301,  0.0651,  1.0000,  1.0000,  1.0000,  0.3025,  1.0000,  1.0000, 0.1254,  1.0000,  1.0000,  0.0987,  0.1296,  1.0000,  1.0000,  1.0000,1.0000,  1.0000,  1.0000,  0.1046,  0.6370,  1.0000,  1.0000]), 
torch.tensor([ 1.6733e+00,  1.5321e+00,  3.1321e+00,  6.8539e+00,  1.9093e+00, 1.3707e+00,  3.0404e+01,  1.8074e+00,  5.2639e+00,  5.8103e+00, 3.9200e+00,  2.2174e+00,  7.2093e+00,  1.2937e+00,  1.6291e+01, 1.0943e+00,  1.1175e+01,  9.1676e-01,  3.1437e+00,  4.4250e+00, 9.9428e-01,  2.6958e+00,  3.0217e+00,  1.1811e+00,  2.1589e+00, 1.2341e+00,  3.0380e+00,  1.2815e+00,  9.5159e-01,  1.1008e+00, 1.7640e+00,  9.8324e-01, -2.7238e+00,  3.4199e+00,  2.9563e+00, 2.8446e+00, -3.2631e+00, -3.4695e+00,  1.8440e+00,  2.3375e+00, 2.2759e+00,  2.2357e+00,  1.8915e+00,  2.0437e+00,  2.0014e+00, 1.5947e+00,  2.0133e+00, -1.8515e+00,  2.1333e+00, -3.6764e+00,-1.3037e+00,  2.7787e+00, -1.5277e+00,  1.5836e+00,  2.6547e+00,-1.5483e+00,  2.7647e+00, -3.8859e-01,  2.2358e+00,  3.6983e+00, 1.1255e+00, -9.6912e-01, -1.1792e+00,  1.8775e+00,  1.0000e+00, 1.0000e+00,  6.9692e-01,  5.8570e-01,  1.0000e+00,  1.0000e+00, 1.2406e-01,  1.5223e-01,  9.9534e-02,  3.0614e-02,  3.1748e-02, 2.2531e-01,  1.0972e-01,  1.0000e+00,  9.6486e-02,  1.0000e+00, 5.8653e-02,  1.0000e+00,  1.0000e+00,  6.9285e-01,  1.0000e+00, 3.5387e-01,  7.4728e-01,  1.0000e+00,  5.6075e-01,  1.0000e+00, 2.4731e-02,  1.0000e+00,  1.0000e+00,  1.0000e+00,  1.0000e+00, 5.0346e-02])
]

val_data = [torch.tensor([ 100,  1.2306e+00,  1.3267e+00,  9.7876e-01,  1.0369e+00,9.1854e-01,  1.8418e+00,  9.1533e-01,  2.8519e+00,  1.9503e+00,3.4430e+00, -1.5766e+00, -1.3944e+00, 1.0000e+03,  1.0000e+03,  1.0000e+03,1.0000e+03,  1.0000e+03,  1.0000e+03,  1.0000e+03]), 
torch.tensor([ 9.9193e-01,  1.1648e+00,  9.8254e-01,  9.6132e-01,  1.8375e+00,3.8994e+00,  3.6014e+00,  3.2145e+00, -1.3591e+00, -5.3188e-01,1.0000e+03,  1.0000e+03,  1.0000e+03,  1.0000e+03,  1.0000e+03]), 
torch.tensor([ 1.2140e+00,  1.0526e+00,  9.4315e-01,  1.0019e+00,  1.0873e+00,1.3110e+00,  1.0040e+00,  3.7680e+00,  2.1524e+00, -1.4864e+00, -8.3999e-01,  2.6346e-01, -5.0837e-01, -1.1333e+00,  1.0000e+03,1.0000e+03,  1.0000e+03,  1.0000e+03,  1.0000e+03,  1.0000e+03,1.0000e+03]), 
torch.tensor([ 1.1221,  0.9610,  1.5986,  2.2752,  1.5112,  1.8867,  1.0636,  0.9275, 1.5850,  2.6268,  1.0660,  3.9252,  3.9860,  1.5659,  1.5081, 10.5770, 3.5393,  1.1358,  3.0528,  1.0030, 10.9143,  1.3539,  1.0590, 16.3011, 2.8047,  3.8725,  2.7263,  3.8176,  2.6596,  1.4966,  1.5429,  0.9380,10.6840,  7.6728,  7.4733,  4.2017,  1.5724,  2.9604,  1.8346,  1.0043, 3.6985,  2.7000,  2.4642,  2.3090,  1.1966,  2.8055,  2.4208,  2.4668, -1.8623, -1.8621,  1.7798,  1.8768,  2.1788,  1.5923,  1.7435,  3.0928, -3.5497, -1.1040, -2.1631,  1.4008, -0.4276, -0.8002, -0.8532,  0.7783, 1.0712,  0.7560, -0.9714,  0.7924, -0.5874,  0.5347,  1.1983, -0.8726, 0.8157, -0.4620,  0.4733,  0.8412, -0.6789, -0.3736,  0.8278,  0.7407, 0.7115,  1.9356, -0.8418, -1.0891, -0.8440, -0.7764,  0.8576,  1.0677,-0.4163,  1.1559,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000, 1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,  1.0000,0.6887,  1.0000,  1.0000,  1.0000,  0.1833,  1.0000,  0.0693,  1.0000,0.2301,  0.0651,  1.0000,  1.0000,  1.0000,  0.3025,  1.0000,  1.0000, 0.1254,  1.0000,  1.0000,  0.0987,  0.1296,  1.0000,  1.0000,  1.0000,1.0000,  1.0000,  1.0000,  0.1046,  0.6370,  1.0000,  1.0000]), 
torch.tensor([ 1.6733e+00,  1.5321e+00,  3.1321e+00,  6.8539e+00,  1.9093e+00, 1.3707e+00,  3.0404e+01,  1.8074e+00,  5.2639e+00,  5.8103e+00, 3.9200e+00,  2.2174e+00,  7.2093e+00,  1.2937e+00,  1.6291e+01, 1.0943e+00,  1.1175e+01,  9.1676e-01,  3.1437e+00,  4.4250e+00, 9.9428e-01,  2.6958e+00,  3.0217e+00,  1.1811e+00,  2.1589e+00, 1.2341e+00,  3.0380e+00,  1.2815e+00,  9.5159e-01,  1.1008e+00, 1.7640e+00,  9.8324e-01, -2.7238e+00,  3.4199e+00,  2.9563e+00, 2.8446e+00, -3.2631e+00, -3.4695e+00,  1.8440e+00,  2.3375e+00, 2.2759e+00,  2.2357e+00,  1.8915e+00,  2.0437e+00,  2.0014e+00, 1.5947e+00,  2.0133e+00, -1.8515e+00,  2.1333e+00, -3.6764e+00,-1.3037e+00,  2.7787e+00, -1.5277e+00,  1.5836e+00,  2.6547e+00,-1.5483e+00,  2.7647e+00, -3.8859e-01,  2.2358e+00,  3.6983e+00, 1.1255e+00, -9.6912e-01, -1.1792e+00,  1.8775e+00,  1.0000e+00, 1.0000e+00,  6.9692e-01,  5.8570e-01,  1.0000e+00,  1.0000e+00, 1.2406e-01,  1.5223e-01,  9.9534e-02,  3.0614e-02,  3.1748e-02, 2.2531e-01,  1.0972e-01,  1.0000e+00,  9.6486e-02,  1.0000e+00, 5.8653e-02,  1.0000e+00,  1.0000e+00,  6.9285e-01,  1.0000e+00, 3.5387e-01,  7.4728e-01,  1.0000e+00,  5.6075e-01,  1.0000e+00, 2.4731e-02,  1.0000e+00,  1.0000e+00,  1.0000e+00,  1.0000e+00, 5.0346e-02])
]

train_labels = torch.tensor([0, 0, 0, 0, 0, 0])
val_labels = torch.tensor([0, 0, 0])
'''


input_HS_file_tt = 'HS_tt_full_no_pad_08.pkl'
input_PU_file_tt = 'PU_tt_full_no_pad_08.pkl'
output_HS_file_VBF = 'HS_VBF_full_no_pad_08.pkl'
output_PU_file_VBF = 'PU_VBF_full_no_pad_08.pkl'
saved_model_name = 'trn_pt_eta_15_epoch_08_17th_june.pth'

# Load data
with open(input_PU_file_tt, 'rb') as f:
    PU_data = pickle.load(f)
    
PU_data_no_pad = PU_data [:100000]

# Prepare data
train_data_tracks = []
train_labels = []
train_i_values = []
train_j_values = []
error_indices = []

for idx, entry in enumerate(PU_data_no_pad):
    try:
        sumpt = entry[0]
        track_pt_list = entry[1]
        track_eta_list = entry[2]
        minDr_jet_list = entry[3]
        close_jet_pt_list = entry[6]
        i_value = entry[4]
        j_value = entry[5]
        sumptW = entry[7]
        tracks_tensor = torch.tensor([track_pt_list, track_eta_list])
        #tracks_tensor = torch.tensor([track_pt_list, track_eta_list, minDr_jet_list])
        #combined_data = torch.cat([torch.tensor([sumpt]), tracks_tensor.flatten()])
        combined_data = tracks_tensor.flatten()
        train_data_tracks.append(combined_data)
        train_labels.append(0)
        train_i_values.append(i_value)
        train_j_values.append(j_value)
    except Exception as e:
        error_indices.append((idx, str(e)))
        continue

train_data = train_data_tracks
print(" train_data :", train_data[0])
train_labels = torch.tensor(train_labels)
train_i_values = torch.tensor(train_i_values)
train_j_values = torch.tensor(train_j_values)

train_data, val_data, train_labels, val_labels, train_i_values, val_i_values, train_j_values, val_j_values = train_test_split(
    train_data, train_labels, train_i_values, train_j_values, test_size=0.2, random_state=42)

train_data, test_PU_data, train_labels, test_PU_labels, train_i_values, test_PU_i_values, train_j_values, test_PU_j_values = train_test_split(
    train_data, train_labels, train_i_values, train_j_values, test_size=0.1, random_state=42)

print("Train Data Shape:", len(train_data))
print("train_labels Shape:", len(train_labels))
print("train_i_values Shape:", len(train_i_values))
print("train_j_values Shape:", len(train_j_values))
print("val Data Shape:", len(val_data))
print("val_labels Shape:", len(val_labels))
print("val_i_values Shape:", len(val_i_values))
print("val_j_values Shape:", len(val_j_values))
print("test PU Data Shape:", len(test_PU_data))
print("test_PU_labels Shape:", len(test_PU_labels))
print("test_PU_i_values Shape:", len(test_PU_i_values))
print("test_PU_j_values Shape:", len(test_PU_j_values))


latent_dim = 8
        
class TransformerAnomalyDetector_v0(nn.Module):
    def __init__(self, input_dim, latent_dim, nhead=1, num_layers=1, dim_feedforward=32, dropout=0.1):
        super(TransformerAnomalyDetector_v0, self).__init__()
        self.embedding = nn.Linear(input_dim, latent_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        #print("self.transformer_encoder :", self.transformer_encoder)
        self.decoder = nn.Linear(latent_dim, input_dim)
        #print("self.decoder :", self.decoder)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        
    def forward(self, sequences, lengths):
        embedded = [self.embedding(seq.unsqueeze(1)) for seq in sequences]
        #print("embedded :", embedded)
        padded_embedded = torch.nn.utils.rnn.pad_sequence(embedded, batch_first=True)
        print("padded_embedded :", padded_embedded)

        max_len = max(lengths)
        attention_masks = torch.zeros((len(sequences), max_len), dtype=torch.bool)
        for i, length in enumerate(lengths):
            attention_masks[i, :length] = 1
        print("attention_masks :", attention_masks)

        src = padded_embedded.permute(1, 0, 2)
        #print("src :", src)

        transformer_output = self.transformer_encoder(src, src_key_padding_mask=~attention_masks)
        #print("transformer_output 1 :", transformer_output)

        transformer_output = transformer_output.permute(1, 0, 2)  # Shape: [batch_size, max_len, latent_dim]
        #print("transformer_output 2:", transformer_output)

        outputs = []
        for i, length in enumerate(lengths):
            sequence_output = transformer_output[i, :length]
            sequence_decoded = self.decoder(sequence_output)
            outputs.append(sequence_decoded)
            
        #print("outputs:", outputs)
        return outputs


#def train_model(model, train_data, train_labels, train_i_values, train_j_values, criterion, optimizer, num_epochs=10, batch_size=256):

def train_model(model, train_data, train_labels, val_data, val_labels, criterion, optimizer, num_epochs=15, batch_size=256):

    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0
        for idx in range(0, len(train_data), batch_size):
            end_idx = min(idx + batch_size, len(train_data))
            chunk_train_data = train_data[idx:end_idx]
            chunk_train_labels = train_labels[idx:end_idx]
            #chunk_train_i_values = train_i_values[idx:end_idx]
            #chunk_train_j_values = train_j_values[idx:end_idx]
            #print("idx :", idx)

            train_lengths = torch.tensor([len(seq) for seq in chunk_train_data])
            
            optimizer.zero_grad()
            outputs = model(chunk_train_data, train_lengths)
            #print("outputs :", outputs)
            #print("chunk_train_data :", chunk_train_data)
            #print("chunk_train_data size :", len(chunk_train_data))

            #mod_outputs = [tensor.view(-1).detach() for tensor in outputs]
            mod_outputs = [tensor.view(-1) for tensor in outputs]
            mod_outputs2 = [tensor.view(-1, model.decoder.out_features) for tensor in outputs]
            #print("mod_outputs :", mod_outputs)
            #print("mod_outputs2 :", mod_outputs2)


            losses = []
            batch_loss = 0
            for i in range(len(chunk_train_data)):
                loss = criterion(mod_outputs[i], chunk_train_data[i])
                losses.append(loss.item())
                #loss.backward()
                running_loss += loss.item()
                batch_loss += loss
                
            batch_loss.backward()  # Compute the gradient for the whole batch
            optimizer.step()
 
        epoch_loss = running_loss / len(train_data)
        train_losses.append(epoch_loss)


        # Validation phase
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for idx in range(0, len(val_data), batch_size):
                end_idx = min(idx + batch_size, len(val_data))
                chunk_val_data = val_data[idx:end_idx]
                chunk_val_labels = val_labels[idx:end_idx]

                val_lengths = torch.tensor([len(seq) for seq in chunk_val_data])
                #print("val_lengths  :", val_lengths)

                val_outputs = model(chunk_val_data, val_lengths)
                #print("chunk_val_data size :", len(chunk_val_data))
                #print("val_outputs  :", val_outputs)
                #mod_val_outputs = [tensor.view(-1).detach() for tensor in val_outputs]
                mod_val_outputs = [tensor.view(-1) for tensor in val_outputs]

                #print("mod_val_outputs  :", mod_val_outputs)
                
                mod_val_losses = []
                for i in range(len(chunk_val_data)):
                    loss = criterion(mod_val_outputs[i], chunk_val_data[i])
                    mod_val_losses.append(loss.item())
                    val_running_loss += loss.item()
                    
                #print("mod_val_losses  :", mod_val_losses)

        epoch_val_loss = val_running_loss / len(val_data)
        #print("val_running_loss :", val_running_loss, " len(val_data) :",  len(val_data), " epoch_val_loss :", epoch_val_loss)
        val_losses.append(epoch_val_loss)

        # Ensure losses are not zero before taking the logarithm
        if epoch_loss == 0:
            avg_loss_log = float('-inf')  # or a very large negative value
        else:
            avg_loss_log = np.log(epoch_loss)

        if epoch_val_loss == 0:
            avg_val_loss_log = float('-inf')  # or a very large negative value
        else:
            avg_val_loss_log = np.log(epoch_val_loss)
        
        end_time = time.time()
        epoch_time = end_time - start_time
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss_log:.4f}, Val Loss: {avg_val_loss_log:.4f}, Time: {epoch_time:.2f} seconds")
    return train_losses, val_losses
    
train_data = [torch.tensor(data, requires_grad=True) for data in train_data]
val_data = [torch.tensor(data, requires_grad=False) for data in val_data]

# Hyperparameters
input_dim = 1
d_model = 8
nhead = 1
dim_feedforward = 32
num_layers = 2
learning_rate = 0.001
criterion = nn.MSELoss()
num_epochs = 15
batch_size = 256

#model = TransformerAnomalyDetector_v0(input_dim, d_model, nhead, dim_feedforward, num_layers)
model = TransformerAnomalyDetector_v0(input_dim, latent_dim, nhead, dim_feedforward, num_layers)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print("Training starts")

#train_model(model, train_data, train_labels, train_i_values, train_j_values, criterion, optimizer, num_epochs, batch_size)
train_losses, val_losses = train_model(model, train_data, train_labels, val_data, val_labels, criterion, optimizer, num_epochs, batch_size)

torch.save(model.state_dict(), saved_model_name)
print("Training completed on all data.")


