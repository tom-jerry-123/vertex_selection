"""
Main file.

I processed track pt data by batches of 500 events and put them in csv
I did not add code for efficiency plots here as I need the root files to make them.
Run this file for model training and results. No modifications to code needed
"""


from autoencoder import Autoencoder
from data_loading import *
from plotting import *


if __name__ == "__main__":
    """
    Trains an autoencoder model and tests it using data from folder and files
    :param autoencoder: autoencoder neural net to use
    :param folder: folder of data file. if it exists
    :param file: file prefix (assuming files are numbered at the end to indicate batch number)
    :return:
    """
    # Load training, testing data
    ttbar_train, ttbar_x, ttbar_y = load_train_test("data_files/50_track_ttbar_pt_", (0, 2), (2, 3))
    ttbar_pt2, _trash = load_data("data_files/ttbar_sum_pt2_", (2, 3))

    last = 1
    test_thresholds = [last + i * 0.2 for i in range(0, 500)]

    last = 5
    pt_thresholds = [last + i for i in range(200)]

    model = Autoencoder(input_dim=50, code_dim=3, architecture=(32,), regularization="L2")
    model.train_model(ttbar_train, epochs=15, plot_valid_loss=False)

    # Load weights of saved model
    # model.load_weights("models/final_model.h5")

    # Compute reconstruction errors
    rec_err = model.reconstruction_error(ttbar_x)

    # Get classifications. The y-labels are used to separate data by event
    ttbar_encoder_yhat = Autoencoder.get_classification(rec_err, ttbar_y)
    ttbar_base_yhat = Autoencoder.get_classification(ttbar_pt2, ttbar_y)

    # Print Recall Values
    ttbar_encoder_recall = np.sum((ttbar_encoder_yhat == 1) & (ttbar_y == 1)) / np.sum((ttbar_y == 1))
    ttbar_base_recall = np.sum((ttbar_base_yhat == 1) & (ttbar_y == 1)) / np.sum((ttbar_y == 1))

    # Print Recall Scores
    print("\n*** Printing Recall Scores ***")
    print(f"{'':10} {'ENCODER':10} {'SUM-PT2':10}")
    print(f"{'TTBAR':10} {ttbar_encoder_recall:<10.4f} {ttbar_base_recall:<10.4f}")
    # print(f"{'TTBAR':10} {ttbar_encoder_recall:<10.4f} N/A")

    # Separate pu, vbf_hs, ttbar_hs data for roc curves
    # Do this for errors first
    ttbar_mask = ttbar_y == 1
    pu_err = rec_err[~ttbar_mask]
    ttbar_hs_err = rec_err[ttbar_mask]
    # Now, doing this for sum-pt2 values
    pu_pt2 = ttbar_pt2[~ttbar_mask]
    ttbar_hs_pt2 = ttbar_pt2[ttbar_mask]

    # Random sample indices for pu vertices (we don't want to plot all of them, or else it'll be too cluttered)
    rand_pu_idxs = np.random.choice(len(pu_err), int(len(pu_err) * 0.05), replace=False)
    # Plot log errors
    plot_log_reco_err(pu_err, ttbar_hs_err)
    # plot err vs pt2
    plot_err_vs_pt2(err_lst=[pu_err[rand_pu_idxs], ttbar_hs_err],
                    pt_lst=[pu_pt2[rand_pu_idxs], ttbar_hs_pt2], labels=['PU', 'TTBAR HS'])
