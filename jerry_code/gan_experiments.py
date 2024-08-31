
from data_loading import load_csv, load_truth_hs_z, train_test_split
from plotting import plot_log_reco_err, plot_err_vs_pt2
from gan.gan_autoencoder import GanNetwork
import numpy as np
import keras


def data_loading_wrapper(feat_path, truth_hs_path, split_idx_num):
    """
    Arrays are labelled ttbar from prior code
    Though can be used to load vbf file as well
    Loads data to keep things in run_experiment() (de facto main function) clean
    :return:
    """
    data, y = load_csv(feat_path)
    hs_z_data = load_truth_hs_z(truth_hs_path)
    # Note: after here, ttbar y is only for testing data
    train_set, test_set, y, hs_zs = train_test_split(data, y,
                            split_idx_num=split_idx_num, remove_training_hs=True, truth_hs_data=hs_z_data)
    # split into features
    pt_idxs = np.arange(0, 100, 2)
    train = train_set[:, pt_idxs]
    X = test_set[:, pt_idxs]
    reco_zs = test_set[:, 100]
    event_nums = test_set[:, 101].astype(int)
    # for now, we'll calculate pt2
    pt2 = np.sum(X ** 2, axis=1)

    return train, X, y, pt2, reco_zs, event_nums, hs_zs


def evaluate_results(pt2, errs, y, reco_zs, hs_zs, model_name="model", dataset_name="Data"):
    # Get efficiencies vs density
    pt2_eff, pt2_midpts, pt2_std = get_efficiencies_vs_density(pt2, y, reco_zs, hs_zs,
                                                               num_bins=10, plot_hist=False, algo_name="Sum-pt2")
    model_eff, model_midpts, model_std = get_efficiencies_vs_density(errs, y, reco_zs, hs_zs,
                                                                     num_bins=10, plot_hist=False, algo_name=model_name)

    # Plot the efficiencies
    line_plot([pt2_midpts, model_midpts], [pt2_eff, model_eff], [pt2_std, model_std],
              ['Sum-pt2', model_name], title="Efficiency vs. Density for "+dataset_name, xlabel="Vertex Density",
              ylabel="Efficiency")

    # Print Average Efficiency Scores
    print("\n*** Printing Recall Scores ***")
    print(f"{'':10} {model_name:10} {'SUM-PT2':10}")
    print(f"{dataset_name:10} {np.mean(model_eff):<10.4f} {np.mean(pt2_eff):<10.4f}")

    # Now, plot the log reco errors and err vs pt2 plots
    # separate pu and hs errors for plotting
    hs_mask = y == 1
    pu_err = errs[~hs_mask]
    hs_err = errs[hs_mask]
    pu_pt2 = pt2[~hs_mask]
    hs_pt2 = pt2[hs_mask]

    # Plot log errors
    # Random sample indices for pu vertices (we don't want to plot all of them, or else it'll be too cluttered)
    rand_pu_idxs = np.random.choice(len(pu_err), 10000, replace=False)
    pu_err = pu_err[rand_pu_idxs]
    pu_pt2 = pu_pt2[rand_pu_idxs]

    plot_log_reco_err(pu_err, hs_err, dataset_name=dataset_name)
    plot_err_vs_pt2([pu_err, hs_err], [pu_pt2, hs_pt2], ["PU", "HS"], dataset_name=dataset_name)


def run_gan_experiment():
    # *** Loading Data ***
    ttbar_train, ttbar_X, ttbar_y, ttbar_pt2, ttbar_reco_zs, ttbar_event_nums, ttbar_hs_zs = data_loading_wrapper(
        "data_files/ttbar_small_500e.csv",
        "data_files/ttbar_hs_truth_z.csv", 300)

    # *** Training / saving / loading model, then doing inference ***
    model = GanNetwork(input_dim=50, latent_dim=5)
    model.compile(d_optimizer=keras.optimizers.Adam(1e-4), a_optimizer=keras.optimizers.Adam(1e-4))
    model.fit(ttbar_train, ttbar_train, epochs=20, batch_size=256)
    model.save_model("models/gan_full_5d_code")
    # model = GanNetwork.load_model("models/gan_full_5d_code")
    ttbar_scores = model.get_discriminator_predictions(ttbar_X)

    # *** Evaluating Model ***
    evaluate_results(ttbar_pt2, ttbar_scores, ttbar_y, ttbar_reco_zs, ttbar_hs_zs, model_name="Critic",
                     dataset_name="TTBAR")


if __name__ == "__main__":
    run_gan_experiment()