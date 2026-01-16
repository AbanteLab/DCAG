#!/usr/bin/env python3

### scANVI for cell annotation and label transfer ###
# Following https://docs.scvi-tools.org/en/1.3.0/tutorials/notebooks/scrna/tabula_muris.html
#%%
import argparse
import tempfile
import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import seaborn as sns
import torch
from collections import Counter
from sklearn.model_selection import train_test_split
import multiprocessing

from utils import *

#%% args for interactive testing
# args = argparse.Namespace(
#     seed=3,
#     num_epochs=20,
#     dataset_path="/pool01/projects/abante_lab/cag_propagation/deep_dive_adata_v2_6k.h5",
#     # dataset_path="/pool01/projects/abante_lab/cag_propagation/deep_dive_10x_v1_6k_test0.5_adata.h5",
#     # dataset_path="/pool01/projects/abante_lab/cag_propagation/deep_dive_10x_test0.1_adata.h5",
#     # dataset_path="/pool01/projects/abante_lab/cag_propagation/handsaker_deepdive_thompson2020.h5",
#     # dataset_path="/pool01/projects/abante_lab/cag_propagation/handsaker_deepdive10X_thompson2020.h5",
#     # dataset_path="/pool01/projects/abante_lab/cag_propagation/multitech_supervised_adata_v2_6000.h5",
#     batch_size=5000,       
#     cuda=True,
#     multitech = False,
#     unseen_y_prop=0.0,      # If multitech true, this arg is ignored
#     plot=True,
#     latent_dim = 20,
#     n_hidden_layers = 2,
#     save_latent = True)
#%%
def main(args):
#%%
    superprint("Starting ...")

    # Load and pre-process data
    if args.multitech:
        _, _, _, _, _, _, adata = get_data_multitech(
            dataset_path=args.dataset_path, batch_size=args.batch_size, cuda=args.cuda, test_size=0.4, val_size=0.1, seed=args.seed
        )
        batch_key = "DONOR" # Can't use BATCH because 10X don't have batch info
        train_adata, test_adata = split_anndata_multitech(adata, batch_key, test_size=0.2, seed=args.seed)
        label_key = "CAGPHASE"
    else:
        if args.unseen_y_prop == 0.0:
            # Supervised learning
            _, _, _, _, _, _, adata = get_data_supervised(
                dataset_path=args.dataset_path, batch_size=args.batch_size, cuda=args.cuda, test_size=0.4, val_size=0.1, seed=args.seed
            )
            train_adata, test_adata = split_anndata(adata, test_size=0.2, seed=args.seed)
            label_key = "CAGPHASE"
            batch_key = "BATCH"

        else:
            # Masked learning
            _, _, _, _, _, _, adata = get_data_masked(
                dataset_path=args.dataset_path, batch_size=args.batch_size, cuda=args.cuda, test_size=0.4, val_size=0.1, masking_prop = args.unseen_y_prop, seed=args.seed
            )
            train_adata, test_adata = split_anndata_with_masking(adata, test_size=0.2, seed=args.seed)
            label_key = "CAGPHASE_MASKING"
            batch_key = "BATCH"

#%%
    scvi.settings.seed = args.seed
    device = "cuda" if args.cuda else "cpu"

    sc.set_figure_params(figsize=(6, 6), frameon=False)
    sns.set_theme()
    torch.set_float32_matmul_precision("high")

    NUM_WORKERS = max(1, int(0.7*(multiprocessing.cpu_count() - 1 )))
    scvi.settings.dataloader_num_workers = NUM_WORKERS

    # Train base scVI
    scvi.model.SCVI.setup_anndata(train_adata, layer="raw", batch_key=batch_key)
    scvi_model = scvi.model.SCVI(train_adata, n_layers=args.n_hidden_layers, n_latent=args.latent_dim)

    scvi_model.train(train_size=0.9, validation_size=0.1,
                    batch_size=args.batch_size, 
                    early_stopping=True,
                    max_epochs=args.num_epochs,
                    accelerator=device,
                    early_stopping_monitor="elbo_validation",
                    early_stopping_patience=20,
                    early_stopping_min_delta=0.005)

    #%% scANVI annotation transfer
    label_vals = train_adata.obs["CAGPHASE"].dropna().values  # labelled cells only
    counts = Counter(label_vals)
    min_count = min(counts.values())

    scanvi_model = scvi.model.SCANVI.from_scvi_model(
        scvi_model,
        adata=train_adata,
        unlabeled_category="unknown",
        labels_key=label_key,
    )
    scanvi_model.train(train_size=0.9, validation_size=0.1,
                    batch_size=args.batch_size, 
                    early_stopping=True,
                    n_samples_per_label=min_count,
                    max_epochs=args.num_epochs,
                    accelerator=device,
                    early_stopping_monitor="elbo_validation",
                    early_stopping_patience=20,
                    early_stopping_min_delta=0.005)

    SCANVI_LATENT_KEY = "X_scANVI"
    SCANVI_PREDICTION_KEY = "C_scANVI"

    train_adata.obsm[SCANVI_LATENT_KEY] = scanvi_model.get_latent_representation(train_adata)
    train_adata.obs[SCANVI_PREDICTION_KEY] = scanvi_model.predict(train_adata)
    #%%
    test_adata.obsm[SCANVI_LATENT_KEY] = scanvi_model.get_latent_representation(test_adata)
    test_adata.obs[SCANVI_PREDICTION_KEY] = scanvi_model.predict(test_adata)

    # %%
    superprint("Training complete! Evaluating model...")

    output_path = "/pool01/projects/abante_lab/cag_propagation/scanvi_results/"
    # output_path = "/gpfs/projects/ub212/cag_propagation/data/model_results/"

    model_sufix, dataset_name = get_model_sufix(scanvi_model, args)

    if args.plot:
        plot_scanvi_pca_latent(scanvi_model, train_adata, latent="Z", output_path=f"{output_path}{model_sufix}_pca_z_trainset.pdf", show_plot=True)
        plot_scanvi_pca_latent(scanvi_model, test_adata, latent="Z", output_path=f"{output_path}{model_sufix}_pca_z_testset.pdf", show_plot=True)

        plot_losses_scanvi(scanvi_model, f"{output_path}{model_sufix}_losses.pdf", show_plot=True)

    evaluate_model(scanvi_model, train_adata, test_adata, args, dataset_name, model_sufix, output_path=output_path)

    # Save latent representation in adata
    if args.save_latent == True:
        # Concatenate train and test adata to save full dataset
        adata = anndata.concat([train_adata, test_adata])
        adata.write_h5ad(f"{output_path}adata_with_latent_{model_sufix}.h5ad")
# %%
if __name__ == "__main__":

    assert pyro.__version__.startswith("1.9.1")
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="single-cell Anotation using Variational Inference"
    )
    parser.add_argument("-s", "--seed", default=0, type=int, help="rng seed")
    parser.add_argument(
        "-n", "--num-epochs", default=1000, type=int, help="number of training epochs"
    )
    parser.add_argument(
        "-d",
        "--dataset_path",
        default="/pool01/projects/abante_lab/cag_propagation/deep_dive_adata_v1_2k.h5",
        type=str,
        help="file for which dataset to use"
    )
    parser.add_argument(
        "-bs", "--batch-size", default=1000, type=int, help="mini-batch size"
    )
    parser.add_argument(
        "-lr", "--learning-rate", default=0.001, type=float, help="learning rate"
    )
    parser.add_argument(
        "--cuda", action="store_true", default=False, help="whether to use cuda"
    )
    parser.add_argument(
        "--latent_dim", type=int, default=20, help="Latent dimension for the model"
    )
    parser.add_argument(
        "--n_hidden_layers", type=int, default=1, help="Number of hidden layers in each encoder/decoder neural network"
    )
    parser.add_argument(
        "--plot", action="store_true", default=False, help="whether to make a plot"
    )
    parser.add_argument(
        "--save_latent", action="store_true", default=False, help="whether to save the latent encodings in adata"
    )
    parser.add_argument(
        "--unseen_y_prop", type=float, default=0.0, help="Proportion of masked labels"
    )
    parser.add_argument(
        "--multitech", action="store_true", default=False, help="Whether to use multitech training"
    )
    args = parser.parse_args()

    main(args)
