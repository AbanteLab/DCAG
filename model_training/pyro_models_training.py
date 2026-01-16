#!/usr/bin/env python3

### Deep Generative Modeling for Single Cell Data with Pyro ###
#%%
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.optim import Adam

import pyro
from pyro.optim import MultiStepLR
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate, Trace_ELBO

from models import scANVI, DCAG, CustomELBO, Baseline
from utils import *

#%% args for interactive testing
# args = argparse.Namespace(
#     seed=1,
#     num_epochs=5000,
#     dataset_path="/pool01/projects/abante_lab/cag_propagation/deep_dive_adata_v1_8k.h5",
#     # dataset_path="/pool01/projects/abante_lab/cag_propagation/deep_dive_10x_v1_6k_test0.5_adata.h5",
#     # dataset_path="/pool01/projects/abante_lab/cag_propagation/handsaker_deepdive_thompson2020.h5",
#     # dataset_path="/pool01/projects/abante_lab/cag_propagation/handsaker_deepdive10X_thompson2020.h5",
#     # dataset_path="/pool01/projects/abante_lab/cag_propagation/multitech_supervised_adata_v2_6000.h5",
#     batch_size=1000,     
#     learning_rate=1e-4,
#     cuda=True,
#     multitech = False,
#     unseen_y_prop=0.0,      # If multitech true, this arg is ignored
#     plot=True,
#     model="DCAG", 
#     lambda_reg=0.01,       # weight of orthogonality regularization term
#     alpha_y = 10.0,
#     alpha_ct = 0.1,
#     latent_dim_y = 60,
#     latent_dim_ct = 20,
#     reg_method = 'ortho',   # 'ortho' or 'hsic'
#     save_latent = False,
#     n_hidden_layers = 2
# )
#%%
def main(args):
    #%%
    superprint("Starting model training...")
    
    # Fix random number seed
    pyro.util.set_rng_seed(args.seed)

    # Load and pre-process data
    if args.multitech:
        dataloader, val_dataloader, test_dataloader, num_genes, l_mean, l_scale, adata = get_data_multitech(
            dataset_path=args.dataset_path, batch_size=args.batch_size, cuda=args.cuda, test_size=0.4, val_size=0.1, seed=args.seed
        )
        # Compute alpha_y based on unlabeled data proportion
        unlabeled_prop = adata.obs['CAGPHASE'].isna().sum() / adata.n_obs
        args.alpha_y = float(alpha_scal(unlabeled_prop))

    else:
        if args.unseen_y_prop == 0.0:
            # Supervised learning
            dataloader, val_dataloader, test_dataloader, num_genes, l_mean, l_scale, adata = get_data_supervised(
                dataset_path=args.dataset_path, batch_size=args.batch_size, cuda=args.cuda, test_size=0.4, val_size=0.1, seed=args.seed
            )
        else:
            # Masked learning
            dataloader, val_dataloader, test_dataloader, num_genes, l_mean, l_scale, adata = get_data_masked(
                dataset_path=args.dataset_path, batch_size=args.batch_size, cuda=args.cuda, test_size=0.4, val_size=0.1, masking_prop = args.unseen_y_prop, seed=args.seed
            )

    # Instantiate instance of model/guide and various neural networks
    if args.model == "scANVI":
        model = scANVI(
            num_genes=num_genes,
            num_cag_cat=dataloader.Y.shape[1],
            l_loc=l_mean,
            l_scale=l_scale,
            latent_dim=args.latent_dim_y,         # NOTE: by default, latent_dim is set to 20 in the class
            alpha=args.alpha_y,                   # NOTE: by default, alpha is set to 0.01
            scale_factor=1.0 / (args.batch_size * num_genes),
            n_hidden_layers=args.n_hidden_layers
        )
    elif args.model == "DCAG":
        model = DCAG(
            num_genes=num_genes,
            num_cag_cat=dataloader.Y.shape[1],
            num_ct_cat=len(dataloader.CT.unique()),
            l_loc=l_mean,
            l_scale=l_scale,
            latent_dim_y=args.latent_dim_y,
            latent_dim_ct=args.latent_dim_ct,
            alpha_y=args.alpha_y,
            alpha_ct=args.alpha_ct,
            scale_factor=1.0 / (args.batch_size * num_genes),
            n_hidden_layers=args.n_hidden_layers
        )
    elif args.model == "Baseline":
        model = Baseline(
            num_genes=num_genes,
            num_cag_cat=dataloader.Y.shape[1],
            l_loc=l_mean,
            l_scale=l_scale,
            latent_dim=args.latent_dim_y,
            alpha=args.alpha_y,
            scale_factor=1.0 / (args.batch_size * num_genes),
            n_hidden_layers=args.n_hidden_layers
        )

    if args.cuda:
        model.cuda()

    # Setup an optimizer (Adam) and learning rate scheduler.
    scheduler = MultiStepLR(
        {
            "optimizer": Adam,
            "optim_args": {"lr": args.learning_rate},
            "milestones": [20,40,60,80,100,150,200,300],
            "gamma": 0.95,
        }
    )

    # Tell Pyro to enumerate out y when y is unobserved
    guide = config_enumerate(model.guide, "parallel", expand=True)

    # Setup a variational objective for gradient-based learning.
    if model.__class__.__name__ == "scANVI" or model.__class__.__name__ =='Baseline':
        elbo = TraceEnum_ELBO(strict_enumeration_warning=False)
    elif model.__class__.__name__ == "DCAG":
        elbo = CustomELBO(args.lambda_reg, args.reg_method)
    svi = SVI(model.model, guide, scheduler, elbo)

    loss_history = []
    hsic_history = []
    test_history = []
    test_hsic_history = []

    # Early stopping parameters
    patience = 10   # how many epochs to wait after no improvement
    min_delta = 0.005
    epochs_no_improve = 0
    best_val_loss = float("inf")
    best_state = None  # to save the best model

    # Training loop
    for epoch in range(args.num_epochs):
        losses = []
        hsics = []

        for batch in dataloader:
            x, y, ct = batch[:3]
            if y is not None:
                y = y.type_as(x)
            if model.__class__.__name__ == "scANVI" or model.__class__.__name__ == "Baseline":
                loss = svi.step(x, y)
            elif model.__class__.__name__ == "DCAG":
                loss = svi.step(x, y, ct)
                hsics.append(elbo.hsic_val)
            losses.append(loss)

        # Tell the scheduler we've done one epoch.
        scheduler.step()

        mean_train_loss = np.mean(losses)
        mean_train_hsic = np.mean(hsics) if hsic_history else 0
        print("[Epoch %04d]  Loss: %.5f     HSIC:%.5f" % (epoch, mean_train_loss, mean_train_hsic))
        loss_history.append([epoch, mean_train_loss])
        hsic_history.append([epoch, mean_train_hsic])

        # Validation
        if epoch % 10 == 0:
            test_losses = []
            test_hsics = []
            for batch in val_dataloader:
                x_val, y_val, ct_val = batch[:3]
                if y_val is not None:
                    y_val = y_val.type_as(x_val)
                    ct_val = ct_val.type_as(x_val)
                if model.__class__.__name__ == "scANVI" or model.__class__.__name__ == "Baseline":
                    val_loss = svi.evaluate_loss(x_val, y_val)
                elif model.__class__.__name__ == "DCAG":
                    val_loss = svi.evaluate_loss(x_val, y_val, ct_val)
                    test_hsics.append(elbo.hsic_val)
                test_losses.append(val_loss)

            mean_val_loss = np.mean(test_losses)
            mean_val_hsic = np.mean(test_hsics) if test_hsics else 0
            print("[Epoch %04d]  Val Loss: %.5f     Val HSIC: %.5f" % (epoch, mean_val_loss, mean_val_hsic))
            test_history.append([epoch, mean_val_loss])
            test_hsic_history.append([epoch, mean_val_hsic])

            # Early stopping check
            if mean_val_loss + min_delta < best_val_loss:
                best_val_loss = mean_val_loss
                epochs_no_improve = 0
                # save best model state
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
                # Restore best model before returning
                if best_state is not None:
                    model.load_state_dict(best_state)
    
    superprint("Training complete! Evaluating model...")
    # Put neural networks in eval mode (needed for batchnorm)
    model.eval()

    output_path = "/pool01/projects/abante_lab/cag_propagation/scanvi_results/"
    # output_path = "/gpfs/projects/ub212/cag_propagation/data/model_results/"

    model_sufix, dataset_name = get_model_sufix(model, args)

    if args.plot:
        if model.__class__.__name__ == "scANVI":
            plot_scanvi_pca_latent(model, dataloader, latent="z2", output_path=f"{output_path}{model_sufix}_pca_z2_trainset.pdf", show_plot=True)
            plot_scanvi_pca_latent(model, dataloader, latent="z1", output_path=f"{output_path}{model_sufix}_pca_z1_trainset.pdf")
            plot_scanvi_pca_latent(model, test_dataloader, latent="z2", output_path=f"{output_path}{model_sufix}_pca_z2_testset.pdf")
            plot_scanvi_pca_latent(model, test_dataloader, latent="z1", output_path=f"{output_path}{model_sufix}_pca_z1_testset.pdf")
            
        elif model.__class__.__name__ == "DCAG":
            plot_scanvi_pca_latent(model, dataloader, latent="z_y", output_path=f"{output_path}{model_sufix}_pca_zy_trainset.pdf", show_plot=True)
            plot_scanvi_pca_latent(model, dataloader, latent="z_ct", output_path=f"{output_path}{model_sufix}_pca_zct_trainset.pdf", show_plot=True)
            plot_scanvi_pca_latent(model, test_dataloader, latent="z_y", output_path=f"{output_path}{model_sufix}_pca_zy_testset.pdf", show_plot=True)
            plot_scanvi_pca_latent(model, test_dataloader, latent="z_ct", output_path=f"{output_path}{model_sufix}_pca_zct_testset.pdf", show_plot=True)

        elif model.__class__.__name__ == "Baseline":
            plot_scanvi_pca_latent(model, dataloader, latent="z2", output_path=f"{output_path}{model_sufix}_pca_z2_trainset.pdf", show_plot=True)
            plot_scanvi_pca_latent(model, test_dataloader, latent="z2", output_path=f"{output_path}{model_sufix}_pca_z2_testset.pdf")
        
        plot_losses(loss_history, test_history, output_path=f"{output_path}{model_sufix}_loss.pdf")
    
    evaluate_model(model, dataloader, test_dataloader, args, dataset_name, model_sufix, output_path=output_path)

    # Save latent representation in adata
    if args.save_latent == True:
        adata = encode_all(model, args)
        adata.write_h5ad(f"{output_path}adata_with_latent_{model_sufix}.h5ad")

#%%
# Plot probabilistic model
    # data_iter = iter(dataloader)  
    # batch = next(data_iter)
    # x, y, ct = batch
    # dot = pyro.render_model(model.model, model_args=(x, y, ct), render_distributions=True)
    # dot.render(filename=f"{output_path}probgraph__{args.model.__name__}", format="png") 

#%%
if __name__ == "__main__":

    assert pyro.__version__.startswith("1.9.1")
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="single-cell ANnotation using Variational Inference"
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
        "--alpha_y", type=float, default=0.1, help="Alpha parameter for the model"
    )
    parser.add_argument(
        "--alpha_ct", type=float, default=0.1, help="Alpha parameter for the model"
    )
    parser.add_argument(
        "--lambda_reg", type=float, default=1e6, help="Weight of orthogonality regularization term. Only used if model is DCAG"
    )
    parser.add_argument(
        "--latent_dim_y", type=int, default=20, help="Latent dimension for the model Y Z"
    )
    parser.add_argument(
        "--latent_dim_ct", type=int, default=20, help="Latent dimension for the model CT Z"
    )
    parser.add_argument(
        "--n_hidden_layers", type=int, default=1, help="Number of hidden layers in each encoder/decoder neural network"
    )
    parser.add_argument(
        "--reg_method", type=str, default="ortho", help="Regularization method: 'ortho' or 'hsic'. Only used if model is DCAG"
    )
    parser.add_argument(
        "--plot", action="store_true", default=False, help="whether to make a plot"
    )
    parser.add_argument(
        "--save_latent", action="store_true", default=False, help="whether to save the latent encodings in adata"
    )
    parser.add_argument(
        "--model", type=str, default="scANVI", help="Model class to train"
    )
    parser.add_argument(
        "--unseen_y_prop", type=float, default=0.0, help="Proportion of masked labels"
    )
    parser.add_argument(
        "--multitech", action="store_true", default=False, help="Whether to use multitech training"
    )
    args = parser.parse_args()

    main(args)