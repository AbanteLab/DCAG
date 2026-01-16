#!/usr/bin/env python3

### Function definitions for Single Cell Data with Pyro and scANVI###
#%%
import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
import anndata as ad
import h5py
import scanpy as sc
import scipy.sparse as sp
import numpy as np
import torch.nn.functional as F
import math
from datetime import datetime
from collections import defaultdict
import pyro
from pyro.infer import config_enumerate

import seaborn as sns
import matplotlib
from matplotlib.patches import Patch

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.nn.functional import softmax
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, f1_score, silhouette_score, accuracy_score
from sklearn.decomposition import PCA

#%%

def superprint(message):
    """Prints a message with a timestamp."""
    
    # Get the current date and time
    now = datetime.now()
    
    # Format the date and time
    timestamp = now.strftime("[%Y-%m-%d %H:%M:%S]")
    
    # Print the message with the timestamp
    print(f"{timestamp} {message}")

class BatchDataLoader:
    """
    Dataloader for supervised learning (all samples in all batches are labeled)
    """
    def __init__(self, X, Y, CT, batch_size, shuffle=True):
        self.X = X
        self.Y = Y
        self.CT = CT
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = X.size(0)

    def __iter__(self):
        indices = torch.randperm(self.num_samples) if self.shuffle else torch.arange(self.num_samples)
        for start_idx in range(0, self.num_samples, self.batch_size):
            end_idx = start_idx + self.batch_size
            batch_idx = indices[start_idx:end_idx]
            yield self.X[batch_idx], self.Y[batch_idx], self.CT[batch_idx]

    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size
    
class SemiLabeled_BatchDataLoader:
    """
    Dataloader for semi-supervosed learning combining labeled and unlabeled batches.
    """
    def __init__(self, labeled_vars, unlabeled_vars, batch_size, shuffle=True):

        self.labeled_X, self.labeled_Y, self.labeled_CT, _, self.labeled_TECH = labeled_vars
        self.unlabeled_X, self.unlabeled_Y, self.unlabeled_CT, _, self.unlabeled_TECH = unlabeled_vars

        # Combine labeled and unlabeled training data    
        self.X = torch.cat([self.labeled_X, self.unlabeled_X], dim=0)
        self.Y = torch.cat([self.labeled_Y, self.unlabeled_Y], dim=0)
        self.CT = torch.cat([self.labeled_CT, self.unlabeled_CT], dim=0)
        self.TECH = torch.cat([self.labeled_TECH, self.unlabeled_TECH], dim=0)

        self.batch_size = batch_size
        self.shuffle = shuffle

        self.num_labeled_samples = self.labeled_X.size(0)
        self.num_unlabeled_samples = self.unlabeled_X.size(0)

        self.num_labeled_batches = int(np.ceil(self.num_labeled_samples / batch_size))
        self.num_unlabeled_batches = int(np.ceil(self.num_unlabeled_samples / batch_size))

        # cache one-hot dimensionality for correct unlabeled mask
        self.num_classes = self.labeled_Y.size(1) if self.labeled_Y.ndim > 1 else 1

    def __iter__(self):
        # create shuffled or sequential indices
        labeled_indices = torch.randperm(self.num_labeled_samples) if self.shuffle else torch.arange(self.num_labeled_samples)
        unlabeled_indices = torch.randperm(self.num_unlabeled_samples) if self.shuffle else torch.arange(self.num_unlabeled_samples)

        # split into batches
        labeled_batches = [
            labeled_indices[i * self.batch_size:(i + 1) * self.batch_size]
            for i in range(self.num_labeled_batches)
        ]
        unlabeled_batches = [
            unlabeled_indices[i * self.batch_size:(i + 1) * self.batch_size]
            for i in range(self.num_unlabeled_batches)
        ]

        # randomly interleave labeled and unlabeled batches
        mixed_tags = ['L'] * len(labeled_batches) + ['U'] * len(unlabeled_batches)
        np.random.shuffle(mixed_tags)

        for tag in mixed_tags:
            if tag == 'L' and labeled_batches:
                idx = labeled_batches.pop()
                yield (
                    self.labeled_X[idx],
                    self.labeled_Y[idx],
                    self.labeled_CT[idx],
                )
            elif tag == 'U' and unlabeled_batches:
                idx = unlabeled_batches.pop()
                batch_size = idx.size(0)
                # one-hot shape: (batch_size, num_classes)
                unlabeled_Y_mask = torch.full((batch_size, self.num_classes), -1.0, device=self.unlabeled_Y.device)
                yield (
                    self.unlabeled_X[idx],
                    unlabeled_Y_mask,
                    self.unlabeled_CT[idx],
                )

    def __len__(self):
        return self.num_labeled_batches + self.num_unlabeled_batches
    
def load_dataset_separate_variables(dataset_path, cuda = False):
    """
    Read dataset and return X and metadata in separate objects.
    """

    # Load the dataset
    if hasattr(ad, "io"):
        adata = ad.io.read_h5ad(dataset_path)
    else:
        adata = ad.read_h5ad(dataset_path)

    # Keep only rows where CAGPHASE is not 'D'
    adata = adata[adata.obs['CAGPHASE'] != 'D'].copy()

    # Convert phases into ints
    int_labels = adata.obs['CAGPHASE'].astype('category').cat.codes.to_numpy()
    Y = torch.from_numpy(int_labels).long()

    # If Y has non-labeled indices (labeled as -1):
    if -1 not in Y.unique().tolist():
        Y = F.one_hot(Y, num_classes=len(np.unique(int_labels))).float()
    else:
        # Count classes as non-negative labels only
        num_classes = len(torch.unique(Y[Y >= 0]))

        # Initialize all rows with -1
        Y_onehot = -1 * torch.ones((Y.size(0), num_classes), dtype=torch.float)

        # Mask valid indices (non -1)
        mask = Y >= 0
        Y_onehot[mask] = F.one_hot(Y[mask], num_classes=num_classes).float()
        Y = Y_onehot

    # Convert celltypes into ints
    CT = adata.obs['CELLTYPE'].astype('category').cat.codes.to_numpy()
    CT = torch.from_numpy(CT).long()
    # Convert batch into ints
    SB = adata.obs['BATCH'].astype('category').cat.codes.to_numpy()
    SB = torch.from_numpy(SB).long()

    # Check if the TECHNOLOGY column exists; if not, create it with default value
    if 'TECHNOLOGY' not in adata.obs.columns:
        adata.obs['TECHNOLOGY'] = 'Unknown'
        
    # Convert technology into ints
    TECH = adata.obs['TECHNOLOGY'].astype('category').cat.codes.to_numpy()
    TECH = torch.from_numpy(TECH).long()
    # 10X is assigned 0 and DeepDive is assigned 1
    # TODO: save this mapping equivalence in a file to access it in results evaluation

    # Convert count matrix to tensor
    X = torch.from_numpy(sp.csr_matrix.todense(adata.X)).float()

    # Prior mean and std for log count latent variable `l`
    log_counts = X.sum(-1).log()
    l_mean, l_scale = log_counts.mean().item(), log_counts.std().item()

    if cuda:
        X, Y, CT, SB, TECH = X.cuda(), Y.cuda(), CT.cuda(), SB.cuda(), TECH.cuda()

    return X, Y, CT, SB, TECH, l_mean, l_scale, adata

def split_and_subset(indices, test_size, vars_list, seed=0, return_indices = False):
    """
    Splits indices into train/test stratifying on Y accordingly.
    To stratify on Y, this must be the second element of vars_list.
    """

    Y = vars_list[1]

    # Split stratifying on Y
    train_idx_np, test_idx_np = train_test_split(
        indices, test_size=test_size, stratify=Y.cpu(), shuffle=True, random_state=seed
    )

    train_idx = torch.tensor(train_idx_np, device=Y.device)
    test_idx = torch.tensor(test_idx_np, device=Y.device)

    X, Y, CT, SB, TECH = vars_list

    X_train = X[train_idx]
    Y_train = Y[train_idx]
    CT_train = CT[train_idx]
    SB_train = SB[train_idx]
    TECH_train = TECH[train_idx]
    train_vars = (X_train, Y_train, CT_train, SB_train, TECH_train)

    X_test = X[test_idx]
    Y_test = Y[test_idx]
    CT_test = CT[test_idx]
    SB_test = SB[test_idx]
    TECH_test = TECH[test_idx]
    test_vars = (X_test, Y_test, CT_test, SB_test, TECH_test)

    if return_indices:
        return train_vars, test_vars, train_idx_np, test_idx_np
    else:
        return train_vars, test_vars

def get_data_supervised(dataset_path, batch_size, cuda=False, test_size=0.2, val_size = 0.1, seed=0):
    """
    Load dataset and create train, test and validation dataloaders for supervised learning.
    """
    
    # Load the dataset
    X, Y, CT, SB, TECH, l_mean, l_scale, adata = load_dataset_separate_variables(dataset_path, cuda)

    torch.manual_seed(seed)

    idx = np.array([i for i in range(X.size(0))])

    # Split labeled indices into train/test
    trainval_vars, test_vars = split_and_subset(idx, test_size, [X, Y, CT, SB, TECH], seed)

    # Get val ratio
    val_ratio = val_size / (1-test_size)

    # Get indices for trainval set
    trainval_idx_np = np.array([i for i in range(trainval_vars[0].size(0))])

    # Split tranval indices into train/val
    train_vars, val_vars = split_and_subset(trainval_idx_np, val_ratio, trainval_vars, seed)

    X_train, Y_train, CT_train, _, _ = train_vars
    X_test, Y_test, CT_test, _, _ = test_vars
    X_val, Y_val, CT_val, _, _= val_vars

    return (
        BatchDataLoader(X_train, Y_train, CT_train, batch_size),
        BatchDataLoader(X_val, Y_val, CT_val, batch_size),
        BatchDataLoader(X_test, Y_test, CT_test, batch_size),
        X_train.size(-1),
        l_mean,
        l_scale, 
        adata
    )

def get_data_masked(dataset_path, batch_size, cuda=False, test_size=0.2, val_size = 0.1, masking_prop = 0.2, seed=0):
    """
    Load dataset and create train, test and validation dataloaders for masked learning.
    """
    
    # Load the dataset
    X, Y, CT, SB, TECH, l_mean, l_scale, adata = load_dataset_separate_variables(dataset_path, cuda)

    torch.manual_seed(seed)

    idx = np.array([i for i in range(X.size(0))])

    # Split indices into labeled and unlabeled
    labeled_vars, unlabeled_vars, labeled_idx_np, unlabeled_idx_np = split_and_subset(idx, masking_prop, [X, Y, CT, SB, TECH], seed, return_indices = True)

    # Save in anndata object whether the row was labeled or unlabeled in the masked learning
    adata.obs['MASKING_STAT'] = 'labeled'
    col = adata.obs.columns.get_loc('MASKING_STAT')
    adata.obs.iloc[unlabeled_idx_np.tolist(), col] = 'unlabeled'

    # Get indices for labeled set
    labeled_idx = np.array([i for i in range(labeled_vars[0].size(0))])

    # Split labeled indices into train/test
    trainval_vars, test_vars = split_and_subset(labeled_idx, test_size, labeled_vars, seed)
    
    X_test, Y_test, CT_test, _, _ = test_vars

    val_ratio = val_size / (1-test_size)

    # Get indices for trainval set
    trainval_idx_np = np.array([i for i in range(trainval_vars[0].size(0))])
    
    # Split labeled indices into train/test
    train_vars, val_vars = split_and_subset(trainval_idx_np, val_ratio, trainval_vars, seed)
    
    X_val, Y_val, CT_val, _, _= val_vars

    return (
        SemiLabeled_BatchDataLoader(train_vars, unlabeled_vars, batch_size),
        BatchDataLoader(X_val, Y_val, CT_val, batch_size),
        BatchDataLoader(X_test, Y_test, CT_test, batch_size),
        X_test.size(-1),
        l_mean,
        l_scale, 
        adata
    )

def get_data_multitech(dataset_path, batch_size, cuda=False, test_size=0.2, val_size = 0.1, seed=0):
    """
    Load dataset and create train, test and validation dataloaders for semi-supervised learning.
    """
    
    # Load the dataset
    X, Y, CT, SB, TECH, l_mean, l_scale, adata = load_dataset_separate_variables(dataset_path, cuda)

    torch.manual_seed(seed)

    idx = np.array([i for i in range(X.size(0))])

    # Get labeled indices (DeepDive)
    labeled_idx = np.where(TECH.cpu().numpy() == 1)[0]
    X_labeled = X[labeled_idx]
    Y_labeled = Y[labeled_idx]
    CT_labeled = CT[labeled_idx]
    SB_labeled = SB[labeled_idx]
    TECH_labeled = TECH[labeled_idx]

    labeled_idx = np.array([i for i in range(X_labeled.size(0))])

    # Split labeled indices into train/test
    trainval_vars, test_vars = split_and_subset(labeled_idx, test_size, [X_labeled, Y_labeled, CT_labeled, SB_labeled, TECH_labeled], seed)

    # Get val ratio
    val_ratio = val_size / (1-test_size)

    # Get indices for trainval set
    trainval_idx_np = np.array([i for i in range(trainval_vars[0].size(0))])
    
    # Split labeled indices into train/test
    labeled_train_vars, val_vars = split_and_subset(trainval_idx_np, val_ratio, trainval_vars, seed)

    X_test, Y_test, CT_test, _, _ = test_vars
    X_val, Y_val, CT_val, _, _= val_vars

    # Get unlabeled indices (10X)
    unlabeled_idx = np.where(TECH.cpu().numpy() == 0)[0]

    X_train_unlabeled = X[unlabeled_idx]
    Y_train_unlabeled = Y[unlabeled_idx]
    CT_train_unlabeled = CT[unlabeled_idx]
    SB_train_unlabeled = SB[unlabeled_idx]
    TECH_train_unlabeled = TECH[unlabeled_idx]
    unlabeled_train_vars = (X_train_unlabeled, Y_train_unlabeled, CT_train_unlabeled, SB_train_unlabeled, TECH_train_unlabeled)

    return (
        SemiLabeled_BatchDataLoader(labeled_train_vars, unlabeled_train_vars, batch_size),
        BatchDataLoader(X_val, Y_val, CT_val, batch_size),
        BatchDataLoader(X_test, Y_test, CT_test, batch_size),
        X_test.size(-1),
        l_mean,
        l_scale, 
        adata
    )

def split_anndata(adata, test_size=0.2, seed=0):
    """
    Train/test split for AnnData, stratifying only labeled cells. 
    All unlabeled cells are kept in training set.
    Test size is with respect the totality of labeled cells.
    """

    # Fill NaN celltypes with 'unknown' 
    if isinstance(adata.obs["CELLTYPE"].dtype, pd.CategoricalDtype):
        adata.obs["CELLTYPE"] = adata.obs["CELLTYPE"].cat.add_categories(["unknown"])
    adata.obs['CELLTYPE'] = adata.obs['CELLTYPE'].fillna('unknown')

    labels = adata.obs["CAGPHASE"].values
    indices = np.arange(adata.n_obs)

    # Split labeled cells with stratification
    labeled_mask = ~pd.isna(labels)
    labeled_indices = indices[labeled_mask]
    labeled_labels = labels[labeled_mask]

    train_lab, test_lab = train_test_split(
        labeled_indices,
        test_size=test_size,
        stratify=labeled_labels,
        shuffle=True,
        random_state=seed
    )

    # Get unlabeled indices
    unlabeled_indices = indices[~labeled_mask]

    # Combine
    train_idx = np.concatenate([train_lab, unlabeled_indices])

    adata_train = adata[train_idx].copy()
    adata_test = adata[test_lab].copy()

    return adata_train, adata_test

def split_anndata_with_masking(adata, test_size=0.2, seed=0):
    """
    Train/test split using only labeled samples, then add unlabeled samples to training.
    Creates .obs['CAGPHASE_MASKING'] with original labels for labeled samples 
    and 'unknown' for unlabeled samples.
    """
    # Identify labeled and unlabeled samples
    labeled_mask = adata.obs["MASKING_STAT"] == "labeled"
    unlabeled_mask = ~labeled_mask

    labeled_indices = np.where(labeled_mask)[0]
    unlabeled_indices = np.where(unlabeled_mask)[0]

    # Stratify using only labeled samples
    labels = adata.obs.loc[labeled_mask, "CAGPHASE"].values

    train_labeled_idx, test_idx = train_test_split(
        labeled_indices,
        test_size=test_size,
        stratify=labels,
        shuffle=True,
        random_state=seed
    )

    # Add unlabeled samples to the training set
    train_idx = np.concatenate([train_labeled_idx, unlabeled_indices])

    # Slice the AnnData object
    adata_train = adata[train_idx].copy()
    adata_test = adata[test_idx].copy()

    # Create new combined label column (add unknown as a category of this column first)
    if isinstance(adata_train.obs["CAGPHASE"].dtype, pd.CategoricalDtype):
        adata_train.obs["CAGPHASE"] = adata_train.obs["CAGPHASE"].cat.add_categories(["unknown"])

    adata_train.obs["CAGPHASE_MASKING"] = adata_train.obs["CAGPHASE"].where(
        adata_train.obs["MASKING_STAT"] == "labeled",
        other="unknown"
    )
    adata_test.obs["CAGPHASE_MASKING"] = adata_test.obs["CAGPHASE"]  # test contains only labeled samples

    return adata_train, adata_test

def split_anndata_multitech(adata, batch_key, test_size=0.2, seed=0):
    """
    Train/test split using only labeled samples, then add unlabeled samples to training.
    """

    # Remove rows with null values in batch key
    adata = adata[~adata.obs[batch_key].isna()].copy()

    # Identify labeled and unlabeled samples
    labeled_mask = adata.obs["TECHNOLOGY"] == "DeepDive"
    unlabeled_mask = ~labeled_mask

    labeled_indices = np.where(labeled_mask)[0]
    unlabeled_indices = np.where(unlabeled_mask)[0]

    # Stratify using only labeled samples
    labels = adata.obs.loc[labeled_mask, "CAGPHASE"].values

    train_labeled_idx, test_idx = train_test_split(
        labeled_indices,
        test_size=test_size,
        stratify=labels,
        shuffle=True,
        random_state=seed
    )

    # Add unlabeled samples to the training set
    train_idx = np.concatenate([train_labeled_idx, unlabeled_indices])

    # Slice the AnnData object
    adata_train = adata[train_idx].copy()
    adata_test = adata[test_idx].copy()

    # Change NaN cagphase to 'unknown' in training set
    if isinstance(adata_train.obs["CAGPHASE"].dtype, pd.CategoricalDtype):
        adata_train.obs["CAGPHASE"] = adata_train.obs["CAGPHASE"].cat.add_categories(["unknown"])

    adata_train.obs["CAGPHASE"] = adata_train.obs["CAGPHASE"].fillna("unknown")

    return adata_train, adata_test

def extract_z1(scanvi, X, Y):
    """
    Encode samples to obtain z1 latent representation using trained model.
    """

    with torch.no_grad():
        # Encode z2
        z2_loc = scanvi.z2l_encoder(X)[0]  # (N, z2_dim)

        # Encode z1 with z2 and mixed y
        z1_loc, _ = scanvi.z1_encoder(z2_loc, Y)
        z1_latents = z1_loc.cpu().numpy()

    return z1_latents

def plot_scanvi_pca_latent(model, dataloader, latent="z2", output_path="scanvi_pca_latent.pdf", show_plot=False, multitech=False):
    """
    Plots a 2D PCA of the specified latent space (z1 or z2), colored by CAGPHASE and CELLTYPE.

    Handles unlabeled samples in CAGPHASE by assigning them to 'unlabeled'.

    Parameters:
        model: Trained model
        dataloader: Data loader with X, Y and CT attribute. If model SCANVI, pass anndata object.
        latent: "z1" or "z2" (default: "z2") if model is scANVI; "z_y" or "z_ct" if model is DCAG
        output_path: File path to save the plot
        show_plot: Whether to display the plot interactively
    """
    
    if model.__class__.__name__ == "SCANVI":
        X = dataloader.X
        Y = dataloader.obs["CAGPHASE"]
        CT = dataloader.obs["CELLTYPE"]
        if multitech:
            CT_pred = dataloader.obs['C_scANVI']

        latent_vec = dataloader.obsm['X_scANVI']

    else:
        X = dataloader.X
        Y = dataloader.Y
        CT = dataloader.CT

        if model.__class__.__name__ == "scANVI":
            # Compute latent representation
            if latent == "z2":
                latent_vec = model.z2l_encoder(X)[0].detach().cpu().numpy()
            elif latent == "z1":
                latent_vec = extract_z1(model, X, Y)
            else:
                raise ValueError("latent must be 'z1' or 'z2'")
        elif model.__class__.__name__ == "DCAG":
            if latent == "z_ct":
                latent_vec = model.zctl_encoder(X)[0].detach().cpu().numpy()
            elif latent == "z_y":
                latent_vec = model.zyl_encoder(X)[0].detach().cpu().numpy()
            else:
                raise ValueError("latent must be 'z_y' or 'z_ct'")
        elif model.__class__.__name__ == "Baseline":
            latent_vec = model.z2l_encoder(X)[0].detach().cpu().numpy()

    pca = PCA(n_components=2)
    latent_pca = pca.fit_transform(latent_vec)

    features_to_plot = ["Y", "CT"]
    if multitech:
        features_to_plot = ['CT', 'CT_pred', 'ORIGIN']
    n_features = len(features_to_plot)
    fig, axes = plt.subplots(1, n_features, figsize=(6 * n_features, 5))

    if n_features == 1:
        axes = [axes]

    for i, feature in enumerate(features_to_plot):
        ax = axes[i]

        if feature == "Y":
            if model.__class__.__name__ == "SCANVI":
                color = Y
            else:
                # Revert one hot encoding
                if Y.ndim > 1 and Y.shape[1] > 1:
                    mask_all_invalid = (Y == -1).all(dim=-1)
                    Y = torch.argmax(Y, dim=-1)
                    Y[mask_all_invalid] = -1
                color = pd.Series(Y.cpu().numpy())
                label_dict = {-1: 'No label', 0: 'A', 1: 'B', 2: 'C', 3:'D'}
            
        elif feature == "CT" or feature == "CT_pred":
            if model.__class__.__name__ == "SCANVI":
                color = CT 
                if multitech:
                    color = CT_pred
            else:
                color = pd.Series(CT.cpu().numpy())
                label_dict = {-1: 'No label', 0: 'SPN', 1: 'astrocyte', 2: 'endothelia', 3: 'interneuron', 4: 'microglia', 5: 'oligodendrocyte', 6: 'polydendrocyte'}         

        elif feature == "ORIGIN":
            color = dataloader.obs['ORIGIN']

        # Convert to categorical for consistent coloring
        color = color.astype(str)  # ensure string dtype
        unique_vals = sorted(color.dropna().unique())  
        palette = sns.color_palette("tab10", n_colors=len(unique_vals))
        color_map = dict(zip(unique_vals, palette))
        color_vals = color.map(color_map).fillna("lightgray")

        # Build legend with label_dict applied
        handles = []
        for val, col in color_map.items():
            if model.__class__.__name__ == "SCANVI":
                label = val
            else:
                int_val = int(val)  # convert back to int key
                label = label_dict.get(int_val, str(val))  # fallback to original if missing
            handles.append(Patch(color=col, label=label))

        # scatter
        scatt = ax.scatter(
            latent_pca[:, 0], latent_pca[:, 1], c=color_vals, s=5, alpha=0.7, rasterized=True
        )
        ax.legend(handles=handles, loc="upper right", fontsize="x-small", title=feature)

        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.set_title(f"{latent} PCA colored by {feature}")

    fig.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close(fig)

def plot_losses(loss_history, test_history, output_path):
    # Unpack (epoch, loss) pairs
    train_epochs, train_losses = zip(*loss_history)
    val_epochs, val_losses = zip(*test_history)

    plt.figure()
    plt.plot(train_epochs, train_losses, label='Train')
    plt.plot(val_epochs, val_losses, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_losses_scanvi(model, output_path, show_plot=False):
    hist = model.history_

    # Extract dataframes
    train_df = hist['train_loss_epoch']
    val_df = hist['validation_loss']

    # Extract epoch index and loss values
    train_epochs = train_df.index.values
    train_values = train_df.iloc[:, 0].values

    val_epochs = val_df.index.values
    val_values = val_df.iloc[:, 0].values

    plt.figure(figsize=(7, 5))
    plt.plot(train_epochs, train_values, label="Train Loss", linewidth=2)
    plt.plot(val_epochs, val_values, label="Validation Loss", linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close()

def plot_confusion(y_true, y_pred, title):
    """
    Plots a confusion matrix and computes balanced accuracy to display alongside.
    """

    accuracy = balanced_accuracy_score(y_true, y_pred)
    print(f"{title}: Balanced Accuracy = {accuracy:.4f}")
    cm = confusion_matrix(y_true, y_pred)
    labels = np.arange(cm.shape[0])
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    cm_color = np.divide(cm_norm, row_sums, where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_color, interpolation='nearest', cmap=plt.cm.Blues)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm_color[i, j] > 0.5 else "black")

    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel='True label',
        xlabel='Predicted label',
        title=title
    )
    plt.colorbar(im, ax=ax, label="Normalized by row (true class)")
    plt.tight_layout()
    plt.show()
    return accuracy

def performance_metrics_scanvi(adata):
    """
    Plots a confusion matrix of scANVI models and computes balanced accuracy to display alongside. 
    """
    y_true = adata.obs["CAGPHASE"]
    y_pred = adata.obs["C_scANVI"]

    # Ensure both are categorical and aligned
    y_true = y_true.astype("category")
    y_pred = y_pred.astype("category").cat.set_categories(y_true.cat.categories)

    # Convert to integer-coded labels for confusion matrix
    y_true_int = y_true.cat.codes.values
    y_pred_int = y_pred.cat.codes.values

    # Plot confusion matrix
    accuracy = plot_confusion(
        y_true=y_true_int,
        y_pred=y_pred_int,
        title="SCANVI Confusion Matrix"
    )
    return accuracy

def masked_accuracy(model, dataloader, args):
    """
    Computes accuracy of masked cells during training.
    """

    if model.__class__.__name__ == "SCANVI":
        # Get masked indices
        unlabeled_mask = dataloader.obs["MASKING_STAT"] == "unlabeled"
        y_pred = dataloader.obs['C_scANVI'][unlabeled_mask] 
        y_masked_true_all = dataloader.obs['CAGPHASE'][unlabeled_mask]
    else:
        # Check if dataloader has masked labels
        if not hasattr(dataloader,'unlabeled_Y'):
            raise ValueError("Data loader doesn't have masked cells")
        
        # Compute latent representation
        if model.__class__.__name__ == "DCAG":
            latent_rep = model.zyl_encoder(dataloader.unlabeled_X)[0]
        elif model.__class__.__name__ == "scANVI":
            latent_rep = model.z2l_encoder(dataloader.unlabeled_X)[0]

        # Get predicted probabilities 
        y_logits = model.y_classifier(latent_rep)
        y_probs = softmax(y_logits, dim=-1).data.cpu().numpy()
        y_pred = np.argsort(y_probs, axis=1)[:, -1]
        
        y_masked_true_all = dataloader.unlabeled_Y.cpu().numpy()

        if y_masked_true_all.ndim > 1:
            # In case it is one-hot encoded
            y_masked_true_all = np.argmax(y_masked_true_all, axis=1)

    # Compute accuracy
    accuracy = plot_confusion(y_masked_true_all, y_pred, f"Confusion Matrix ({args.unseen_y_prop * 100}% cells masked)")
    print(f"Balanced Accuracy ((Masking {args.unseen_y_prop * 100}% cells)): {accuracy:.4f}")

    return accuracy

def performance_metrics(model, dataloader, train_test = 'train'):
    """
    Computes and prints performance metrics (balanced accuracy, F1 score) for the scANVI model.
    Parameters:
        model: Trained scANVI model
        dataloader: BatchDataLoader or similar with X and Y attributes. In SCANVI model, pass anndata object.
        train_test: 'train' or 'test' to indicate which dataset is being evaluated
    Returns:
        accuracy: Balanced accuracy score
        f1: F1 score (macro)
    """
    if model.__class__.__name__ == 'SCANVI':
        y_true = dataloader.obs["CAGPHASE"]
        y_pred = dataloader.obs["C_scANVI"]

        # Ensure both are categorical and aligned
        y_true = y_true.astype("category")
        y_pred = y_pred.astype("category").cat.set_categories(y_true.cat.categories)

        # Convert to integer-coded labels
        y_true_all = y_true.cat.codes.values
        y_pred = y_pred.cat.codes.values

    else:
        if model.__class__.__name__ == "scANVI" or model.__class__.__name__ == "Baseline":
            # Compute latent representation
            latent_rep = model.z2l_encoder(dataloader.X)[0]
        elif model.__class__.__name__ == "DCAG":
            latent_rep = model.zyl_encoder(dataloader.X)[0]

        # Get predicted probabilities 
        y_logits = model.y_classifier(latent_rep)
        y_probs = softmax(y_logits, dim=-1).data.cpu().numpy()
        y_pred = np.argsort(y_probs, axis=1)[:, -1]

        # Get ground truth labels from dataloader (tensor -> numpy)
        y_true_all = dataloader.Y.cpu().numpy()

        if y_true_all.ndim > 1:
            # In case it is one-hot encoded
            mask_all_invalid = (y_true_all == -1).all(axis=1)
            y_true_all = np.argmax(y_true_all, axis=1)
            y_true_all[mask_all_invalid] = -1

        # Compute metrics only on labeled data
        valid_indices = np.where(y_true_all != -1)[0]
        y_true_all = y_true_all[valid_indices]
        y_pred = y_pred[valid_indices]

    # Compute accuracy
    accuracy = plot_confusion(y_true_all, y_pred, f"Confusion Matrix ({train_test})")
    print(f"Balanced Accuracy ({train_test}): {accuracy:.4f}")

    # Compute F1 score
    f1 = f1_score(y_true_all, y_pred, average='macro')
    print(f"F1 Score ({train_test}): {f1:.4f}")

    # Compute individual class accuracies
    classes = {0: 'A', 1: 'B', 2: 'C'}
    class_accuracies = {}
    for clas, name in classes.items():
        cls_indices = np.where(y_true_all == clas)[0]
        if len(cls_indices) > 0:
            acc = accuracy_score(y_true_all[cls_indices], y_pred[cls_indices])
            print(f"Accuracy {name} ({train_test} ): {acc:.4f}")
            class_accuracies[name] = acc
        else:
            print(f"Accuracy {name} ({train_test}): N/A (no samples)")

    # Compute A+B vs C accuracy
    # Convert to binary: 1 = A or B, 0 = C
    y_true_ab_c = np.isin(y_true_all, [0, 1]).astype(int)
    y_pred_ab_c = np.isin(y_pred, [0, 1]).astype(int)

    accuracy_ab_c = balanced_accuracy_score(y_true_ab_c, y_pred_ab_c)
    print(f"Accuracy (A+B vs C) ({train_test}): {accuracy_ab_c:.4f}")

    return accuracy, f1, accuracy_ab_c, class_accuracies

def compute_silhouette_score(model, dataloader, n_components_var_per=0.6):
    """
    Computes the silhouette score for the latent representation of the data.
    Parameters:
        scanvi: Trained scANVI model
        dataloader: BatchDataLoader or similar with X and Y attributes
        n_components: Number of latent dimensions to use (default: 2)
    Returns:
        Silhouette score (float)
    """
    
    if model.__class__.__name__ == "SCANVI":
        X = dataloader.X
        Y = dataloader.obs["CAGPHASE"]
        CT = dataloader.obs["CELLTYPE"]

        latent_rep = dataloader.obsm['X_scANVI']

    else:
        X = dataloader.X
        Y = dataloader.Y
        CT = dataloader.CT

        if model.__class__.__name__ == "scANVI" or model.__class__.__name__ == "Baseline":
            # Compute latent representation
            latent_rep = model.z2l_encoder(X)[0].detach().cpu().numpy()
        elif model.__class__.__name__ == "DCAG":
            latent_rep = model.zyl_encoder(X)[0].detach().cpu().numpy()
        else:
            raise ValueError("Model must be scANVI, DCAG, or Baseline")
        
    # Reduce to n_components that explain > variance than n_components_var_per
    #  if latent is higher-dimensional
    if latent_rep.shape[1] > 1:
        pca = PCA(n_components=n_components_var_per)
        latent_rep = pca.fit_transform(latent_rep)
    
    # Get labels (convert one-hot to integer if needed)
    if CT.ndim > 1:
        CT = torch.argmax(CT, dim=1)

    if Y.ndim > 1:
        Y = torch.argmax(Y, dim=1)

    if model.__class__.__name__ == "SCANVI":
        Y = Y.values
        CT = CT.values
    else:
        Y = Y.cpu().numpy()
        CT = CT.cpu().numpy()

    ct_score = silhouette_score(latent_rep, CT)
    y_score = silhouette_score(latent_rep, Y)
    
    return ct_score, y_score

def evaluate_model(model, dataloader, test_dataloader, args, dataset_name, model_sufix, output_path="/pool01/projects/abante_lab/cag_propagation/scanvi_results/"):
    """
    Writes model evaluation metrics to a TSV file.
    """
    
    # Evaluate the model classification accuracy
    superprint("Evaluating SCANVI model...")
    ba_train, f1_train, accuracy_ab_c_train, class_accuracies_train = performance_metrics(model, dataloader, train_test='train')
    ba_test, f1_test, accuracy_ab_c_test, class_accuracies_test = performance_metrics(model, test_dataloader, train_test='test')

    # Evaluate model on the training cells with masked label
    if args.unseen_y_prop != 0.0:
        m_accuracy = masked_accuracy(model, dataloader, args)
    else:
        m_accuracy = None

    # Evaluate the model latent space with silhouette score
    superprint("Evaluating SCANVI latent space...")
    ct_sil_train, y_sil_train = compute_silhouette_score(model, dataloader)
    ct_sil_test, y_sil_test = compute_silhouette_score(model, test_dataloader)

    superprint("Saving model evaluation metrics...")
    # Store model metrics
    if model.__class__.__name__ == "SCANVI":
        metrics = {
            "seed": [args.seed],
            "dataset": [dataset_name],
            "n_hidden_layers":[f'{args.n_hidden_layers}'],
            "latent_dim": [args.latent_dim],
            "unseen_y_prop": [f'{args.unseen_y_prop}'],
            "balanced_accuracy_train": [ba_train],
            "balanced_accuracy_test": [ba_test],
            "f1_train": [f1_train],
            "f1_test": [f1_test],
            "ct_silhouette_train": [ct_sil_train],
            "ct_silhouette_test": [ct_sil_test],
            "y_silhouette_train": [y_sil_train],
            "y_silhouette_test": [y_sil_test],
            "accuracy_ab_c_train": [accuracy_ab_c_train],
            "accuracy_ab_c_test": [accuracy_ab_c_test],
            "class_accuracies_train": [class_accuracies_train],
            "class_accuracies_test": [class_accuracies_test],
            "balanced_accuracy_masked": [m_accuracy]
        }
        if args.multitech:
            metrics_file = os.path.join(f"{output_path}{model.__class__.__name__}_semisupervised_multitech_metrics.tsv")
        else:
            metrics_file = os.path.join(f"{output_path}{model.__class__.__name__}_supervised_metrics.tsv")
    elif model.__class__.__name__ == "scANVI" or model.__class__.__name__ == "Baseline":
        metrics = {
            "seed": [args.seed],
            "dataset": [dataset_name],
            "n_hidden_layers":[f'{args.n_hidden_layers}'],
            "alpha": [args.alpha_y],
            "latent_dim": [args.latent_dim_y],
            "unseen_y_prop": [f'{args.unseen_y_prop}'],
            "balanced_accuracy_train": [ba_train],
            "balanced_accuracy_test": [ba_test],
            "f1_train": [f1_train],
            "f1_test": [f1_test],
            "ct_silhouette_train": [ct_sil_train],
            "ct_silhouette_test": [ct_sil_test],
            "y_silhouette_train": [y_sil_train],
            "y_silhouette_test": [y_sil_test],
            "accuracy_ab_c_train": [accuracy_ab_c_train],
            "accuracy_ab_c_test": [accuracy_ab_c_test],
            "class_accuracies_train": [class_accuracies_train],
            "class_accuracies_test": [class_accuracies_test],
            "balanced_accuracy_masked": [m_accuracy]
        }
        if args.multitech:
            metrics_file = os.path.join(f"{output_path}{model.__class__.__name__}_semisupervised_multitech_metrics.tsv")
        else:
            metrics_file = os.path.join(f"{output_path}{model.__class__.__name__}_supervised_metrics.tsv")
    elif model.__class__.__name__ == "DCAG":
        metrics = {
            "seed": [args.seed],
            "dataset": [dataset_name],
            "n_hidden_layers":[f'{args.n_hidden_layers}'],
            "lamda_reg": [f'{args.lambda_reg}'],
            "alpha_y": [args.alpha_y],
            "alpha_ct": [args.alpha_ct],
            "latent_dim_y": [args.latent_dim_y],
            "latent_dim_ct": [args.latent_dim_ct],
            "unseen_y_prop": [f'{args.unseen_y_prop}'],
            "balanced_accuracy_train": [ba_train],
            "balanced_accuracy_test": [ba_test],
            "f1_train": [f1_train],
            "f1_test": [f1_test],
            "ct_silhouette_train": [ct_sil_train],
            "ct_silhouette_test": [ct_sil_test],
            "y_silhouette_train": [y_sil_train],
            "y_silhouette_test": [y_sil_test],
            "accuracy_ab_c_train": [accuracy_ab_c_train],
            "accuracy_ab_c_test": [accuracy_ab_c_test],
            "class_accuracies_train": [class_accuracies_train],
            "class_accuracies_test": [class_accuracies_test],
            "balanced_accuracy_masked": [m_accuracy]
        }
        if args.multitech:
            metrics_file = os.path.join(f"{output_path}{model.__class__.__name__}_{args.reg_method}_semisupervised_multitech_metrics.tsv")
        else:
            metrics_file = os.path.join(f"{output_path}{model.__class__.__name__}_{args.reg_method}_supervised_metrics.tsv")
    metrics_df = pd.DataFrame(metrics)
    write_header = not os.path.exists(metrics_file)
    metrics_df.to_csv(metrics_file, sep="\t", index=False, mode='a', header=write_header)

    superprint(f"Model evaluation saved in {metrics_file}.")

    if ba_test > 0.7:
        # Save the model if test balanced accuracy > 70%
        model_file = os.path.join(f"{output_path}{model_sufix}_model.pth")
        torch.save(model.state_dict(), model_file)
        superprint(f"Model saved in {model_file}.")

def encode_all(model, args):
    """
    Encode all cells and add latent representation to adata.
    """

    X, _, _, _, _, _, _, adata = load_dataset_separate_variables(args.dataset_path, cuda = args.cuda)

    # Compute latent representation
    if model.__class__.__name__ == "scANVI" or model.__class__.__name__ == "Baseline":
        latent_rep_zy = model.z2l_encoder(X)[0]

        # Add latent representation to adata
        adata.obsm[f"Z_{model.__class__.__name__}"] = latent_rep_zy.detach().cpu().numpy()
    
    elif model.__class__.__name__ == "DCAG":
        latent_rep_zy = model.zyl_encoder(X)[0]
        latent_rep_zct = model.zctl_encoder(X)[0]

        # Add latent representation to adata
        adata.obsm["Z_y_DCAG"] = latent_rep_zy.detach().cpu().numpy()
        adata.obsm["Z_ct_DCAG"] = latent_rep_zct.detach().cpu().numpy()
    
    return adata

def get_model_sufix(model, args):
    """
    Constructs model sufix based on model hyperparameters for saving.
    """

    # Dataset name extraction
    if not args.multitech:
        dataset_ops = args.dataset_path.split('/')[-1].split('.')[0].split('_')[-2:]
        dataset_name = '_'.join(dataset_ops)
    else:
        dataset_name = '.'.join(args.dataset_path.split('/')[-1].split('.')[:-1])

    # Construct model sufix based on model hyperparameters
    if model.__class__.__name__ == "scANVI":
        if args.multitech:
            model_sufix = f'{model.__class__.__name__}_semisupervised_a{args.alpha_y}_latentdim{args.latent_dim_y}_nhidden{args.n_hidden_layers}_dataset{dataset_name}_seed{args.seed}'
        else:
            model_sufix = f'{model.__class__.__name__}_supervised_a{args.alpha_y}_latentdim{args.latent_dim_y}_nhidden{args.n_hidden_layers}_masked{args.unseen_y_prop}_dataset{dataset_name}_seed{args.seed}'

    if model.__class__.__name__ == "SCANVI":
        if args.multitech:
            model_sufix = f'{model.__class__.__name__}_semisupervised_hiddenlayers{args.n_hidden_layers}_latentdim{args.latent_dim}_dataset{dataset_name}_seed{args.seed}'
        else:
            model_sufix = f'{model.__class__.__name__}_supervised_hiddenlayers{args.n_hidden_layers}_latentdim{args.latent_dim}_masked{args.unseen_y_prop}_dataset{dataset_name}_seed{args.seed}'

    elif model.__class__.__name__ == "DCAG":
        # model_sufix = f'{model.__class__.__name__}_supervised_ay{args.alpha_y}_act{args.alpha_ct}_latentdimy{args.latent_dim_y}_latentdimct{args.latent_dim_ct}_seed{args.seed}'
        if args.multitech:
            model_sufix = f'{model.__class__.__name__}_{args.reg_method}_semisupervised_ay{args.alpha_y}_act{args.alpha_ct}_latentdimy{args.latent_dim_y}_latentdimct{args.latent_dim_ct}_lambdareg{args.lambda_reg}_nhidden{args.n_hidden_layers}_dataset{dataset_name}_seed{args.seed}'
        else:
            model_sufix = f'{model.__class__.__name__}_{args.reg_method}_supervised_ay{args.alpha_y}_act{args.alpha_ct}_latentdimy{args.latent_dim_y}_latentdimct{args.latent_dim_ct}_lambdareg{args.lambda_reg}_masked{args.unseen_y_prop}_nhidden{args.n_hidden_layers}_dataset{dataset_name}_seed{args.seed}'
        
    elif model.__class__.__name__ == "Baseline":
        if args.multitech:
            model_sufix = f'{model.__class__.__name__}_semisupervised_alpha{args.alpha_y}_latentdim{args.latent_dim_y}_nhidden{args.n_hidden_layers}_dataset{dataset_name}_seed{args.seed}'
        else:
            model_sufix = f'{model.__class__.__name__}_supervised_alpha{args.alpha_y}_latentdim{args.latent_dim_y}_nhidden{args.n_hidden_layers}_dataset{dataset_name}_seed{args.seed}'
        
    return model_sufix, dataset_name

def alpha_scal(unseen_y_prop):
    """
    Calculates alpha for CAG classification weighting based on the proportion of unseen labels.
    Equation derived from polynomial regression on empirical data.
    """
    return 20.5209*unseen_y_prop**2 - 7.4546*unseen_y_prop + 0.359