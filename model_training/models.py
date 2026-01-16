#!/usr/bin/env python3
#%%
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.patches import Patch
from torch.distributions import constraints
from torch.nn.functional import softmax, softplus
from torch.optim import Adam
import torch.nn.functional as F
import scanpy as sc
from scipy.special import gamma
from pyro.nn import PyroModule
# from torch_mist import estimate_mi

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine

from pyro.distributions.util import broadcast_shape
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate
from pyro.optim import MultiStepLR

# Helper for making fully-connected neural networks
def make_fc(dims, dropout=0.2):
    layers = []
    for in_dim, out_dim in zip(dims, dims[1:]):
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(nn.Dropout(dropout))
        layers.append(nn.LayerNorm(out_dim))
        layers.append(nn.ReLU())
    return nn.Sequential(*layers[:-1])  # Exclude final ReLU non-linearity

def calculate_hidden_dims(input_dim, output_dim, n_hidden_layers):
    """
    Calculate hidden layer dimensions that decrease from input_dim to output_dim
    using powers of 2.
    """

    # Get the range of exponents (powers of 2)
    start_exp = int(np.floor(np.log2(input_dim)))
    end_exp = int(np.ceil(np.log2(output_dim)))

    # Choose equispaced exponents between start and end
    exps = np.linspace(start_exp, end_exp, n_hidden_layers + 2)[1:-1]  # drop input & latent

    # Convert back to dimensions (round to nearest power of 2)
    hidden_dims = [2 ** int(round(e)) for e in exps]

    return hidden_dims

# Splits a tensor in half along the final dimension
def split_in_half(t):

    return t.reshape(t.shape[:-1] + (2, -1)).unbind(-2)

# Helper for broadcasting inputs to neural net
def broadcast_inputs(input_args):
    shape = broadcast_shape(*[s.shape[:-1] for s in input_args]) + (-1,)
    input_args = [s.expand(shape) for s in input_args]
    return input_args

# Used in parameterizing p(z2 | z1, y)
class Z2Decoder(nn.Module):
    
    def __init__(self, z1_dim, y_dim, z2_dim, hidden_dims):
        
        super().__init__()
        
        # dimensions of the input to the neural network
        dims = [z1_dim + y_dim] + hidden_dims + [2 * z2_dim]
        
        self.fc = make_fc(dims)

    def forward(self, z1, y):

        # concat z1 and y on the batch dimension
        z1_y = torch.cat([z1, y], dim=-1)
        
        # TODO: We reshape the input to be two-dimensional so that nn.BatchNorm1d behaves correctly
        # _z1_y = z1_y.reshape(-1, z1_y.size(-1))
        # hidden = self.fc(_z1_y)
        hidden = self.fc(z1_y)
        
        # If the input was three-dimensional we now restore the original shape
        # hidden = hidden.reshape(z1_y.shape[:-1] + hidden.shape[-1:])
        loc, scl = split_in_half(hidden)
        
        # Here and elsewhere softplus ensures that scale is positive. Note that we generally
        # expect softplus to be more numerically stable than exp.
        scl = softplus(scl)
        
        return loc, scl

# Used in parameterizing p(x | z2)
class XDecoder(nn.Module):
    def __init__(self, num_genes, z2_dim, hidden_dims):
        super().__init__()
        dims = [z2_dim] + hidden_dims + [2 * num_genes]
        self.fc = make_fc(dims)

    def forward(self, z2):
        gate_logits, mu = split_in_half(self.fc(z2))
        mu = softmax(mu, dim=-1)
        return gate_logits, mu

# Used in parameterizing q(z2 | x) and q(l | x)
class Z2LEncoder(nn.Module):

    def __init__(self, num_genes, z2_dim, hidden_dims):
        super().__init__()
        
        dims = [num_genes] + hidden_dims + [2 * z2_dim + 2]
        
        self.fc = make_fc(dims)

    def forward(self, x):
        # Transform the counts x to log space for increased numerical stability.
        # Note that we only use this transform here; in particular the observation
        # distribution in the model is a proper count distribution.
        x = torch.log1p(x)
        h1, h2 = split_in_half(self.fc(x))
        z2_loc, z2_scale = h1[..., :-1], softplus(h2[..., :-1])
        l_loc, l_scale = h1[..., -1:], softplus(h2[..., -1:])
        return z2_loc, z2_scale, l_loc, l_scale

# Used in parameterizing q(z1 | z2, y)
class Z1Encoder(nn.Module):

    def __init__(self, num_cag_cat, z1_dim, z2_dim, hidden_dims):
        
        super().__init__()
        
        dims = [num_cag_cat + z2_dim] + hidden_dims + [2 * z1_dim]
        
        self.fc = make_fc(dims)

    def forward(self, z2, y):

        # This broadcasting is necessary since Pyro expands y during enumeration (but not z2)
        z2_y = broadcast_inputs([z2, y])
        z2_y = torch.cat(z2_y, dim=-1)
        
        # We reshape the input to be two-dimensional so that nn.BatchNorm1d behaves correctly
        _z2_y = z2_y.reshape(-1, z2_y.size(-1))
        hidden = self.fc(_z2_y)
        
        # If the input was three-dimensional we now restore the original shape
        hidden = hidden.reshape(z2_y.shape[:-1] + hidden.shape[-1:])
        loc, scl = split_in_half(hidden)
        scl = softplus(scl)
        
        return loc, scl

# Used in parameterizing q(y | z2)
class Classifier(nn.Module):
    def __init__(self, z2_dim, hidden_dims, num_cag_cat, dropout=0.2):
        super().__init__()
        dims = [z2_dim] + hidden_dims + [num_cag_cat]
        # Don't change FC achitecture here
        self.fc = make_fc(dims)

    def forward(self, x):
        logits = self.fc(x)
        return logits
    
def gram_matrix(x: torch.Tensor, y: torch.Tensor, gammas) -> torch.Tensor:
    """
    (From Yosef Lab, https://github.com/suinleelab/StrastiveVI/tree/main)
    Calculate the maximum mean discrepancy gram matrix with multiple gamma values.

    Args:
    ----
        x: Tensor with shape (B, P, M) or (P, M).
        y: Tensor with shape (B, R, M) or (R, M).
        gammas: 1-D tensor with the gamma values.

    Returns
    -------
        A tensor with shape (B, P, R) or (P, R) for the distance between pairs of data
        points in `x` and `y`.
    """
    # if not torch.is_tensor(gammas):
    #     gammas = torch.ones(x.shape[1], dtype=torch.float32) * gammas
    #     gammas = gammas.to(x.device)

    # gammas = gammas.unsqueeze(1)
    pairwise_distances = torch.cdist(x, y, p=2.0)

    pairwise_distances_sq = torch.square(pairwise_distances)
    # tmp = torch.matmul(gammas, torch.reshape(pairwise_distances_sq, (1, -1)))
    tmp = gammas * torch.reshape(pairwise_distances_sq, (1, -1))
    tmp = torch.reshape(torch.sum(torch.exp(-tmp), 0), pairwise_distances_sq.shape) ### shape of tmp: (1, batch_size*batch_size)
    return tmp

def hsic(x, y): #, gamma=1.):
    """
    (From Yosef Lab, https://github.com/suinleelab/StrastiveVI/tree/main)
    Calculate the empirical estimate of the Hilbert-Schmidt Independence Criterion (HSIC)
    between two variables X and Y given samples from each variable.
    """
    m = x.shape[0]
    d_x = x.shape[1]
    g_x = 2 * gamma(0.5 * (d_x+1)) / gamma(0.5 * d_x)
    K = gram_matrix(x, x, gammas=1./(2. * g_x))

    d_y = y.shape[1]
    g_y = 2 * gamma(0.5 * (d_y+1)) / gamma(0.5 * d_y)
    L = gram_matrix(y, y, gammas=1./(2. * g_y))
    
    H = torch.eye(m) - 1.0 / m * torch.ones((m, m))
    H = H.float().cuda()
    HSIC = torch.trace(torch.mm(L, torch.mm(H, torch.mm(K, H)))) / ((m - 1) ** 2)

    return HSIC

class CustomELBO(pyro.infer.Trace_ELBO):
    """Custom ELBO that adds a regularization term to enforce independence between Zx and Zy."""

    def __init__(self, lambda_reg=1e6, reg_method = 'ortho', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_reg = lambda_reg
        self.reg_method = reg_method
        self.hsic_val = 0

    def compute_orthogonality_loss(self, Zx, Zy):
        """Compute the orthogonality loss between Zx and Zy."""
        # Zx: [batch_size, dim_x], Zy: [batch_size, dim_y]
        # Center
        Zx_centered = Zx - Zx.mean(dim=0, keepdim=True)
        Zy_centered = Zy - Zy.mean(dim=0, keepdim=True)

        # Standardize
        std_Zx = Zx_centered.std(dim=0, keepdim=True) + 1e-8
        std_Zy = Zy_centered.std(dim=0, keepdim=True) + 1e-8

        Zx_norm = Zx_centered / std_Zx
        Zy_norm = Zy_centered / std_Zy

        # Cross-correlation matrix: [dim_x, dim_y]
        cross_corr = Zx_norm.T @ Zy_norm / Zx.shape[0]

        # Frobenius norm as penalty
        norm = torch.norm(cross_corr, p="fro")
        normalized_norm = norm / cross_corr.shape[0]

        return self.lambda_reg * normalized_norm

    def loss(self, model, guide, *args, **kwargs):
        # Compute the standard ELBO loss
        loss = super().loss(model, guide, *args, **kwargs)

        # Run guide to get Zx and Zy
        guide_trace = pyro.poutine.trace(guide).get_trace(*args, **kwargs)
        Zx = guide_trace.nodes["z_ct"]["value"]
        Zy = guide_trace.nodes["z_y"]["value"]

        if self.reg_method == 'ortho':
            # Compute orthogonality loss
            ortho_loss = self.compute_orthogonality_loss(Zx, Zy)

            return loss + ortho_loss # already multiplied by lambda_reg in the function
        
        elif self.reg_method == 'hsic':
            # Compute HSIC
            hsic_value = hsic(Zx, Zy)
            self.hsic_val = hsic_value.item()

            return loss + self.lambda_reg * hsic_value
        else:
            raise ValueError(f"Unknown reg_method: {self.reg_method}")
    
# First prototype of the SSVAE model for CAG label propagation.
class scANVI(nn.Module):

    # TODO: consider the following improvements:
    # [ ] Deal with semi-supervised learning: only works if batches are fully labeled or fully unlabeled
    # [ ] Consider an ordinal classification loss for y if the CAGs have an order
    # [ ] Come up with a different name for the model (since it's not really scANVI)
    
    def __init__(
        self,
        num_genes,
        num_cag_cat,
        l_loc,
        l_scale,
        latent_dim=20,
        alpha=0.01,
        scale_factor=1.0,
        n_hidden_layers=1
    ):
        assert isinstance(num_genes, int)
        self.num_genes = num_genes

        assert isinstance(num_cag_cat, int) and num_cag_cat > 1
        self.num_cag_cat = num_cag_cat

        # This is the dimension of both z1 and z2
        assert isinstance(latent_dim, int) and latent_dim > 0
        self.latent_dim = latent_dim

        self.n_hidden_layers = n_hidden_layers

        # The next two hyperparameters determine the prior over the log_count latent variable `l`
        assert isinstance(l_loc, float)
        self.l_loc = l_loc
        assert isinstance(l_scale, float) and l_scale > 0
        self.l_scale = l_scale

        # This hyperparameter controls the strength of the auxiliary classification loss
        assert isinstance(alpha, float) and alpha > 0
        self.alpha = alpha

        assert isinstance(scale_factor, float) and scale_factor > 0
        self.scale_factor = scale_factor

        super().__init__()

        # Setup the various neural networks used in the model and guide. TODO: review dimensions
        self.z2_decoder = Z2Decoder(
            z1_dim=self.latent_dim,
            y_dim=self.num_cag_cat,
            z2_dim=self.latent_dim,
            hidden_dims=sorted(calculate_hidden_dims(self.latent_dim + self.num_cag_cat, 2 * self.latent_dim, self.n_hidden_layers)),
        )
        self.x_decoder = XDecoder(
            num_genes=num_genes, 
            hidden_dims=sorted(calculate_hidden_dims(self.latent_dim, 2 * self.num_genes, self.n_hidden_layers)),
            z2_dim=self.latent_dim
        )
        self.z2l_encoder = Z2LEncoder(
            num_genes=num_genes, z2_dim=self.latent_dim, 
            hidden_dims=sorted(calculate_hidden_dims(self.num_genes, 2 * self.latent_dim + 2, self.n_hidden_layers), reverse=True)
        )
        self.y_classifier = Classifier(
            z2_dim=self.latent_dim, hidden_dims=[50], num_cag_cat=num_cag_cat
        )
        self.z1_encoder = Z1Encoder(
            num_cag_cat=num_cag_cat,
            z1_dim=self.latent_dim,
            z2_dim=self.latent_dim,
            hidden_dims=sorted(calculate_hidden_dims(self.latent_dim + self.num_cag_cat, 2 * self.latent_dim, self.n_hidden_layers), reverse=True),
            # hidden_dims=[50],
        )

        self.epsilon = 5.0e-3

    def model(self, x, y=None):
        
        # Register various nn.Modules with Pyro
        pyro.module("scanvi", self)

        # This gene-level parameter modulates the variance of the observation distribution
        theta = pyro.param("inverse_dispersion", 10.0 * x.new_ones(self.num_genes), constraint=constraints.positive)
        
        # Alternatively, we can use a uniform prior over the class probabilities
        cag_class_probs = 0.3 * torch.ones(self.num_cag_cat, device=x.device)

        # scale_factor so that the ELBO is normalized wrt the number of datapoints and genes
        with pyro.plate("batch", x.shape[0]), poutine.scale(scale=self.scale_factor):

            # sample agnostic latent variable z1 from a standard normal distribution
            z1 = pyro.sample("z1", dist.Normal(0, x.new_ones(self.latent_dim)).to_event(1))

            # Check if batch is unlabeled
            is_unlabeled = (y < 0).any()

            # Sample y from OneHotCategorical
            if is_unlabeled:
                y = pyro.sample("y", dist.OneHotCategorical(probs=cag_class_probs))
            else:
                y = pyro.sample("y", dist.OneHotCategorical(probs=cag_class_probs), obs=y)

            # Sample z2 from the decoder using z1 and y
            z2_loc, z2_scale = self.z2_decoder(z1, y)
            z2 = pyro.sample("z2", dist.Normal(z2_loc, z2_scale).to_event(1))

            # Sample l from the encoder using x
            l_scale = self.l_scale * x.new_ones(1)
            l = pyro.sample("l", dist.LogNormal(self.l_loc, l_scale).to_event(1))

            # Note that by construction mu is normalized (i.e. mu.sum(-1) == 1) and the
            # total scale of counts for each cell is determined by `l`
            gate_logits, mu = self.x_decoder(z2)
            
            # compute the logits for the observation distribution
            nb_logits = (l * mu + self.epsilon).log() - (theta + self.epsilon).log()
            
            # sample x from the observation distribution
            x_dist = dist.ZeroInflatedNegativeBinomial(
                gate_logits=gate_logits, total_count=theta, logits=nb_logits
            )
            
            # Observe the datapoint x using the observation distribution x_dist
            pyro.sample("x", x_dist.to_event(1), obs=x)

    # The guide specifies the variational distribution
    def guide(self, x, y=None):

        pyro.module("scanvi", self)
        
        # NOTE: first we sample Z2 and L, then Y|Z2 if not observed, and then we sample Z1|Z2,Y

        # scale_factor so that the ELBO is normalized wrt the number of datapoints and genes
        with pyro.plate("batch", x.shape[0]), poutine.scale(scale=self.scale_factor):
        
            # Sample z2 and l from the encoder using x
            z2_loc, z2_scale, l_loc, l_scale = self.z2l_encoder(x)
            
            # Sample l from the LogNormal distribution
            pyro.sample("l", dist.LogNormal(l_loc, l_scale).to_event(1))
            
            # Sample z2 from the Normal distribution
            z2 = pyro.sample("z2", dist.Normal(z2_loc, z2_scale).to_event(1))

            # Sample y from the classifier using z2
            y_logits = self.y_classifier(z2)
            y_dist = dist.OneHotCategorical(logits=y_logits)

            # Check if batch is unlabeled
            is_unlabeled = (y < 0).any()
            
            if is_unlabeled or y is None:
                
                # x is unlabeled so sample y using q(y|z2)
                y = pyro.sample("y", y_dist)
            
            else:
                
                # x is labeled so add a classification loss term
                # (this way q(y|z2) learns from both labeled and unlabeled data)
                classification_loss = y_dist.log_prob(y)

                # Note that the negative sign appears because we're adding this term in the guide
                # and the guide log_prob appears in the ELBO as -log q
                pyro.factor("classification_loss", -self.alpha * classification_loss, has_rsample=False)

            # Sample z1 from the encoder using z2 and y
            z1_loc, z1_scale = self.z1_encoder(z2, y)
            pyro.sample("z1", dist.Normal(z1_loc, z1_scale).to_event(1))

class DCAG(nn.Module):
    def __init__(
        self,
        num_genes,
        num_cag_cat,
        num_ct_cat,
        l_loc,
        l_scale,
        latent_dim_y=20,
        latent_dim_ct=20,
        alpha_y=0.01,
        alpha_ct=0.01,
        scale_factor=1.0,
        class_weights_y=None,
        n_hidden_layers=1
    ):
        super().__init__()

        self.num_genes = num_genes
        self.num_cag_cat = num_cag_cat
        self.num_ct_cat = num_ct_cat
        self.n_hidden_layers = n_hidden_layers

        self.latent_dim_y = latent_dim_y
        self.latent_dim_ct = latent_dim_ct

        # hidden layers size 
        hidden_dim_zyl = calculate_hidden_dims(self.num_genes, self.latent_dim_y, self.n_hidden_layers)
        hidden_dim_zctl = calculate_hidden_dims(self.num_genes, self.latent_dim_ct, self.n_hidden_layers)
        hidden_dim_x = calculate_hidden_dims(self.num_genes, self.latent_dim_ct + self.latent_dim_y, self.n_hidden_layers)

        self.l_loc = l_loc
        self.l_scale = l_scale

        self.alpha_y = alpha_y
        self.alpha_ct = alpha_ct

        self.scale_factor = scale_factor
        self.epsilon = 5.0e-3

        # Class weights for Y (for imbalanced classes)
        if class_weights_y is not None:
            self.class_weights_y = torch.tensor(class_weights_y, dtype=torch.float32)
        else:
            self.class_weights_y = torch.ones(num_cag_cat, dtype=torch.float32)

        # decoders
        self.x_decoder = XDecoder(
            num_genes=num_genes,
            z2_dim=latent_dim_y + latent_dim_ct,
            hidden_dims=sorted(hidden_dim_x),
        )

        # encoders
        self.zyl_encoder = Z2LEncoder(
            num_genes=num_genes,
            z2_dim=latent_dim_y,
            hidden_dims=sorted(hidden_dim_zyl,reverse=True),
        )
        self.zctl_encoder = Z2LEncoder(
            num_genes=num_genes,
            z2_dim=latent_dim_ct,
            hidden_dims=sorted(hidden_dim_zctl,reverse=True),
        )

        # classifiers (used in both model + guide)
        self.y_classifier = Classifier(
            z2_dim=latent_dim_y,
            hidden_dims=[latent_dim_y//2],
            num_cag_cat=num_cag_cat,
        )
        self.ct_classifier = Classifier(
            z2_dim=latent_dim_ct,
            hidden_dims=[latent_dim_ct//2],
            num_cag_cat=num_ct_cat,
        )

    def model(self, x, y=None, ct=None):
        pyro.module("scstrastivevi", self)

        theta = pyro.param(
            "inverse_dispersion",
            10.0 * x.new_ones(self.num_genes),
            constraint=constraints.positive,
        )

        with pyro.plate("batch", x.shape[0]), poutine.scale(scale=self.scale_factor):
            
            # priors on z_y and z_ct
            z_y = pyro.sample("z_y", dist.Normal(0, x.new_ones(self.latent_dim_y)).to_event(1))
            z_ct = pyro.sample("z_ct", dist.Normal(0, x.new_ones(self.latent_dim_ct)).to_event(1))

            # generate y from z_y
            y_logits = self.y_classifier(z_y)

            # Check if batch is unlabeled
            is_unlabeled = (y < 0).any()

            # Sample y from OneHotCategorical
            if is_unlabeled:
                y = pyro.sample("y", dist.OneHotCategorical(logits=y_logits))
            else:
                y = pyro.sample("y", dist.OneHotCategorical(logits=y_logits), obs=y)

            # generate ct from z_ct
            ct_logits = self.ct_classifier(z_ct)

            ct_unlabeled = (ct < 0).any()
            if ct_unlabeled:
                pyro.sample("ct", dist.Categorical(logits=ct_logits))
            else:
                pyro.sample("ct", dist.Categorical(logits=ct_logits), obs=ct)

            # prior on l
            l_scale = self.l_scale * x.new_ones(1)
            l = pyro.sample("l", dist.LogNormal(self.l_loc, l_scale).to_event(1))

            # combine z_y and z_ct → generate X
            z_concat = torch.cat([z_y, z_ct], dim=-1)
            gate_logits, mu = self.x_decoder(z_concat)

            nb_logits = (l * mu + self.epsilon).log() - (theta + self.epsilon).log()
            x_dist = dist.ZeroInflatedNegativeBinomial(
                gate_logits=gate_logits, total_count=theta, logits=nb_logits
            )
            pyro.sample("x", x_dist.to_event(1), obs=x)

    def guide(self, x, y=None, ct=None):
        pyro.module("scstrastivevi", self)

        tot_class_loss_y = []
        tot_class_loss_ct = []

        with pyro.plate("batch", x.shape[0]), poutine.scale(scale=self.scale_factor):

            # encode z_y, l, z_ct
            z_y_loc, z_y_scale, l_loc, l_scale = self.zyl_encoder(x)
            z_ct_loc, z_ct_scale, _, _ = self.zctl_encoder(x)

            pyro.sample("l", dist.LogNormal(l_loc, l_scale).to_event(1))
            z_y = pyro.sample("z_y", dist.Normal(z_y_loc, z_y_scale).to_event(1))
            z_ct = pyro.sample("z_ct", dist.Normal(z_ct_loc, z_ct_scale).to_event(1))

            # classifier for y
            y_logits = self.y_classifier(z_y)
            y_dist = dist.OneHotCategorical(logits=y_logits)

            # Check if batch is unlabeled
            is_unlabeled = (y < 0).any()

            if y is None or is_unlabeled:
                pyro.sample("y", y_dist)
            else:
                # compute log probabilities
                log_prob = y_dist.log_prob(y)
                
                # Calculate class weight per batch with pseudocount
                y_idx = torch.argmax(y, dim=-1)
                y_counts = torch.bincount(y_idx, minlength=self.num_cag_cat).float() + 1.0
                batch_weights = y_counts / y_counts.sum()
                batch_weights = batch_weights.to(y.device)[y_idx]
                
                # Apply class weights inversely proportional to class prevalence for Y
                # REMOVED WHEN WE HAVE DEALTH WITH CLASS D!
                # mask = (y_idx != (self.num_cag_cat - 1)).float()
                # class_loss_y = -self.alpha_y * log_prob / batch_weights * mask
                class_loss_y = -self.alpha_y * log_prob / batch_weights

                # record classification loss
                pyro.factor(
                    "classification_loss_y",
                    class_loss_y,
                    has_rsample=False,
                )
                tot_class_loss_y.append(class_loss_y.mean())

            # classifier for ct (no weighting, but could be added similarly)
            ct_logits = self.ct_classifier(z_ct)
            ct_dist = dist.Categorical(logits=ct_logits)

            ct_unlabeled = (ct < 0).any()

            if ct is None or ct_unlabeled:
                pyro.sample("ct", ct_dist)
            else:
                class_loss_ct = -self.alpha_ct * ct_dist.log_prob(ct)
                pyro.factor(
                    "classification_loss_ct",
                    class_loss_ct,
                    has_rsample=False,
                )
                tot_class_loss_ct.append(class_loss_ct.mean())

        # Print summed classification losses after the plate
        total_loss_y = torch.mean(torch.stack(tot_class_loss_y)) if tot_class_loss_y else torch.tensor(0.0)
        total_loss_ct = torch.mean(torch.stack(tot_class_loss_ct)) if tot_class_loss_ct else torch.tensor(0.0)
        print(f"L(class y): {total_loss_y.item():.4f}")
        print(f"L(class ct): {total_loss_ct.item():.4f}")

class Baseline(PyroModule):
    def __init__(self, num_genes, num_cag_cat, l_loc, l_scale,
                 latent_dim=20, alpha=0.01, scale_factor=1.0, n_hidden_layers=1):
        super().__init__()
        self.num_genes = num_genes
        self.num_cag_cat = num_cag_cat
        self.latent_dim = latent_dim
        self.l_loc = l_loc
        self.l_scale = l_scale
        self.alpha = alpha
        self.scale_factor = scale_factor
        self.epsilon = 5.0e-3
        self.n_hidden_layers = n_hidden_layers

        self.x_decoder = XDecoder(num_genes=num_genes, z2_dim=latent_dim, 
                                  hidden_dims=sorted(calculate_hidden_dims(self.latent_dim, 2 * num_genes, self.n_hidden_layers)))
        self.z2l_encoder = Z2LEncoder(num_genes=num_genes, z2_dim=latent_dim, 
                                      hidden_dims=sorted(calculate_hidden_dims(self.num_genes, 2 * self.latent_dim + 2, self.n_hidden_layers), reverse=True))
        self.y_classifier = Classifier(z2_dim=latent_dim, hidden_dims=[50], num_cag_cat=num_cag_cat)

    def model(self, x, y=None):
        pyro.module("simple_scanvi", self)

        theta = pyro.param("inverse_dispersion",
                           10.0 * x.new_ones(self.num_genes),
                           constraint=constraints.positive)

        with pyro.plate("batch", x.shape[0]), poutine.scale(scale=self.scale_factor):

            # Priors
            z = pyro.sample("z", dist.Normal(0, x.new_ones(self.latent_dim)).to_event(1))
            l_scale = self.l_scale * x.new_ones(1)
            l = pyro.sample("l", dist.LogNormal(self.l_loc, l_scale).to_event(1))

            # Decode x
            gate_logits, mu = self.x_decoder(z)
            nb_logits = (l * mu + self.epsilon).log() - (theta + self.epsilon).log()

            x_dist = dist.ZeroInflatedNegativeBinomial(
                gate_logits=gate_logits,
                total_count=theta,
                logits=nb_logits,
            )
            pyro.sample("x", x_dist.to_event(1), obs=x)

    def guide(self, x, y=None):
        pyro.module("simple_scanvi", self)

        with pyro.plate("batch", x.shape[0]), poutine.scale(scale=self.scale_factor):
            z_loc, z_scale, l_loc, l_scale = self.z2l_encoder(x)

            z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))
            pyro.sample("l", dist.LogNormal(l_loc, l_scale).to_event(1))

            # Auxiliary classifier
            y_logits = self.y_classifier(z)
            y_dist = dist.OneHotCategorical(logits=y_logits)

            if y is not None:
                # classification loss (encourages q(y|z) to learn from labels)
                classification_loss = y_dist.log_prob(y)
                pyro.factor("classification_loss", -self.alpha * classification_loss,
                            has_rsample=False)
# %%
