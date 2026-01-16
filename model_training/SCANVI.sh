#!/bin/bash

LATENT_DIM=(20 40 60)
HIDDEN=(1 2 3)
# UNSEEN=(0.0 0.1 0.25 0.5 0.75)
SEED=(0 1 2)

# Path to metrics file
METRICS_FILE="/pool01/projects/abante_lab/cag_propagation/scanvi_results/SCANVI_semisupervised_multitech_metrics.tsv"

for S in "${SEED[@]}"; do
    for HL in "${HIDDEN[@]}"; do
        for LD in "${LATENT_DIM[@]}"; do
        
            echo "Checking latent_dim=$LD, hidden layers=$HL, seed=$S"

            # Check if the metrics file already contains this combination
            # if grep -q -P "^$S\tv2_6k\t$HL\t$LD" "$METRICS_FILE"; then
            if grep -q -P "^$S\tdeep_dive_10x_v1_6k_test0.5_adata\t$HL\t$LD" "$METRICS_FILE"; then
                echo " -> Skipping: already exists in metrics file"
                continue
            fi

            echo " -> Running"
            # python scvi_cag.py --seed $S --latent_dim $LD --n_hidden_layers $HL --unseen_y_prop $UP --cuda -d "/pool01/projects/abante_lab/cag_propagation/deep_dive_adata_v2_6k.h5"
            python scvi_cag.py --seed $S --latent_dim $LD --n_hidden_layers $HL --multitech --cuda -d "/pool01/projects/abante_lab/cag_propagation/deep_dive_10x_v1_6k_test0.5_adata.h5"

        done
    done
done

echo "All HPO runs completed."