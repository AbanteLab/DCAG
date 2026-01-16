#!/bin/bash

ALPHA_Y=(2.0 5.0 10.0)
ALPHA_CT=(0.1 1.0 5.0 10.0)
LATENT_DIM_Y=(20 40 60)
LATENT_DIM_CT=(20 40)
LAMBDA_REG=(0.001 0.01 0.1)
# UNSEEN=(0.0 0.1 0.25 0.5 0.75 0.9)
SEED=(1 2 3)

# Path to metrics file
METRICS_FILE="/pool01/projects/abante_lab/cag_propagation/scanvi_results/DCAG_ortho_supervised_metrics.tsv"

# for UP in "${UNSEEN[@]}"; do
    for LD_Y in "${LATENT_DIM_Y[@]}"; do
        for LD_CT in "${LATENT_DIM_CT[@]}"; do
            for A_Y in "${ALPHA_Y[@]}"; do
                for A_CT in "${ALPHA_CT[@]}"; do
                    for LR in "${LAMBDA_REG[@]}"; do

                        echo " -> Running"
                        # python pyro_models_training.py --alpha_y $A_Y --alpha_ct $A_CT --latent_dim_y $LD_Y --latent_dim_ct $LD_CT --lambda_reg $LR --unseen_y_prop $UP --model DCAG --cuda -d "/pool01/projects/abante_lab/cag_propagation/deep_dive_adata_v2_6k.h5"
                        python pyro_models_training.py --alpha_y $A_Y --alpha_ct $A_CT --latent_dim_y $LD_Y --latent_dim_ct $LD_CT --lambda_reg $LR --model DCAG --cuda -d "/pool01/projects/abante_lab/cag_propagation/deep_dive_adata_v1_6k.h5"
                    
                    done    
                done
            done
        done
    done
# done

wait
echo "All HPO runs completed."