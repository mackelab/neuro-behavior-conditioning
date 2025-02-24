#!/bin/bash
cd ../

echo "running GLVM many seeds..."


#sleep 2s
python scripts/R_run_GLVM_mask_size.py --all_obs 0 --visualise 1 --offline 0 --beta 1 --epochs 1500 --exp "masked" --one_impute 0 --val_impute 1 --random_impute 0 --all 1 --seed 5000 &> tmp/masked_5000_size.txt

#sleep 2s
python scripts/R_run_GLVM_mask_size.py --all_obs 1  --visualise 1 --offline 0 --beta 1 --epochs 1500 --exp "all_obs"  --one_impute 0 --val_impute 1 --random_impute 0 --all 1 --seed 5000 &> tmp/all_obs_5000_size.txt



for i in {5001..5003}; do
    echo "Next seeds without visualisation...."
    echo "Run $i"
    sleep 2s
    #sleep 2s
    python scripts/R_run_GLVM_mask_size.py --all_obs 0 --visualise 0 --offline 0 --beta 1 --epochs 1500 --exp "masked" --one_impute 0 --val_impute 1 --random_impute 0 --all 1 --seed $i &> tmp/masked_${i}_size.txt
    
    #sleep 2s
    python scripts/R_run_GLVM_mask_size.py --all_obs 1 --visualise 0 --offline 0 --beta 1 --epochs 1500 --exp "all_obs"  --one_impute 0 --val_impute 1 --random_impute 0 --all 1 --seed $i &> tmp/all_obs_${i}_size.txt

    wait
    date
    echo "Done waiting... next seed"
done


