#!/bin/bash
cd ../

tmp_dir="./tmp/"

# check if the tmp directory exists and create it if it does not
if [ ! -d "$tmp_dir" ]; then
    # create it if it does not exist
    mkdir "$tmp_dir"
    echo "Temporary directory created at: $tmp_dir"
else
    echo "Temporary directory already exists at: $tmp_dir"
fi

echo "running GLVM many seeds..."


#sleep 2s
python scripts/run_GLVM.py --all_obs 0 --visualise 1 --offline 0 --beta 1 --epochs 1500 --exp "masked" --one_impute 0 --val_impute 1 --random_impute 0 --all 1 --seed 5000 &> tmp/masked_5000.txt

#sleep 2s
python scripts/run_GLVM.py --all_obs 1  --visualise 1 --offline 0 --beta 1 --epochs 1500 --exp "all_obs"  --one_impute 0 --val_impute 1 --random_impute 0 --all 1 --seed 5000 &> tmp/all_obs_5000.txt



for i in {5001..5019}; do
    echo "Next seeds without visualisation...."
    echo "Run $i"
    sleep 2s
    #sleep 2s
    python scripts/run_GLVM.py --all_obs 0 --visualise 0 --offline 0 --beta 1 --epochs 1500 --exp "masked" --one_impute 0 --val_impute 1 --random_impute 0 --all 1 --seed $i &> tmp/masked_${i}.txt
    
    #sleep 2s
    python scripts/run_GLVM.py --all_obs 1 --visualise 0 --offline 0 --beta 1 --epochs 1500 --exp "all_obs"  --one_impute 0 --val_impute 1 --random_impute 0 --all 1 --seed $i &> tmp/all_obs_${i}.txt

    wait
    date
    echo "Done waiting... next seed"
done


