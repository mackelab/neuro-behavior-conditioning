#!/bin/bash
cd ..

for s in {104..110}; do
  echo "Seed  $s"
  for i in {20..40..20}; do
      echo "Run $i"
      sleep 2s
      python scripts/R_run_monkey_latent_neuro.py --offline 0 --cross_loss 0 --latent_size $i --seed $s > "tmp/revisions_run_monkey_seed_${s}_latent_${i}.txt"
      date
      echo "Done waiting... next seed $s and latent $i"
  done # done latent dim
done # done seeds

echo " "
echo "Done training..."
echo " "
