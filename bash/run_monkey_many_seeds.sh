#!/bin/bash
cd ..

for s in {100..115}; do
  echo "Seed  $s"
  for i in {40..40..5}; do
      echo "Run $i"
      sleep 2s
      python scripts/run_monkey.py --offline 0 --cross_loss 0 --latent_size $i --seed $s > "tmp/run_monkey_seed_${s}_latent_${i}.txt"
      date
      echo "Done waiting... next seed $s and latent $i"
  done # done latent dim
done # done seed

echo " "
echo "Done training..."
echo " "
