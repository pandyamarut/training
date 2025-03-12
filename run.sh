#!/bin/bash

# Base directory for logs
mkdir -p training_speed_logs

# Define parameter ranges to test
SEQ_LENGTHS=(2048 4096 8192)
MICRO_BATCHES=(2 4 8 16)
GRAD_ACCUM=(1 2 4)

# Base config file
BASE_CONFIG="config.yaml"

for SEQ in "${SEQ_LENGTHS[@]}"; do
  for MB in "${MICRO_BATCHES[@]}"; do
    for GA in "${GRAD_ACCUM[@]}"; do
      # Skip very memory-intensive combinations
      if [ $SEQ -eq 8192 ] && [ $MB -gt 4 ]; then
        echo "Skipping potential OOM: seq=$SEQ, mb=$MB, ga=$GA"
        continue
      fi
      
      # Create test name
      TEST_NAME="seq${SEQ}_mb${MB}_ga${GA}"
      CONFIG_FILE="speed_test_${TEST_NAME}.yaml"
      LOG_FILE="training_speed_logs/${TEST_NAME}.log"
      
      echo "Testing configuration: $TEST_NAME"
      
      # Create modified config
      sed -e "s/^sequence_len:.*/sequence_len: ${SEQ}/" \
          -e "s/^micro_batch_size:.*/micro_batch_size: ${MB}/" \
          -e "s/^gradient_accumulation_steps:.*/gradient_accumulation_steps: ${GA}/" \
          -e "s/^num_epochs:.*/num_epochs: 0.01/" \
          $BASE_CONFIG > $CONFIG_FILE
      
      # Run short training job (just enough to measure throughput)
      echo "Starting training for $TEST_NAME"
      torchrun --nproc_per_node=8 \
               --nnodes=2 \
               --node_rank=0 \
               --rdzv_id=123 \
               --rdzv_backend=static \
               --rdzv_endpoint=10.65.0.2:29400 \
               -m axolotl.cli.train $CONFIG_FILE 2>&1 | tee $LOG_FILE
      
      # Extract training speed metrics
      SAMPLES_PER_SEC=$(grep "samples/second" $LOG_FILE | tail -1 | awk '{print $NF}')
      
      echo "$TEST_NAME: $SAMPLES_PER_SEC samples/sec" >> training_speed_logs/summary.txt
      echo "-------------------------------"
    done
  done
done

# Generate simple report
echo "Configuration,Samples_Per_Second" > training_speed_logs/report.csv
grep -E "seq.*:" training_speed_logs/summary.txt | sed 's/: /,/g' >> training_speed_logs/report.csv

echo "Speed tests complete. Results in training_speed_logs/report.csv"
