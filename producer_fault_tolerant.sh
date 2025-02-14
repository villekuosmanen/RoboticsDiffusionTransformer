#!/bin/bash

# Trap Ctrl+C (SIGINT) and Ctrl+\ (SIGQUIT)
trap 'echo "Caught interrupt signal. Exiting..."; exit 1' INT QUIT

while true; do
    echo "Starting Python process..."
    python data/producer_lerobot.py --dataset_type finetune --fill_up

    # Check if the process was killed by a signal
    exit_code=$?
    echo "Process ended with exit code: $exit_code"
    
    # If we received an interrupt signal, break the loop
    if [ $exit_code -eq 130 ] || [ $exit_code -eq 131 ]; then
        echo "Received interrupt signal. Stopping..."
        break
    fi
    
    echo "Restarting in 5 seconds..."
    sleep 5
done