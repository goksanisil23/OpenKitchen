#!/bin/bash

OUTPUT_FILE="program_output.txt"

while true; do
    # Launch your program and redirect its output to the file
    ./env_sim "/home/s0001734/Downloads/racetrack-database/tracks/Spa.csv" > $OUTPUT_FILE &

    # Get its process ID
    PID=$!

    # Monitor the output file and look for "EPISODE 10"
    tail -n 0 -f $OUTPUT_FILE | while IFS= read -r line; do
        if [[ $line == *"EPISODE 5"* ]]; then
            # Kill the program once "EPISODE 10" is detected
            kill $PID
            break
        fi
    done

    # Optionally, wait for a short duration before restarting
    sleep 1
    cat $OUTPUT_FILE
    > $OUTPUT_FILE
done
