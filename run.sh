#!/bin/bash

# GPU Assignment 3 Test Runner
# ----------------------------
# Runs all test cases and logs results
# Directory structure:
# - current_dir/
#   ├── input/         (Test case files)
#   ├── output/        (Expected outputs)
#   ├── submit/        (Your CUDA code)
#   ├── log.txt        (Execution log)
#   └── timing.txt     (Performance metrics)

# Initialize paths
current_dir=$(pwd)
input_dir="$current_dir/input"
output_dir="$current_dir/output"
submit_dir="$current_dir/submit"

# Create log files
log_file="$current_dir/log.txt"
timing_file="$current_dir/timing.txt"
touch $log_file $timing_file

# Compilation
echo "-------== Compiling CUDA Code ==--------" | tee -a $log_file
cd $submit_dir
ROLLNO=$(ls *.cu | tail -1 | cut -d'.' -f1)
cp "${ROLLNO}.cu" "main.cu"

# Compile with NVIDIA compiler
#nvcc -O3 -arch=sm_70 -Xcompiler -Wall -Werror main.cu -o main.out 2>> $log_file

# Compile the CUDA file
bash compile.sh

if [ ! -f "main.out" ]; then
    echo "Compilation failed! Check $log_file" | tee -a $log_file
    exit 1
fi

# Test execution
echo -e "\n=== Running Test Cases ===" | tee -a $log_file
total=0
passed=0

for testcase in $input_dir/*; do
    ((total++))
    filename=$(basename $testcase)
    
    echo "Running $filename..." | tee -a $log_file
    
    # Run with timing (GPU and CPU time)
    /usr/bin/time -f "%e sec" ./main.out < $testcase > cuda.out 2> time.log
    
    # Get results
    actual=$(cat cuda.out)
    expected=$(cat $output_dir/$filename)
    duration=$(cat time.log)
    
    # Verify output
    if [ "$actual" = "$expected" ]; then
        echo "[PASS] $filename ($duration)" | tee -a $log_file
        ((passed++))
    else
        echo "[FAIL] $filename | Expected: $expected, Got: $actual" | tee -a $log_file
    fi
    
    # Save timing data
    echo "$filename $duration" >> $timing_file
done

# Cleanup
rm -f cuda.out time.log

# Final report
echo -e "\n=== Test Summary ===" | tee -a $log_file
echo "Passed $passed/$total tests" | tee -a $log_file
echo "Timing data saved to $timing_file" | tee -a $log_file
