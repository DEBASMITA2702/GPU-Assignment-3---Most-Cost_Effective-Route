# GPU Assignment 3 – Parallel Evacuation Simulation

This repository contains the solution for **GPU Programming - Assignment 3**.  
The objective of this assignment is to **parallelize the evacuation logic** using CUDA, improving performance over the sequential version.

---

## Problem Statement
We are given a simulation where multiple people must evacuate through limited exits.  
The goal is to implement the evacuation logic efficiently using **parallel CUDA kernels**, while ensuring correctness and synchronization.

---

## Input Format
- The input describes the grid/building layout and people’s positions.  
- Each person tries to move towards the nearest exit.  
- Collisions and waiting conditions are handled as per the given rules.

---

## Output Format
- Final evacuation time.  
- Updated grid showing how/when all individuals exited. 

---

## Example
### Input  
Grid: 5x5  
Exits: (0,4), (4,4)  
People: (2,2), (3,1), (1,3)  

### Output
Minimum evacuation time: 6

---

## How to Run
1. Compile the code:
   ```bash
   nvcc Effective_Route.cu -o evac
   
Run with input file:
```
./evac < input.txt > output.txt
```
