# Project_STRIKE
# Project STRIKE: Swarm Intelligence Simulation and Optimization

## Overview

Project STRIKE is a Python-based simulation environment designed to model and optimize the behavior of a multi-agent system, specifically a swarm of Unmanned Aerial Vehicles (UAVs). The primary objective is for the UAV swarm to locate and neutralize a High-Value Target (HVT) in a complex, dynamic environment containing various threats and obstacles.

The project combines several key concepts:

* **Swarm Intelligence (Boids Model):** UAVs use classic flocking behaviors (cohesion, alignment, separation) to maintain a coordinated group.

* **Particle Swarm Optimization (PSO):** The swarm leverages a PSO-like approach to collectively search for and converge on the target.

* **Genetic Algorithm (GA):** A GA is used to evolve and find the optimal behavioral weights for the swarm's agents, maximizing mission success.

* **Dynamic Environment:** The simulation includes static and dynamic obstacles, no-fly zones, and threat zones that drain UAV energy.

* **Data Analysis & Visualization:** The simulation logs key performance metrics and generates plots to visualize mission results and statistical analysis.

## Description

This project simulates the mission of a UAV swarm tasked with neutralizing a High-Value Target (HVT) in a challenging environment. The simulation is built on principles of swarm intelligence, allowing the UAVs to behave as a single, cohesive unit while also adapting to threats and obstacles.

The core of the project is a hybrid optimization approach that combines **Particle Swarm Optimization (PSO)** for target-finding behavior with a **Genetic Algorithm (GA)** to evolve the best possible behavioral weights for the swarm. This allows the system to learn and adapt to the environment, improving mission success over multiple trials.

The simulation environment is rich with dynamic elements, including static obstacles (walls, buildings), dynamic obstacles (other moving objects), and hostile zones that require the UAVs to manage their energy. The project includes extensive logging and visualization tools, providing a powerful platform for researchers and students to analyze swarm behavior and the effectiveness of different behavioral parameters.

## Getting Started

### Prerequisites

To run this project, you will need to have Python installed along with the following libraries. You can install them using `pip`:



pip install numpy pandas matplotlib seaborn


### Running the Simulation

Execute the main script from your terminal:

python STRIKE.py


The script will first run a genetic algorithm to find optimal weights, then perform a series of statistical trials, and finally generate plots to visualize the results. All generated data logs (`.csv` files) and plots (`.png` files) will be saved in the `logs` directory.

## Simulation Components

### Main Classes

* **`Simulation`**: The core class that orchestrates a single simulation trial. It manages the environment, the swarm, and the HVT.

* **`UAV`**: Represents a single drone. It has a state machine (`PATROL`, `RECON`, `ENGAGE`) and applies behavioral rules based on its state and proximity to other agents and the target.

* **`Swarm`**: Manages the collection of all UAVs, their collective behaviors, and overall swarm performance.

* **`HighValueTarget`**: The dynamic target that the swarm is tasked to neutralize.

* **`Environment`**: Manages all environmental elements, including static/dynamic obstacles, no-fly zones, and threat zones.

* **`GeneticAlgorithm`**: The optimization engine that evolves the behavioral weights for the UAVs.

* **`Visualizer`**: A utility class for plotting simulation paths and statistical analysis results.

### Configuration

Simulation parameters can be easily adjusted by modifying the constants at the top of the `STRIKE.py` file:

* `AREA_SIZE`: The size of the square mission area.

* `NUM_UAVS`: The total number of drones in the swarm.

* `NUM_STEPS`: The maximum number of simulation iterations.

* `GA_POPULATION_SIZE`, `GA_GENERATIONS`: Parameters for the genetic algorithm.

* `NUM_STATISTICAL_TRIALS`: The number of trials to run for statistical analysis.

## Analysis and Results

The `logs` directory will contain:

* **`trial_*.csv`**: Detailed logs of each simulation trial, including the position and state of every UAV and the HVT at each step.

* **`trial_*.png`**: Plots visualizing a single trial's paths.

* **`survivability_rate.png`**: A box plot showing the distribution of survivability rates across statistical trials.

* **`distance_distribution.png`**: A histogram showing the distribution of total distance traveled by the swarm.
