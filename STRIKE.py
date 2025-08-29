import random
import math
import csv
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

# -------------------- Simulation Constants and Configuration --------------------
# Mission Area and Swarm Size
AREA_SIZE = 200 # Square area in meters
NUM_UAVS = 50
NUM_STEPS = 1000 # Simulation iterations
NEIGHBOR_RADIUS = 25.0
SENSING_RADIUS = 50.0 # Radius for a UAV to "sense" a target
ENGAGEMENT_RADIUS = 5.0 # Radius for a UAV to "engage" the target
MAX_SPEED = 2.0
UAV_ENERGY_CONSUMPTION = 0.5 # Energy consumed per step
NUM_LEADERS = 5 # Number of designated leaders in the swarm

# Genetic Algorithm Parameters
GA_POPULATION_SIZE = 20
GA_GENERATIONS = 10
GA_MUTATION_RATE = 0.1
GA_CROSSOVER_RATE = 0.7
WEIGHT_LIMIT = 5.0 # Max value for behavioral weights

# Statistical Analysis Parameters
NUM_STATISTICAL_TRIALS = 5

# Mission States (for each UAV)
STATE_PATROL = 0
STATE_RECON = 1
STATE_ENGAGE = 2
STATE_RALLY = 3

# Logging
ENABLE_CSV_LOGGING = True
OUTPUT_DIR = "logs"

# -------------------- Utility Functions --------------------
def distance(pos1, pos2):
    """Calculates the Euclidean distance between two points."""
    return np.linalg.norm(pos1 - pos2)

def normalize(vector):
    """Normalizes a 2D numpy vector."""
    norm = np.linalg.norm(vector)
    if norm > 0:
        return vector / norm
    return np.zeros(2)

# -------------------- Environment Entities --------------------
class HighValueTarget:
    """Represents a High-Value Target (HVT) that can be neutralized."""
    def __init__(self, position, initial_health=100, speed=0.5):
        self.position = np.array(position, dtype=float)
        self.health = initial_health
        self.is_neutralized = False
        self.speed = speed
        self.velocity = self._random_velocity()
        self.path = [list(position)] # Track the HVT's path

    def _random_velocity(self):
        """Generates a random velocity vector."""
        angle = random.uniform(0, 2 * math.pi)
        return np.array([self.speed * math.cos(angle), self.speed * math.sin(angle)])

    def update(self):
        """Moves the target, changing direction periodically and ensuring it stays within bounds."""
        if not self.is_neutralized:
            if random.random() < 0.01:
                self.velocity = self._random_velocity()
            
            # Update position
            self.position += self.velocity
            
            # Boundary bouncing
            if self.position[0] < 0 or self.position[0] > AREA_SIZE:
                self.velocity[0] *= -1
            if self.position[1] < 0 or self.position[1] > AREA_SIZE:
                self.velocity[1] *= -1
            
            # Append to path for visualization
            self.path.append(list(self.position))

class Obstacle:
    """Represents a static circular obstacle."""
    def __init__(self, center, radius):
        self.center = np.array(center)
        self.radius = radius

class DynamicObstacle(Obstacle):
    """Represents a moving circular obstacle."""
    def __init__(self, center, radius, velocity):
        super().__init__(center, radius)
        self.center = np.array(center, dtype=float)
        self.velocity = np.array(velocity)
    
    def update(self, area_size):
        """Updates the obstacle's position with boundary bouncing."""
        self.center += self.velocity
        if not (0 < self.center[0] < area_size): self.velocity[0] *= -1
        if not (0 < self.center[1] < area_size): self.velocity[1] *= -1

class NoFlyZone:
    """Represents a rectangular no-fly zone."""
    def __init__(self, x_range, y_range):
        self.x_range = x_range
        self.y_range = y_range

class ThreatZone:
    """Represents an area that negatively affects UAVs."""
    def __init__(self, x_range, y_range, energy_drain_rate=0.1):
        self.x_range = x_range
        self.y_range = y_range
        self.energy_drain_rate = energy_drain_rate
    
    def is_inside(self, position):
        """Checks if a position is inside the threat zone."""
        return (self.x_range[0] < position[0] < self.x_range[1] and
                self.y_range[0] < position[1] < self.y_range[1])

class Environment:
    """Manages all environmental elements (obstacles, no-fly zones)."""
    def __init__(self, area_size, static_obstacles, dynamic_obstacles, no_fly_zones, threat_zones):
        self.area_size = area_size
        self.static_obstacles = static_obstacles
        self.dynamic_obstacles = dynamic_obstacles
        self.no_fly_zones = no_fly_zones
        self.threat_zones = threat_zones

    def update_dynamics(self):
        """Updates the position of all dynamic elements in the environment."""
        for obs in self.dynamic_obstacles:
            obs.update(self.area_size)

# -------------------- UAV and Swarm Classes --------------------
class UAV:
    """Represents a single Unmanned Aerial Vehicle (drone) with a state machine."""
    def __init__(self, uav_id, position, is_leader=False):
        self.uav_id = uav_id
        self.position = np.array(position, dtype=float)
        self.velocity = np.random.uniform(-1, 1, size=2)
        self.state = STATE_PATROL
        self.is_active = True
        self.is_leader = is_leader
        self.path = [list(position)]
        self.energy = 100.0
        self.target_locked = False
        self.target_id = -1
        self.pbest_position = self.position.copy()
        self.pbest_fitness = float('inf')
        self.total_distance = 0.0
        self.collisions = 0
        self.time_to_target_lock = -1

    def update_state(self, hvt, neighbors, obstacles, threat_zones):
        """Updates the UAV's state based on its environment."""
        # Check for mission-ending conditions
        if not self.is_active or self.energy <= 0:
            self.is_active = False
            return
        
        # Sense the target
        dist_to_hvt = distance(self.position, hvt.position)
        if dist_to_hvt < SENSING_RADIUS and not hvt.is_neutralized and not self.target_locked:
            self.state = STATE_RECON
            self.target_locked = True
            # Log time to target lock if it's the first lock
            if self.time_to_target_lock == -1:
                self.time_to_target_lock = time.time()
        
        # Check for engagement range
        if dist_to_hvt < ENGAGEMENT_RADIUS and self.target_locked:
            self.state = STATE_ENGAGE
        elif self.target_locked:
            self.state = STATE_RECON
        else:
            self.state = STATE_PATROL

    def apply_behavior(self, hvt, swarm, environment, weights):
        """Calculates and applies the new velocity based on combined behaviors."""
        if not self.is_active:
            return

        # Check for threat zones and apply energy drain if inside
        for zone in environment.threat_zones:
            if zone.is_inside(self.position):
                self.energy -= zone.energy_drain_rate
        
        # Initialize forces
        force = np.zeros(2)

        # Apply behaviors based on state
        if self.state == STATE_PATROL:
            force += self.boids_cohesion(swarm, weights['cohesion'])
            force += self.boids_alignment(swarm, weights['alignment'])
            force += self.boids_separation(swarm, weights['separation'])
            force += self.obstacle_avoidance(environment, weights['obstacle_avoidance'])
            force += self.no_fly_zone_avoidance(environment, weights['no_fly_zone'])
            force += self.threat_zone_avoidance(environment, weights['threat_zone_avoidance'])
            force += self.wander_behavior(weights['wander'])

        elif self.state == STATE_RECON:
            force += self.pso_target_attraction(hvt, weights['pso_attraction'])
            force += self.boids_cohesion(swarm, weights['cohesion'])
            force += self.boids_alignment(swarm, weights['alignment'])
            force += self.boids_separation(swarm, weights['separation'])
            force += self.obstacle_avoidance(environment, weights['obstacle_avoidance'])
            force += self.no_fly_zone_avoidance(environment, weights['no_fly_zone'])
            force += self.threat_zone_avoidance(environment, weights['threat_zone_avoidance'])

        elif self.state == STATE_ENGAGE:
            # Direct attack on the target
            target_direction = hvt.position - self.position
            force += normalize(target_direction) * weights['engage_strength']
            self.kinetic_strike(hvt, weights['kinetic_strike'])
            force += self.no_fly_zone_avoidance(environment, weights['no_fly_zone'])
            force += self.threat_zone_avoidance(environment, weights['threat_zone_avoidance'])

        # Update velocity and handle speed limits
        self.velocity += force
        self.velocity = normalize(self.velocity) * min(np.linalg.norm(self.velocity), MAX_SPEED)
        
        # Update position and energy
        old_pos = self.position.copy()
        self.position += self.velocity
        self.total_distance += distance(old_pos, self.position)
        self.energy -= UAV_ENERGY_CONSUMPTION
        
        # Log path for visualization
        self.path.append(list(self.position))
        
        # Update pbest for PSO
        current_fitness = distance(self.position, hvt.position)
        if current_fitness < self.pbest_fitness:
            self.pbest_fitness = current_fitness
            self.pbest_position = self.position.copy()

    def boids_cohesion(self, swarm, weight):
        """Rule 1: Steer towards the average position of local flockmates."""
        neighbors = [uav.position for uav in swarm if uav.is_active and distance(self.position, uav.position) < NEIGHBOR_RADIUS and uav != self]
        if not neighbors:
            return np.zeros(2)
        
        center_of_mass = np.mean(neighbors, axis=0)
        return normalize(center_of_mass - self.position) * weight

    def boids_alignment(self, swarm, weight):
        """Rule 2: Steer towards the average heading of local flockmates."""
        neighbors_vel = [uav.velocity for uav in swarm if uav.is_active and distance(self.position, uav.position) < NEIGHBOR_RADIUS and uav != self]
        if not neighbors_vel:
            return np.zeros(2)
        
        avg_velocity = np.mean(neighbors_vel, axis=0)
        return normalize(avg_velocity) * weight

    def boids_separation(self, swarm, weight):
        """Rule 3: Steer to avoid crowding local flockmates."""
        force = np.zeros(2)
        for uav in swarm:
            if uav.is_active and uav != self:
                dist = distance(self.position, uav.position)
                if 0 < dist < 5: # Small avoidance radius
                    force -= normalize(uav.position - self.position) / dist
        return force * weight

    def pso_target_attraction(self, hvt, weight):
        """Particle Swarm Optimization (PSO) based attraction to the target."""
        cognitive_component = normalize(self.pbest_position - self.position)
        social_component = normalize(hvt.position - self.position) # Simplified gbest to HVT position
        
        pso_force = (cognitive_component * random.uniform(0, 2)) + (social_component * random.uniform(0, 2))
        return normalize(pso_force) * weight

    def obstacle_avoidance(self, environment, weight):
        """Avoids static and dynamic obstacles."""
        force = np.zeros(2)
        all_obstacles = environment.static_obstacles + environment.dynamic_obstacles
        for obs in all_obstacles:
            dist = distance(self.position, obs.center)
            if dist < obs.radius + 10: # Add a buffer
                force += normalize(self.position - obs.center) / (dist + 1e-6) # Add epsilon to avoid division by zero
        return force * weight

    def no_fly_zone_avoidance(self, environment, weight):
        """Avoids rectangular no-fly zones more proactively."""
        force = np.zeros(2)
        AVOIDANCE_BUFFER = 20 # Steer away from the boundary from a distance
        for zone in environment.no_fly_zones:
            # Check if the UAV is near or inside the no-fly zone
            if (zone.x_range[0] - AVOIDANCE_BUFFER < self.position[0] < zone.x_range[1] + AVOIDANCE_BUFFER and
                zone.y_range[0] - AVOIDANCE_BUFFER < self.position[1] < zone.y_range[1] + AVOIDANCE_BUFFER):
                
                # Calculate vector to the nearest edge
                nearest_x = max(zone.x_range[0], min(self.position[0], zone.x_range[1]))
                nearest_y = max(zone.y_range[0], min(self.position[1], zone.y_range[1]))
                
                direction_to_center = np.array([nearest_x, nearest_y]) - self.position
                dist_to_boundary = np.linalg.norm(direction_to_center)
                
                if dist_to_boundary > 0:
                    force -= normalize(direction_to_center) * (1 / dist_to_boundary)
        return force * weight

    def threat_zone_avoidance(self, environment, weight):
        """Adds a repulsive force to avoid threat zones."""
        force = np.zeros(2)
        AVOIDANCE_BUFFER = 10
        for zone in environment.threat_zones:
            # Find the closest point on the zone's boundary
            nearest_x = max(zone.x_range[0], min(self.position[0], zone.x_range[1]))
            nearest_y = max(zone.y_range[0], min(self.position[1], zone.y_range[1]))
            
            boundary_point = np.array([nearest_x, nearest_y])
            dist_to_boundary = distance(self.position, boundary_point)
            
            # Apply repulsive force if near the zone
            if dist_to_boundary < AVOIDANCE_BUFFER:
                force += normalize(self.position - boundary_point) / (dist_to_boundary + 1e-6)
        return force * weight

    def kinetic_strike(self, hvt, strength):
        """Reduces HVT health."""
        if not hvt.is_neutralized:
            hvt.health -= strength
            if hvt.health <= 0:
                hvt.health = 0
                hvt.is_neutralized = True
                print(f"HVT neutralized by UAV {self.uav_id}!")

    def wander_behavior(self, weight):
        """Simple random wandering for patrol state."""
        angle = random.uniform(-math.pi/4, math.pi/4) # Small random change
        rotation_matrix = np.array([[math.cos(angle), -math.sin(angle)],
                                    [math.sin(angle), math.cos(angle)]])
        self.velocity = np.dot(rotation_matrix, self.velocity)
        return normalize(self.velocity) * weight

class Swarm:
    """Manages the collection of UAVs and their global behaviors."""
    def __init__(self, num_uavs, area_size):
        self.uavs = []
        leader_indices = random.sample(range(num_uavs), NUM_LEADERS)
        for i in range(num_uavs):
            position = np.random.uniform(0, area_size, size=2)
            is_leader = i in leader_indices
            self.uavs.append(UAV(i, position, is_leader))
        
    def update_all_uavs(self, hvt, environment, weights):
        """Updates the state and applies behavior for all active UAVs."""
        active_uavs = [uav for uav in self.uavs if uav.is_active]
        for uav in active_uavs:
            uav.update_state(hvt, active_uavs, environment.static_obstacles, environment.threat_zones)
            uav.apply_behavior(hvt, active_uavs, environment, weights)
            
    def get_performance_metrics(self):
        """Calculates and returns a dictionary of key performance metrics."""
        total_distance_traveled = sum(uav.total_distance for uav in self.uavs)
        total_collisions = sum(uav.collisions for uav in self.uavs)
        survivability_rate = sum(1 for uav in self.uavs if uav.is_active) / len(self.uavs)
        
        # Calculate time to target engagement
        target_locked_times = [uav.time_to_target_lock for uav in self.uavs if uav.time_to_target_lock != -1]
        avg_lock_time = np.mean(target_locked_times) if target_locked_times else -1
        
        return {
            'total_distance_traveled': total_distance_traveled,
            'total_collisions': total_collisions,
            'survivability_rate': survivability_rate,
            'avg_time_to_target_lock': avg_lock_time
        }

# -------------------- Main Simulation Class --------------------
class Simulation:
    """The main class to run a full simulation trial."""
    def __init__(self, num_uavs, area_size, hvt, obstacles, dynamic_obstacles, no_fly_zones, threat_zones, weights):
        self.area_size = area_size
        self.environment = Environment(area_size, obstacles, dynamic_obstacles, no_fly_zones, threat_zones)
        self.swarm = Swarm(num_uavs, area_size)
        self.hvt = hvt
        self.weights = weights
        self.log_data = []

    def run_trial(self, trial_id, log_to_csv=True):
        """Runs a single simulation trial."""
        print(f"--- Running Trial {trial_id} ---")
        start_time = time.time()
        self.log_data = []

        for step in range(NUM_STEPS):
            if self.hvt.is_neutralized:
                print(f"HVT neutralized at step {step} in trial {trial_id}.")
                break

            self.environment.update_dynamics()
            self.hvt.update()
            self.swarm.update_all_uavs(self.hvt, self.environment, self.weights)
            
            # Log data for this step
            step_data = {
                'step': step,
                'hvt_x': self.hvt.position[0],
                'hvt_y': self.hvt.position[1],
                'hvt_health': self.hvt.health,
                'hvt_neutralized': self.hvt.is_neutralized,
                'num_active_uavs': sum(1 for uav in self.swarm.uavs if uav.is_active)
            }
            
            # Add UAV positions and states
            for uav in self.swarm.uavs:
                step_data[f'uav_{uav.uav_id}_x'] = uav.position[0]
                step_data[f'uav_{uav.uav_id}_y'] = uav.position[1]
                step_data[f'uav_{uav.uav_id}_state'] = uav.state
                step_data[f'uav_{uav.uav_id}_active'] = uav.is_active
            
            self.log_data.append(step_data)
        
        end_time = time.time()
        duration = end_time - start_time
        
        metrics = self.swarm.get_performance_metrics()
        metrics['hvt_neutralized'] = self.hvt.is_neutralized
        metrics['steps_to_neutralize'] = step if self.hvt.is_neutralized else -1
        metrics['simulation_time'] = duration
        
        if log_to_csv:
            self.save_log_to_csv(trial_id)
        
        return metrics

    def save_log_to_csv(self, trial_id):
        """Saves the simulation log to a CSV file."""
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        
        file_path = os.path.join(OUTPUT_DIR, f'trial_{trial_id}_log.csv')
        df = pd.DataFrame(self.log_data)
        df.to_csv(file_path, index=False)
        print(f"Log for trial {trial_id} saved to {file_path}")

# -------------------- Optimization and Analysis --------------------
class GeneticAlgorithm:
    """Manages the evolutionary process to find optimal swarm weights."""
    def __init__(self, population_size, num_generations, mutation_rate, crossover_rate, hvt_initial_pos, obstacles, dynamic_obstacles, no_fly_zones, threat_zones):
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = self.initialize_population()
        self.hvt_initial_pos = hvt_initial_pos
        self.obstacles = obstacles
        self.dynamic_obstacles = dynamic_obstacles
        self.no_fly_zones = no_fly_zones
        self.threat_zones = threat_zones
        self.best_individual = None
        self.best_fitness = float('inf')

    def initialize_population(self):
        """Creates an initial population of random weight sets."""
        population = []
        for i in range(self.population_size):
            weights = {
                'cohesion': random.uniform(0, WEIGHT_LIMIT),
                'alignment': random.uniform(0, WEIGHT_LIMIT),
                'separation': random.uniform(0, WEIGHT_LIMIT),
                'pso_attraction': random.uniform(0, WEIGHT_LIMIT),
                'obstacle_avoidance': random.uniform(0, WEIGHT_LIMIT),
                'no_fly_zone': random.uniform(0, WEIGHT_LIMIT),
                'engage_strength': random.uniform(0, WEIGHT_LIMIT),
                'kinetic_strike': random.uniform(0, WEIGHT_LIMIT),
                'wander': random.uniform(0, WEIGHT_LIMIT),
                'threat_zone_avoidance': random.uniform(0, WEIGHT_LIMIT) # New weight for threat zone
            }
            population.append(weights)
        return population

    def run_generation(self, generation_num):
        """Runs one generation of the GA, including fitness evaluation, selection, crossover, and mutation."""
        print(f"\n  -- Evaluating Generation {generation_num}/{self.num_generations} --")
        
        fitnesses = [self.evaluate_fitness(weights) for weights in self.population]
        
        # Update best individual
        for i, fitness in enumerate(fitnesses):
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_individual = self.population[i]
        
        print(f"  Best fitness in this generation: {min(fitnesses):.2f}")
        print(f"  Overall best fitness so far: {self.best_fitness:.2f}")

        # Selection: Tournament selection
        new_population = []
        for _ in range(self.population_size):
            parent1 = random.choice(self.population)
            parent2 = random.choice(self.population)
            if self.evaluate_fitness(parent1) < self.evaluate_fitness(parent2):
                new_population.append(parent1)
            else:
                new_population.append(parent2)
        print("  Selection complete.")

        # Crossover
        offspring_population = []
        for i in range(0, self.population_size, 2):
            p1 = new_population[i]
            p2 = new_population[i+1]
            if random.random() < self.crossover_rate:
                c1, c2 = self.crossover(p1, p2)
                offspring_population.extend([c1, c2])
            else:
                offspring_population.extend([p1, p2])
        print("  Crossover complete.")

        # Mutation
        for i in range(len(offspring_population)):
            if random.random() < self.mutation_rate:
                offspring_population[i] = self.mutate(offspring_population[i])
        print("  Mutation complete.")

        self.population = offspring_population[:self.population_size]

    def evaluate_fitness(self, weights):
        """Evaluates the fitness of a single set of weights based on simulation."""
        # Lower fitness is better. We want to neutralize the HVT quickly with low energy cost.
        hvt = HighValueTarget(self.hvt_initial_pos)
        sim = Simulation(NUM_UAVS, AREA_SIZE, hvt, self.obstacles, self.dynamic_obstacles, self.no_fly_zones, self.threat_zones, weights)
        metrics = sim.run_trial(trial_id=0, log_to_csv=False)
        
        # Fitness function: Reward neutralizing the target, penalize high distance and low survivability.
        # This is a simplified fitness function. A more complex one would be better.
        fitness = 0
        if not metrics['hvt_neutralized']:
            fitness += 1000 # High penalty for not neutralizing
        fitness += metrics['total_distance_traveled'] / 100 # Penalize long paths
        fitness -= metrics['survivability_rate'] * 100 # Reward survivability
        
        return fitness

    def crossover(self, p1, p2):
        """Performs a single-point crossover on two weight sets."""
        child1 = p1.copy()
        child2 = p2.copy()
        keys = list(p1.keys())
        crossover_point = random.randint(1, len(keys) - 1)
        for i in range(crossover_point, len(keys)):
            key = keys[i]
            child1[key], child2[key] = child2[key], child1[key]
        return child1, child2

    def mutate(self, individual):
        """Randomly mutates a weight in the individual."""
        key_to_mutate = random.choice(list(individual.keys()))
        individual[key_to_mutate] = random.uniform(0, WEIGHT_LIMIT)
        return individual

    def evolve(self):
        """Runs the main GA loop."""
        print("--- Genetic Algorithm Starting with a population of", self.population_size, "---")
        for gen in range(self.num_generations):
            self.run_generation(gen + 1)
        
        print("\n--- GA Complete ---")
        print("Final Optimal Weights Found:")
        print(self.best_individual)
        return self.best_individual

# -------------------- Visualization and Analysis --------------------
class Visualizer:
    """Handles plotting and visualization of simulation results."""
    def plot_analysis_results(self, output_dir):
        """Generates statistical plots from CSV data."""
        all_metrics = []
        for file in os.listdir(output_dir):
            if file.endswith("_metrics.csv"):
                df = pd.read_csv(os.path.join(output_dir, file))
                all_metrics.append(df)

        if not all_metrics:
            print("No statistical data found to plot. Run statistical analysis first.")
            return

        combined_df = pd.concat(all_metrics, ignore_index=True)

        print("Generating Analysis Plots...")

        # Plot 1: Survivability Rate over trials
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=combined_df['survivability_rate'])
        plt.title('Survivability Rate Across Statistical Trials')
        plt.ylabel('Survivability Rate')
        plt.xlabel('Statistical Trials')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "survivability_rate.png"))
        plt.show()

        # Plot 2: Total Distance Traveled Distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(combined_df['total_distance_traveled'], kde=True)
        plt.title('Distribution of Total Distance Traveled')
        plt.xlabel('Total Distance (m)')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "distance_distribution.png"))
        plt.show()

    def plot_trial_path(self, log_df, hvt_path, obstacles, dynamic_obstacles, no_fly_zones, threat_zones, output_dir):
        """Generates a visualization of a single simulation trial."""
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_xlim(0, AREA_SIZE)
        ax.set_ylim(0, AREA_SIZE)
        ax.set_title('Swarm Simulation: Drone Paths and Mission Progress')
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_aspect('equal')

        # Plot No-Fly Zones
        for zone in no_fly_zones:
            rect = patches.Rectangle((zone.x_range[0], zone.y_range[0]),
                                     zone.x_range[1] - zone.x_range[0],
                                     zone.y_range[1] - zone.y_range[0],
                                     linewidth=1, edgecolor='red', facecolor='salmon', alpha=0.3, label='No-Fly Zone')
            ax.add_patch(rect)
            
        # Plot Threat Zones
        for zone in threat_zones:
            rect = patches.Rectangle((zone.x_range[0], zone.y_range[0]),
                                     zone.x_range[1] - zone.x_range[0],
                                     zone.y_range[1] - zone.y_range[0],
                                     linewidth=1, edgecolor='purple', facecolor='purple', alpha=0.2, label='Threat Zone')
            ax.add_patch(rect)

        # Plot Static Obstacles
        for obs in obstacles:
            circle = patches.Circle(obs.center, obs.radius, color='gray', alpha=0.7, label='Static Obstacle')
            ax.add_patch(circle)

        # Plot Dynamic Obstacles
        for obs in dynamic_obstacles:
            circle = patches.Circle(obs.center, obs.radius, color='darkgray', alpha=0.9, label='Dynamic Obstacle')
            ax.add_patch(circle)

        # Plot UAV Paths
        num_uavs = (log_df.columns.str.contains('uav_.*_x')).sum()
        for i in range(num_uavs):
            uav_path = log_df[[f'uav_{i}_x', f'uav_{i}_y']].values
            ax.plot(uav_path[:, 0], uav_path[:, 1], color='blue', alpha=0.3, linewidth=0.5)

        # Plot HVT Path
        hvt_path_arr = np.array(hvt_path)
        ax.plot(hvt_path_arr[:, 0], hvt_path_arr[:, 1], color='red', linewidth=2, label='HVT Path')
        ax.scatter(hvt_path_arr[-1, 0], hvt_path_arr[-1, 1], color='red', marker='X', s=100, label='Final HVT Position')
        
        # Re-plot the last position of the active UAVs for clarity
        last_step_df = log_df.iloc[-1]
        for i in range(num_uavs):
            if last_step_df[f'uav_{i}_active']:
                ax.scatter(last_step_df[f'uav_{i}_x'], last_step_df[f'uav_{i}_y'], color='blue', s=20, alpha=0.8, edgecolors='black')
        
        # Create a legend, avoiding duplicate labels
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        ax.legend(unique_labels.values(), unique_labels.keys())

        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "simulation_path.png"))
        plt.show()

# -------------------- Main Execution Block --------------------
def main():
    """Main function to run the full Project STRIKE pipeline."""
    # --- Step 1: Initialize Environment and Entities ---
    static_obstacles = [Obstacle((50, 50), 10), Obstacle((150, 150), 15)]
    dynamic_obstacles = [DynamicObstacle((100, 100), 5, (0.5, 0.5)), DynamicObstacle((10, 190), 8, (-0.2, 0.3))]
    no_fly_zones = [NoFlyZone((70, 90), (120, 140))]
    threat_zones = [ThreatZone((20, 50), (170, 190), energy_drain_rate=0.5)]

    # --- Step 2: Genetic Algorithm Optimization ---
    print("--- Starting Genetic Algorithm to Find Optimal Weights ---")
    hvt_initial_pos = np.random.uniform(50, 150, 2)
    ga = GeneticAlgorithm(GA_POPULATION_SIZE, GA_GENERATIONS, GA_MUTATION_RATE, GA_CROSSOVER_RATE, 
                          hvt_initial_pos, static_obstacles, dynamic_obstacles, no_fly_zones, threat_zones)
    best_weights = ga.evolve()
    
    # --- Step 3: Multi-Trial Statistical Analysis ---
    print(f"\n--- Starting {NUM_STATISTICAL_TRIALS} Statistical Trials ---")
    trial_metrics = []
    if os.path.exists(OUTPUT_DIR):
        for file in os.listdir(OUTPUT_DIR):
            if file.endswith("_metrics.csv") or file.endswith("_log.csv"):
                os.remove(os.path.join(OUTPUT_DIR, file))

    for i in range(NUM_STATISTICAL_TRIALS):
        hvt = HighValueTarget(np.random.uniform(50, 150, 2))
        sim = Simulation(NUM_UAVS, AREA_SIZE, hvt, static_obstacles, dynamic_obstacles, no_fly_zones, threat_zones, best_weights)
        metrics = sim.run_trial(trial_id=i, log_to_csv=True)
        trial_metrics.append(metrics)
        
        # Save metrics for statistical plotting later
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(os.path.join(OUTPUT_DIR, f'trial_{i}_metrics.csv'), index=False)

    avg_metrics = pd.DataFrame(trial_metrics).mean(numeric_only=True)
    
    print("\n--- Statistical Analysis Complete ---")
    print("Average Performance Metrics over all trials:")
    print(f"  Average Survivability Rate: {avg_metrics['survivability_rate']:.2f}")
    print(f"  Average HVT Neutralization Success: {avg_metrics['hvt_neutralized']:.2f}")
    print(f"  Average Steps to Neutralize: {avg_metrics['steps_to_neutralize']:.2f}")
    print(f"  Average Total Distance Traveled: {avg_metrics['total_distance_traveled']:.2f}")
    print(f"  Average Time to Target Lock: {avg_metrics['avg_time_to_target_lock']:.2f}")

    # --- Step 4: Visualization ---
    print("\nGenerating Analysis Plots from CSV Data...")
    visualizer = Visualizer()
    visualizer.plot_analysis_results(OUTPUT_DIR)
    
    # Run one final trial for visualization purposes
    print("\n--- Running one final trial for visualization... ---")
    final_hvt = HighValueTarget(np.random.uniform(50, 150, 2))
    final_sim = Simulation(NUM_UAVS, AREA_SIZE, final_hvt, static_obstacles, dynamic_obstacles, no_fly_zones, threat_zones, best_weights)
    final_metrics = final_sim.run_trial(trial_id="final", log_to_csv=True)
    final_log_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'trial_final_log.csv'))
    
    # FIXED: Now passing 'OUTPUT_DIR' to the function
    visualizer.plot_trial_path(final_log_df, final_sim.hvt.path, static_obstacles, dynamic_obstacles, no_fly_zones, threat_zones, OUTPUT_DIR)

if __name__ == "__main__":
    main()
