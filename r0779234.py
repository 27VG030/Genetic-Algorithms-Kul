import Reporter
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Any
from numba import njit

@dataclass
class EAConfig:
    # --- Population sizes
    llambda: int = 50 # parent population size
    mu: int = 100 # offspring per generation
    
    # ---Mutation / self-adaptation
    alpha: float = 0.2 # base mutation probability
    min_alpha: float = 0.05 # floor for self-adaptive alpha
    
    # --- Initialization 
    nn_init_share: float = 0.2 # fraction of population from nearest neighbor - 30% RCL tours, 70% random (diversity)
    lso_init_share: float = 0.5 # fraction of random init tours to improve with LS -  50% of init tours get 2-opt
    initialize_random: bool = False 
    
    # --- Local search intensity
    lso_variation_share: float = 0.3 # fraction of population to LS after variation
    lso_max_changes: int = 50 # cap for 2-opt/3-opt
    local_search_initialisation: bool = True
    local_search_variation: bool = True
    
    # --- Diversity promotion
    alpha_fitness_sharing: float = 1.0
    sigma_share: float = 0.1 # niche radius as fraction of n
    fitness_sharing_selection: bool = False
    fitness_sharing_elimination: bool = True
    crowding: bool = False
    k_crowding: int = 3    
    
    # --- Selection / elimination / elitism
    k_selection: int = 3
    k_elimination: int = 5
    k_greedy: int = 1
    k_elitism: int = 1
    elitism: bool = True
    
    # --- Island model
    island_model: bool = False
    swap_its: int = 5 # migrate every N generations
    swap_share: float = 0.1 # fraction of island pop to swap
    threading: bool = False # use 2 threads for 2 islands (2 CPU cores)
    
    # --- Other
    zero_included: bool = True # cycle-equivalent distance for sharing/crowding
    no_inf: bool = True # reject ∞ edges during NN init
    alpha_self_ad: bool = True
    
    # --- Operator choices
    elimination: str = "lambda_mu"
    selection: str = "tournament"
    initialisation: str = "nn" # "nn" | "random" -> RCL init
    k_greedy: int = 1 # 1=greedy NN; 3–5=real RCL randomization
    recombination: str = "ox" # "pmx" | "edge" | "cx" | "ox" | "csox"
    mutations: tuple[str, ...] = ("swap", "insert", "inversion") # single or pool
    local_search: str = "2opt" # "2opt" | "3opt"
    
    # --- Resolved once
    elimination_fn: Callable = field(default=None, repr=False)
    selection_fn: Callable = field(default=None, repr=False)
    initialisation_fn: Callable = field(default=None, repr=False)
    recombination_fn: Callable = field(default=None, repr=False)
    mutation_fns: tuple[Callable, ...] = field(default=(), repr=False)
    local_search_fn: Callable = field(default=None, repr=False)
    
    # --- Derived at runtime
    length: int = 0
    sigma: float = 0.0
    nn_init_size: int = 0
    random_init_size: int = 0
    swap_size: int = 0
    local_search_init_count: int = 0
    local_search_variation_count: int = 0
 
    def finalize(self, n: int) -> "EAConfig":
        self.length = n
        self.sigma = self.sigma_share * n
        self.nn_init_size = int(self.llambda * self.nn_init_share)
        self.random_init_size = self.llambda - self.nn_init_size
        self.swap_size = int(self.llambda // 2 * self.swap_share)
        self.local_search_init_count = int(self.lso_init_share * self.random_init_size)
        self.local_search_variation_count = int(self.lso_variation_share * self.llambda)
        return self
    
 
class r0779234:

	def __init__(self, config: EAConfig | None = None):
		self.reporter = Reporter.Reporter(self.__class__.__name__)
		self.config = config or EAConfig()

	# The evolutionary algorithm's main loop
	def optimize(self, filename, config: EAConfig | None = None):
		# Read distance matrix from file.		
		file = open(filename)
		distanceMatrix = np.loadtxt(file, delimiter=",")
		file.close()
		cfg = resolve_operators(config or self.config).finalize(len(distanceMatrix))

		if cfg.island_model:
			cfg1, cfg2 = island_configs(cfg)
			island1 = cfg1.initialisation_fn(distanceMatrix, cfg1)
			island2 = cfg2.initialisation_fn(distanceMatrix, cfg2)
			iteration = 1
			while True:
				if iteration % cfg.swap_its == 0:
					island1, island2 = migrate(island1, island2, cfg)
				if cfg.threading:
					island1, island2 = evolve_islands_parallel(island1, island2, cfg1, cfg2, distanceMatrix)
				else:
					island1 = evolve_island(island1, cfg1, distanceMatrix)
					island2 = evolve_island(island2, cfg2, distanceMatrix)
				population = island1 + island2
				meanObjective, bestObjective, bestSolution = report_stats(population)
				timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
				if timeLeft < 0:
					break
				iteration += 1
		else:
			# Initialize population
			population = cfg.initialisation_fn(distanceMatrix, cfg)
			iteration = 1
			while True:
				population = evolve_island(population, cfg, distanceMatrix)
				meanObjective, bestObjective, bestSolution = report_stats(population)
				timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
				if timeLeft < 0:
					break
				iteration += 1
		return 0
    
    
    
class Individual:
    """One candidate tour."""
    def __init__(self, permutation: np.ndarray, alpha: float = 0.2):
        self.permutation = np.asarray(permutation, dtype=np.int64)
        self.fitness: float = np.inf
        self.fitness_share: float = np.inf
        self.alpha = alpha  # for self-adaptive mutation
    def update_fitness(self, distance_matrix: np.ndarray) -> None:
        self.fitness = fitness(distance_matrix, self.permutation)
        
        
# UTILITIES


@njit
def fitness(distance_matrix: np.ndarray, permutation: np.ndarray) -> float:
    """Total tour length."""
    n = len(permutation)
    fitness = 0
    for i in range(n):
        fitness += distance_matrix[permutation[i], permutation[(i + 1) % n]]
        if fitness == np.inf:
            break
    return fitness

@njit
def distance(tour_a: np.ndarray, tour_b: np.ndarray, zero_included: bool = True) -> int:
    """Diversity distance between two tours."""
    if zero_included:
        shift_a = -np.where(tour_a == 0)[0][0]
        shift_b = -np.where(tour_b == 0)[0][0]
        return int(np.sum(np.roll(tour_a, shift_a) != np.roll(tour_b, shift_b)))
    return int(np.sum(tour_a != tour_b))

def report_stats(population: list[Individual]) -> tuple[float, float, np.ndarray]:
    """Mean, best fitness, best tour."""
    fitnesses = [ind.fitness for ind in population]
    best_idx = int(np.argmin(fitnesses))
    return float(np.mean(fitnesses)), float(fitnesses[best_idx]), np.array(population[best_idx].permutation)


# INITIALISATION

def random_initialisation(distance_matrix: np.ndarray, cfg: EAConfig) -> list[Individual]:
    """random permutations + optional LS. (O(n))"""
    population = []
    for _ in range(cfg.llambda):
        perm = np.random.permutation(cfg.length)
        if cfg.local_search_initialisation:
            perm = cfg.local_search_fn(distance_matrix, perm, cfg.lso_max_changes)
        ind = Individual(perm, alpha=cfg.alpha)
        ind.update_fitness(distance_matrix)
        population.append(ind)
    return population


@njit
def _rcl_nn_tour_fast(distance_matrix: np.ndarray, start_node: int, k: int, no_inf: bool) -> np.ndarray:
    """
    Blazing fast Numba kernel for randomized nearest neighbor via Restricted Candidate List.
    Complexity: O(k * N) per step, resulting in O(k * N^2) total construction time.
    """
    n = distance_matrix.shape[0]
    tour = np.empty(n, dtype=np.int64)
    visited = np.zeros(n, dtype=np.bool_)
    
    current = start_node
    tour[0] = current
    visited[current] = True
    
    # Pre-allocate candidate tracking arrays to avoid memory overhead inside the loop
    candidates = np.empty(k, dtype=np.int64)
    cand_costs = np.empty(k, dtype=np.float64)
    
    for step in range(1, n):
        # Reset the candidate pool for the current step
        for i in range(k):
            cand_costs[i] = np.inf
            candidates[i] = -1
            
        valid_count = 0
        
        # Scan unvisited nodes to maintain the top-k nearest neighbors
        for j in range(n):
            if not visited[j]:
                cost = distance_matrix[current, j]
                
                # Skip strictly infinite edges if the configuration demands it
                if no_inf and cost == np.inf:
                    continue
                    
                # Insertion sort logic into the top-k arrays
                if cost < cand_costs[k - 1]:
                    insert_idx = k - 1
                    while insert_idx > 0 and cost < cand_costs[insert_idx - 1]:
                        insert_idx -= 1
                    
                    # Shift elements right to make room
                    for m in range(k - 1, insert_idx, -1):
                        cand_costs[m] = cand_costs[m - 1]
                        candidates[m] = candidates[m - 1]
                        
                    # Insert new candidate
                    cand_costs[insert_idx] = cost
                    candidates[insert_idx] = j
                    
                    if valid_count < k:
                        valid_count += 1
                        
        if valid_count == 0:
            # Dead end reached (graph disconnected or all paths are np.inf). 
            # Fallback: pick the first available unvisited node to complete a valid permutation.
            for j in range(n):
                if not visited[j]:
                    current = j
                    break
        else:
            # Randomly select one city from the restricted candidate list
            limit = min(k, valid_count)
            chosen = np.random.randint(limit)
            current = candidates[chosen]
            
        tour[step] = current
        visited[current] = True
        
    return tour


def nn_initialisation(distance_matrix: np.ndarray, cfg: EAConfig) -> list[Individual]:
    """
    Initializes a population using the Randomized Nearest Neighbor (RCL) algorithm.
    Splits the initialization pool between RCL and pure random based on EAConfig.
    """
    population = []
    n = cfg.length
    
    # If the user sets k_greedy = 1, this acts as a pure greedy nearest neighbor.
    # A value between 3 and 5 is recommended for optimal RCL diversity.
    k_val = max(1, cfg.k_greedy) 
    
    # Generate the RCL Nearest Neighbor portion
    for _ in range(cfg.nn_init_size):
        start_node = np.random.randint(n)
        perm = _rcl_nn_tour_fast(distance_matrix, start_node, k_val, cfg.no_inf)
        
        # Apply Local Search Operator (LSO) conditionally to the initialized individual
        if cfg.local_search_initialisation and np.random.rand() < cfg.lso_init_share:
            perm = cfg.local_search_fn(distance_matrix, perm, cfg.lso_max_changes)
            
        ind = Individual(perm, alpha=cfg.alpha)
        ind.update_fitness(distance_matrix)
        population.append(ind)
        
    # Fill the remaining population size with pure random permutations
    for _ in range(cfg.random_init_size):
        perm = np.random.permutation(n)
        
        if cfg.local_search_initialisation and np.random.rand() < cfg.lso_init_share:
            perm = cfg.local_search_fn(distance_matrix, perm, cfg.lso_max_changes)
            
        ind = Individual(perm, alpha=cfg.alpha)
        ind.update_fitness(distance_matrix)
        population.append(ind)
        
    return population

# SELECTION
def k_t_selection(population: list[Individual], cfg: EAConfig) -> Individual:
    """k-tournament selection. Uses fitness_share if sharing-at-selection enabled."""
    k = cfg.k_selection
    candidates = list(np.random.choice(population, size=k, replace=False))
    if cfg.fitness_sharing_selection:
        return min(candidates, key=lambda ind: ind.fitness_share)
    return min(candidates, key=lambda ind: ind.fitness)



# MUTATION
def _mutation_rate(ind: Individual, cfg: EAConfig) -> float:
    return ind.alpha if cfg.alpha_self_ad else cfg.alpha


def swap_mutation(offspring: Individual, cfg: EAConfig) -> None:
    if np.random.rand() < _mutation_rate(offspring, cfg):
        i, j = np.random.choice(len(offspring.permutation), 2, replace=False)
        p = offspring.permutation
        p[i], p[j] = p[j], p[i]

def insert_mutation(offspring: Individual, cfg: EAConfig) -> None:
    if np.random.rand() < _mutation_rate(offspring, cfg):
        i, j = np.random.choice(len(offspring.permutation), 2, replace=False)
        if i > j:
            i, j = j, i
        offspring.permutation[i : j + 1] = np.roll(offspring.permutation[i : j + 1], 1)

def inversion_mutation(offspring: Individual, cfg: EAConfig) -> None:
    if np.random.rand() < _mutation_rate(offspring, cfg):
        i, j = np.random.choice(len(offspring.permutation), 2, replace=False)
        if i > j:
            i, j = j, i
        offspring.permutation[i : j + 1] = np.flip(offspring.permutation[i : j + 1])

def scramble_mutation(offspring: Individual, cfg: EAConfig) -> None:
    if np.random.rand() < _mutation_rate(offspring, cfg):
        i, j = np.random.choice(len(offspring.permutation), 2, replace=False)
        if i > j:
            i, j = j, i
        np.random.shuffle(offspring.permutation[i : j + 1])


# RECOMBINATION

def _child_from_tour(tour: np.ndarray, parent1: Individual, parent2: Individual, cfg: EAConfig) -> Individual:
    child = Individual(tour.copy(), alpha=cfg.alpha)
    if cfg.alpha_self_ad:
        beta = 2 * np.random.random() - 0.5
        child.alpha = max(cfg.min_alpha, parent1.alpha + beta * (parent2.alpha - parent1.alpha))
    return child


# Numba kernels

@njit
def _ox_kernel(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """O(N) Order Crossover Kernel"""
    n = len(p1)
    child = np.full(n, -1, dtype=np.int64)
    
    i, j = np.random.choice(n, 2, replace=False)
    if i > j:
        i, j = j, i
        
    visited = np.zeros(n, dtype=np.bool_)
    
    # Copy segment from p1
    for k in range(i, j + 1):
        child[k] = p1[k]
        visited[p1[k]] = True
        
    # Fill remaining from p2
    idx = (j + 1) % n
    for k in range(n):
        p2_val = p2[(j + 1 + k) % n]
        if not visited[p2_val]:
            child[idx] = p2_val
            idx = (idx + 1) % n
            
    return child

@njit
def _pmx_kernel(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """O(N) Partially Mapped Crossover Kernel"""
    n = len(p1)
    child = np.full(n, -1, dtype=np.int64)
    
    i, j = np.random.choice(n, 2, replace=False)
    if i > j:
        i, j = j, i
        
    in_segment = np.zeros(n, dtype=np.bool_)
    redirect = np.arange(n, dtype=np.int64)
    # Copy segment from p1; build redirect pairs (p1[k] <-> p2[k])
    for k in range(i, j + 1):
        child[k] = p1[k]
        in_segment[p1[k]] = True
        redirect[p1[k]] = p2[k]
        
    # Fill outside positions from p2, resolving conflicts via redirect chain
    for k in range(n):
        if i <= k <= j:
            continue
        val = p2[k]
        while in_segment[val]:
            val = redirect[val]
        child[k] = val
    return child

@njit
def _erx_kernel(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """O(N) Edge Recombination Crossover Kernel using flat arrays instead of dicts"""
    n = len(p1)
    
    # edge_table: [city_id, neighbor_index] -> neighbor_city_id
    # Max 4 unique neighbors per city in TSP crossover
    edge_table = np.full((n, 4), -1, dtype=np.int64)
    edge_counts = np.zeros(n, dtype=np.int64)
    
    # Build edge table in O(N)
    for p in (p1, p2):
        for k in range(n):
            city = p[k]
            left, right = p[(k - 1) % n], p[(k + 1) % n]
            
            # Insert left neighbor if not present
            found_l = False
            for c in range(edge_counts[city]):
                if edge_table[city, c] == left:
                    found_l = True
                    break
            if not found_l and edge_counts[city] < 4:
                edge_table[city, edge_counts[city]] = left
                edge_counts[city] += 1
                
            # Insert right neighbor if not present
            found_r = False
            for c in range(edge_counts[city]):
                if edge_table[city, c] == right:
                    found_r = True
                    break
            if not found_r and edge_counts[city] < 4:
                edge_table[city, edge_counts[city]] = right
                edge_counts[city] += 1

    child = np.empty(n, dtype=np.int64)
    visited = np.zeros(n, dtype=np.bool_)
    
    current = p1[0] if np.random.rand() < 0.5 else p2[0]
    
    for step in range(n):
        child[step] = current
        visited[current] = True
        
        # Remove current from all neighbor lists (simulate by ignoring visited later)
        
        if step == n - 1:
            break
            
        # Find next node: minimum edges remaining
                # Find next node: minimum edges remaining
        best_next = -1
        min_edges = n + 1
        first_valid = -1

        for idx in range(edge_counts[current]):
            candidate = edge_table[current, idx]
            if visited[candidate]:
                continue

            if first_valid == -1:
                first_valid = candidate

            c_edges = 0
            for j in range(edge_counts[candidate]):
                if not visited[edge_table[candidate, j]]:
                    c_edges += 1

            if c_edges < min_edges:
                min_edges = c_edges
                best_next = candidate
            elif c_edges == min_edges and np.random.rand() < 0.5:
                best_next = candidate

        if best_next == -1:
            if first_valid != -1:
                best_next = first_valid
            else:
                for candidate in range(n):
                    if not visited[candidate]:
                        best_next = candidate
                        break

        current = best_next
        
    return child


@njit
def _cx_kernel(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """O(N) Cycle Crossover."""
    n = len(p1)
    child = np.empty(n, dtype=np.int64)
    visited = np.zeros(n, dtype=np.bool_)
    pos2 = np.empty(n, dtype=np.int64)
    for idx in range(n):
        pos2[p2[idx]] = idx
    use_p1 = True
    for start in range(n):
        if visited[start]:
            continue
        idx = start
        while not visited[idx]:
            visited[idx] = True
            if use_p1:
                child[idx] = p1[idx]
            else:
                child[idx] = p2[idx]
            idx = pos2[p1[idx]]
        use_p1 = not use_p1
    return child


def order_crossover(parent1, parent2, cfg) -> list:
    child_tour = _ox_kernel(parent1.permutation, parent2.permutation)
    return [_child_from_tour(child_tour, parent1, parent2, cfg)]

def partially_mapped_crossover(parent1, parent2, cfg) -> list:
    child_tour = _pmx_kernel(parent1.permutation, parent2.permutation)
    return [_child_from_tour(child_tour, parent1, parent2, cfg)]

def edge_crossover(parent1, parent2, cfg) -> list:
    child_tour = _erx_kernel(parent1.permutation, parent2.permutation)
    return [_child_from_tour(child_tour, parent1, parent2, cfg)]

def cycle_crossover(parent1, parent2, cfg) -> list:
    child_tour = _cx_kernel(parent1.permutation, parent2.permutation)
    return [_child_from_tour(child_tour, parent1, parent2, cfg)]

def c_s_order_crossover(parent1, parent2, cfg) -> list:
    """CSOX relies on OX. Wrapper manages the multi-child generation."""
    # Generate multiple OX children to represent subtour permutations
    c1 = _ox_kernel(parent1.permutation, parent2.permutation)
    c2 = _ox_kernel(parent2.permutation, parent1.permutation)
    return [
        _child_from_tour(c1, parent1, parent2, cfg),
        _child_from_tour(c2, parent2, parent1, cfg)
    ]





# LOCAL SEARCH

def two_opt(distance_matrix: np.ndarray, permutation: np.ndarray, max_changes: int = 100) -> np.ndarray:
    """first-improvement 2-opt with change cap."""
    tour = permutation.copy()
    n = len(tour)
    improved = True
    changes = 0
    while improved and changes < max_changes:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n if i > 0 else n - 1):
                a, b = tour[i], tour[(i + 1) % n]
                c, d = tour[j], tour[(j + 1) % n]
                old_cost = distance_matrix[a, b] + distance_matrix[c, d]
                new_cost = distance_matrix[a, c] + distance_matrix[b, d]
                if new_cost < old_cost:
                    tour[i + 1 : j + 1] = tour[i + 1 : j + 1][::-1]
                    improved = True
                    changes += 1
                    break
            if improved:
                break
    return tour

def three_opt(distance_matrix: np.ndarray, permutation: np.ndarray, max_changes: int = 100) -> np.ndarray:
    """delegate to 2-opt until real 3-opt is implemented."""
    return two_opt(distance_matrix, permutation, max_changes)


# DIVERSITY PROMOTION

def fitness_sharing_sel(population: list[Individual], cfg: EAConfig) -> None:
    """Adjust fitness_share before selection."""
    for ind in population:
        ind.fitness_share = ind.fitness
    for i, ind_i in enumerate(population):
        niche = 0.0
        for j, ind_j in enumerate(population):
            if i == j:
                continue
            dist = distance(ind_i.permutation, ind_j.permutation, cfg.zero_included)
            if dist <= cfg.sigma:
                niche += 1.0 - (dist / cfg.sigma) ** cfg.alpha_fitness_sharing
        ind_i.fitness_share = ind_i.fitness * (1.0 + niche)
        
def fitness_sharing_el(
    population: list[Individual],
    cfg: EAConfig,
    size: int,
    mu_lambda: bool = True,
) -> list[Individual]:
    """Fitness sharing during elimination."""
    for ind in population:
        ind.fitness_share = ind.fitness
    new_population: list[Individual] = []
    pool = list(population)
    for _ in range(size):
        best = min(pool, key=lambda ind: ind.fitness_share)
        new_population.append(best)
        pool.remove(best)
        for ind in pool:
            dist = distance(best.permutation, ind.permutation, cfg.zero_included)
            if dist <= cfg.sigma:
                ind.fitness_share += ind.fitness * (
                    1.0 - (dist / cfg.sigma) ** cfg.alpha_fitness_sharing
                )
    return new_population

def crowding_replace(population: list[Individual], offspring: Individual, cfg: EAConfig) -> list[Individual]:
    """De Jong crowding: replace most similar of k random individuals."""
    if len(population) == 0:
        return [offspring]
    candidates = list(np.random.choice(population, size=min(cfg.k_crowding, len(population)), replace=False))
    closest = min(
        candidates,
        key=lambda ind: distance(offspring.permutation, ind.permutation, cfg.zero_included),
    )
    new_pop = [ind for ind in population if ind is not closest]
    new_pop.append(offspring)
    return new_pop



# ELIMINATION


def mu_elimination(offspring: list[Individual], cfg: EAConfig, size: int) -> list[Individual]:
    """(lambda, mu): keep best mu from offspring only."""
    return sorted(offspring, key=lambda ind: ind.fitness)[:size]

def lambda_mu_elimination(merged: list[Individual], cfg: EAConfig, size: int) -> list[Individual]:
    """(lambda + mu): keep best size from merged pool."""
    if cfg.fitness_sharing_elimination:
        return fitness_sharing_el(merged, cfg, size, mu_lambda=True)
    return sorted(merged, key=lambda ind: ind.fitness)[:size]

def k_t_elimination(merged: list[Individual], cfg: EAConfig, size: int) -> list[Individual]:
    """k-tournament elimination."""
    if cfg.fitness_sharing_elimination:
        return fitness_sharing_el(merged, cfg, size, mu_lambda=False)
    new_population = []
    pool = list(merged)
    for _ in range(size):
        candidates = list(np.random.choice(pool, size=cfg.k_elimination, replace=False))
        winner = min(candidates, key=lambda ind: ind.fitness)
        new_population.append(winner)
        pool.remove(winner)
    return new_population


# ISLAND MODEL 


def evolve_island(
    population: list[Individual],
    cfg: EAConfig,
    distance_matrix: np.ndarray,
) -> list[Individual]:
    """One generation on one island."""
    offspring: list[Individual] = []
    target_offspring = cfg.mu // 2  # pairs of children; adjust if your crossover returns 1 child
    if cfg.fitness_sharing_selection:
        fitness_sharing_sel(population, cfg)
    # --- Variation: crossover + mutation ---
    while len(offspring) < target_offspring:
        parent1 = cfg.selection_fn(population, cfg)
        parent2 = cfg.selection_fn(population, cfg)
        children = cfg.recombination_fn(parent1, parent2, cfg)
        for child in children:
            mut_fn = cfg.mutation_fns[np.random.randint(len(cfg.mutation_fns))]
            mut_fn(child, cfg)
            child.update_fitness(distance_matrix)
            offspring.append(child)
            if len(offspring) >= target_offspring:
                break
    # --- Elitism ---
    elites: list[Individual] = []
    if cfg.elitism:
        sorted_pop = sorted(population, key=lambda ind: ind.fitness)
        elites = sorted_pop[: cfg.k_elitism]
    # --- Mutate parents (optional, as in reference) ---
    for ind in population:
        mut_fn = cfg.mutation_fns[np.random.randint(len(cfg.mutation_fns))]
        mut_fn(ind, cfg)
        ind.update_fitness(distance_matrix)
    if cfg.elitism:
        population = elites + [ind for ind in population if ind not in elites]
    # --- Local search on part of population ---
    if cfg.local_search_variation:
        count = min(cfg.local_search_variation_count, len(population))
        for i in range(count):
            improved = cfg.local_search_fn(distance_matrix, population[i].permutation, cfg.lso_max_changes)
            if not np.array_equal(improved, population[i].permutation):
                population[i].permutation = improved
                population[i].update_fitness(distance_matrix)
    merged = population + offspring
    return cfg.elimination_fn(merged, cfg, len(population))


def migrate(
    island1: list[Individual],
    island2: list[Individual],
    cfg: EAConfig,
) -> tuple[list[Individual], list[Individual]]:
    """Exchange swap_size individuals between islands."""
    n_swap = min(cfg.swap_size, len(island1), len(island2))
    if n_swap <= 0:
        return island1, island2
    idx1 = np.random.choice(len(island1), n_swap, replace=False)
    idx2 = np.random.choice(len(island2), n_swap, replace=False)
    migrants1 = [island1[i] for i in idx1]
    migrants2 = [island2[i] for i in idx2]
    for i in sorted(idx1, reverse=True):
        del island1[i]
    for i in sorted(idx2, reverse=True):
        del island2[i]
    return island1 + migrants2, island2 + migrants1


def island_configs(base: EAConfig) -> tuple[EAConfig, EAConfig]:
    """Create two island configs with different operators."""
    from dataclasses import replace
    cfg1 = resolve_operators(replace(base, recombination="ox", mutations=("swap", "insert"))).finalize(base.length)
    cfg2 = resolve_operators(replace(base, recombination="pmx", mutations=("swap", "inversion"))).finalize(base.length)
    return cfg1, cfg2


def evolve_islands_parallel(
    island1: list[Individual],
    island2: list[Individual],
    cfg1: EAConfig,
    cfg2: EAConfig,
    distance_matrix: np.ndarray,
) -> tuple[list[Individual], list[Individual]]:
    """Use 2 threads (Numba-heavy kernels release GIL)."""
    import threading
    results: list[list[Individual]] = [island1, island2]
    def worker(idx: int, cfg: EAConfig) -> None:
        results[idx] = evolve_island(results[idx], cfg, distance_matrix)
    t1 = threading.Thread(target=worker, args=(0, cfg1))
    t2 = threading.Thread(target=worker, args=(1, cfg2))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    return results[0], results[1]


OPERATORS = {
	# elimination
	"lambda_mu": lambda_mu_elimination,
	"mu_only": mu_elimination,
	"k_tournament": k_t_elimination,
	# selection
	"tournament": k_t_selection,
	# init
	"nn": nn_initialisation,
	"random": random_initialisation,
	# crossover
	"pmx": partially_mapped_crossover,
	"edge": edge_crossover,
	"cx": cycle_crossover,
	"ox": order_crossover,
	"csox": c_s_order_crossover,
	# mutation
	"scramble": scramble_mutation,
	"swap": swap_mutation,
	"insert": insert_mutation,
	"inversion": inversion_mutation,
	# local search
	"2opt": two_opt,
	"3opt": three_opt,
}

def resolve_operators(cfg: EAConfig) -> EAConfig:
    cfg.elimination_fn = OPERATORS[cfg.elimination]
    cfg.selection_fn = OPERATORS[cfg.selection]
    cfg.initialisation_fn = OPERATORS[cfg.initialisation]
    cfg.recombination_fn = OPERATORS[cfg.recombination]
    cfg.mutation_fns = tuple(OPERATORS[m] for m in cfg.mutations)
    cfg.local_search_fn = OPERATORS[cfg.local_search]
    return cfg
    
if __name__ == "__main__":
    ea = r0779234(None)
    ea.optimize("benchmark/tour50.csv")   