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
    nn_init_share: float = 0.2 # fraction of population from nearest neighbor
    lso_init_share: float = 0.5 # fraction of random init tours to improve with LS
    initialize_random: bool = False 
    
    # --- Local search intensity
    lso_variation_share: float = 1.0 # fraction of population to LS after variation
    lso_max_changes: int = 100 # cap for 2-opt/3-opt
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
    initialisation: str = "random" # "nn" | "random"
    recombination: str = "pmx" # "pmx" | "edge" | "cx" | "ox" | "csox"
    mutations: tuple[str, ...] = ("scramble",) # single or pool
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
    """random permutations + optional LS."""
    population = []
    for _ in range(cfg.llambda):
        perm = np.random.permutation(cfg.length)
        if cfg.local_search_initialisation:
            perm = cfg.local_search_fn(distance_matrix, perm, cfg.lso_max_changes)
        ind = Individual(perm, alpha=cfg.alpha)
        ind.update_fitness(distance_matrix)
        population.append(ind)
    return population

def nn_initialisation(distance_matrix: np.ndarray, cfg: EAConfig) -> list[Individual]:
    """falls back to random until implemented."""
    # TODO: greedy NN + diversification + cfg.nn_init_share / random_init_size split
    return random_initialisation(distance_matrix, cfg)


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

def order_crossover(parent1: Individual, parent2: Individual, cfg: EAConfig) -> list[Individual]:
    """DRY RUN: OX crossover."""
    p1, p2 = parent1.permutation, parent2.permutation
    n = len(p1)
    i, j = sorted(np.random.choice(n, 2, replace=False))
    child = np.full(n, -1, dtype=np.int64)
    child[i : j + 1] = p1[i : j + 1]
    fill = [x for x in p2 if x not in child]
    idx = 0
    for k in range(n):
        if child[k] == -1:
            child[k] = fill[idx]
            idx += 1
    return [_child_from_tour(child, parent1, parent2, cfg)]

def partially_mapped_crossover(parent1: Individual, parent2: Individual, cfg: EAConfig) -> list[Individual]:
    """use OX until PMX is implemented."""
    return order_crossover(parent1, parent2, cfg)

def cycle_crossover(parent1: Individual, parent2: Individual, cfg: EAConfig) -> list[Individual]:
    """."""
    return order_crossover(parent1, parent2, cfg)

def c_s_order_crossover(parent1: Individual, parent2: Individual, cfg: EAConfig) -> list[Individual]:
    """coordinated OX variant."""
    return order_crossover(parent1, parent2, cfg)

def edge_crossover(parent1: Individual, parent2: Individual, cfg: EAConfig) -> list[Individual]:
    """slow in Python"""
    return order_crossover(parent1, parent2, cfg)

# Optional helpers when implementing edge crossover:
def remove_ref(edge_table: dict, elem: int) -> None:
    for key in edge_table:
        edge_table[key] = [x for x in edge_table[key] if x != elem]
        
def find_next_element(edge_table: dict, elem: int) -> int:
    # TODO: pick best edge from table
    return edge_table[elem][0]




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