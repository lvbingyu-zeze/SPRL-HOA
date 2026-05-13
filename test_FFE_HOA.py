import numpy as np
from scipy.spatial.distance import cdist
from typing import Callable, Tuple, List, Optional, Dict
import warnings
warnings.filterwarnings('ignore')

# ==================== CNN MODULE (PyTorch) ====================
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. CNN features will be disabled.")

class SolutionEncoder:
    """
    Encode each solution as an image of size H x W with 3 channels:
    - Channel 1: The solution itself (normalized)
    - Channel 2: Neighborhood aggregation (mean of neighbors)
    - Channel 3: Fitness value (broadcasted)
    """
    def __init__(self, dim: int, image_size: Tuple[int, int] = None):
        self.dim = dim
        # Determine image size - ensure minimum 4x4 for CNN pooling
        if image_size is None:
            h = max(4, int(np.ceil(np.sqrt(dim))))
            w = max(4, int(np.ceil(dim / h)))
            self.H, self.W = h, w
        else:
            self.H, self.W = max(4, image_size[0]), max(4, image_size[1])
        self.pad_size = self.H * self.W - dim

    def encode(self, population: np.ndarray, fitness: np.ndarray,
               neighbors_agg: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Encode population to image batch: (N, 3, H, W)
        """
        N, D = population.shape

        # Pad solutions to H*W
        if self.pad_size > 0:
            padded = np.pad(population, ((0, 0), (0, self.pad_size)), mode='edge')
        else:
            padded = population

        # Channel 1: Solution itself (normalized to [0,1] per individual)
        ch1 = padded.reshape(N, self.H, self.W)
        ch1_min = ch1.min(axis=(1,2), keepdims=True)
        ch1_max = ch1.max(axis=(1,2), keepdims=True)
        ch1_range = ch1_max - ch1_min
        ch1_range[ch1_range == 0] = 1.0
        ch1 = (ch1 - ch1_min) / ch1_range

        # Channel 2: Neighborhood aggregation
        if neighbors_agg is None:
            neighbors_agg = np.mean(population, axis=0, keepdims=True)
            neighbors_agg = np.repeat(neighbors_agg, N, axis=0)

        if self.pad_size > 0:
            padded_agg = np.pad(neighbors_agg, ((0, 0), (0, self.pad_size)), mode='edge')
        else:
            padded_agg = neighbors_agg
        ch2 = padded_agg.reshape(N, self.H, self.W)
        ch2_min = ch2.min(axis=(1,2), keepdims=True)
        ch2_max = ch2.max(axis=(1,2), keepdims=True)
        ch2_range = ch2_max - ch2_min
        ch2_range[ch2_range == 0] = 1.0
        ch2 = (ch2 - ch2_min) / ch2_range

        # Channel 3: Fitness (broadcasted across image)
        f_min, f_max = fitness.min(), fitness.max()
        f_range = f_max - f_min if f_max != f_min else 1.0
        fitness_norm = (fitness - f_min) / f_range
        ch3 = np.broadcast_to(fitness_norm[:, None, None], (N, self.H, self.W))

        # Stack channels: (N, 3, H, W)
        images = np.stack([ch1, ch2, ch3], axis=1)
        return images.astype(np.float32)


class AdaptiveCNN(nn.Module):
    """
    Att48 dateset
    Adaptive CNN that handles variable input sizes.
    Architecture based on paper specifications:
    - Conv1: 1->32 channels, kernel 3, padding 1
    - Conv2: 32->64 channels, kernel 5, padding 2
    - MaxPool: 2x2, stride 2
    - Adaptive pooling to fixed size
    - Output head: 100-dim ReLU -> 1-dim Sigmoid (probability)
    """
    def __init__(self, input_channels: int = 3, feature_dim: int = 64):
        super(AdaptiveCNN, self).__init__()

        # Feature extraction layers (matching paper: 1->32, 32->64)
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Adaptive pooling to fixed size (handles any input)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Flattened size after adaptive pool: 64 * 4 * 4 = 1024
        self.flat_size = 64 * 4 * 4

        # Feature projection to desired dimension
        self.feature_proj = nn.Linear(self.flat_size, feature_dim)

        # Output head for probability (100-dim ReLU -> 1-dim Sigmoid)
        self.fc1 = nn.Linear(feature_dim, 100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            features: (N, feature_dim) - extracted feature vector F_CNN
            probabilities: (N, 1) - probability from sigmoid
        """
        # Feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        # Adaptive pooling to fixed 4x4
        x = self.adaptive_pool(x)

        # Flatten
        flat = x.view(x.size(0), -1)  # (N, flat_size)

        # Project to feature dimension
        features = F.relu(self.feature_proj(flat))  # (N, feature_dim)

        # Output probability
        h = F.relu(self.fc1(features))
        prob = torch.sigmoid(self.fc2(h))

        return features, prob


# ==================== IMPROVED HFOA WITH CNN ====================

class FFE_HOA:
    """
    FFE-HOA

    Key enhancement: CNN module extracts features from solution images
    to adaptively guide step size and direction updates.

    Paper equations implemented:
    s_{i,j} = s_{i,j} + x1 * f_{i,j} * (N^{global}_j - N_{i,j})
    d_{i,j} = d_{i,j} + x2 * f_{i,j} * (N_{i,j} - N^{local}_j)

    where f_{i,j} is the j-th component of CNN feature vector F_CNN.
    """

    def __init__(
        self,
        objective_func: Callable,
        dim: int,
        bounds: List[Tuple[float, float]],
        population_size: int = 50,
        max_iterations: int = 100,
        alpha: float = 0.6,      # x1 in paper: learning coefficient for step
        beta: float = 0.7,       # x2 in paper: learning coefficient for direction
        w: float = 0.5,
        n_clusters: int = 4,
        V_male: float = 2.0,
        V_female: float = 1.0,
        step_size_init: float = 0.1,
        minimize: bool = True,
        constraint_func: Optional[Callable] = None,
        epsilon: float = 1e-6,
        max_threshold_iterations: int = 10,
        # CNN parameters
        use_cnn: bool = True,
        cnn_feature_dim: int = 64,
        cnn_update_freq: int = 5,     # Update CNN every N iterations
        cnn_learning_rate: float = 1e-3,
        device: str = 'cpu'
    ):
        self.objective_func = objective_func
        self.dim = dim
        self.bounds = np.array(bounds, dtype=float)
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.beta = beta
        self.w = w
        self.n_clusters = n_clusters
        self.V_male = V_male
        self.V_female = V_female
        self.step_size_init = step_size_init
        self.minimize = minimize
        self.constraint_func = constraint_func
        self.epsilon = epsilon
        self.max_threshold_iterations = max_threshold_iterations

        # CNN settings
        self.use_cnn = use_cnn and TORCH_AVAILABLE
        self.cnn_feature_dim = cnn_feature_dim
        self.cnn_update_freq = cnn_update_freq
        self.device = torch.device(device if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu')

        if len(bounds) != dim:
            raise ValueError(f"Bounds length ({len(bounds)}) must match dimension ({dim})")

        self.range_min = np.min(self.bounds[:, 1] - self.bounds[:, 0])

        # Initialize CNN components
        if self.use_cnn:
            self.encoder = SolutionEncoder(dim)
            self.H, self.W = self.encoder.H, self.encoder.W
            self.cnn = AdaptiveCNN(input_channels=3, feature_dim=cnn_feature_dim).to(self.device)
            self.cnn_optimizer = torch.optim.Adam(self.cnn.parameters(), lr=cnn_learning_rate)
            self.cnn_criterion = nn.MSELoss()

        # State variables
        self.population = None
        self.genders = None
        self.fitness_f1 = None
        self.fitness_f2 = None
        self.step_sizes = None
        self.direction_vectors = None
        self.cnn_features = None  # Store CNN features for each individual
        self.cnn_probabilities = None

        # Tracking
        self.best_solution = None
        self.best_fitness = np.inf if minimize else -np.inf
        self.fitness_history = []
        self.best_fitness_history = []
        self.gender_history = []
        self.threshold_history = []
        self.cnn_loss_history = []

    def _inverse_fitness(self, fitness_vals: np.ndarray) -> np.ndarray:
        f_max = np.max(fitness_vals)
        f_min = np.min(fitness_vals)
        if abs(f_max - f_min) < 1e-10:
            return np.ones_like(fitness_vals) * f_max
        return f_max + f_min - fitness_vals

    def _initialize_population(self):
        lower = self.bounds[:, 0]
        upper = self.bounds[:, 1]
        self.population = np.random.uniform(lower, upper, size=(self.population_size, self.dim))
        self.genders = np.array(['F'] * self.population_size)
        self.step_sizes = np.ones((self.population_size, self.dim)) * self.step_size_init
        self.direction_vectors = np.random.uniform(-1, 1, (self.population_size, self.dim))
        norms = np.linalg.norm(self.direction_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.direction_vectors = self.direction_vectors / norms

        # Initialize CNN features
        if self.use_cnn:
            self.cnn_features = np.zeros((self.population_size, self.cnn_feature_dim))
            self.cnn_probabilities = np.zeros(self.population_size)

    def _evaluate_fitness(self, population: np.ndarray) -> np.ndarray:
        fitness = np.array([self.objective_func(ind) for ind in population])
        if self.constraint_func is not None:
            penalties = np.array([self.constraint_func(ind) for ind in population])
            fitness = fitness + penalties
        return fitness

    def _clip_to_bounds(self, positions: np.ndarray) -> np.ndarray:
        lower = self.bounds[:, 0]
        upper = self.bounds[:, 1]
        return np.clip(positions, lower, upper)

    def _compute_neighbors_aggregation(self) -> np.ndarray:
        """Compute neighborhood aggregation for each individual."""
        agg = np.zeros_like(self.population)
        for i in range(self.population_size):
            neighbors = self._find_neighbors(i)
            if len(neighbors) > 0:
                agg[i] = np.mean(self.population[neighbors], axis=0)
            else:
                agg[i] = self.population[i]
        return agg

    def _extract_cnn_features(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract CNN features and probabilities for current population.
        Returns: (features_array, probabilities_array)
        """
        if not self.use_cnn:
            return None, None

        self.cnn.eval()
        with torch.no_grad():
            neighbors_agg = self._compute_neighbors_aggregation()
            images = self.encoder.encode(self.population, self.fitness_f1, neighbors_agg)
            images_tensor = torch.from_numpy(images).to(self.device)
            features, probs = self.cnn(images_tensor)
            return features.cpu().numpy(), probs.cpu().numpy().flatten()

    def _train_cnn(self, iteration: int):
        """
        Train CNN to predict fitness quality.
        Target: normalized fitness (0=best, 1=worst)
        """
        if not self.use_cnn:
            return 0.0

        self.cnn.train()

        neighbors_agg = self._compute_neighbors_aggregation()
        images = self.encoder.encode(self.population, self.fitness_f1, neighbors_agg)
        images_tensor = torch.from_numpy(images).to(self.device)

        # Target: normalized fitness
        if self.minimize:
            f_min, f_max = self.fitness_f1.min(), self.fitness_f1.max()
            f_range = f_max - f_min if f_max != f_min else 1.0
            targets = (self.fitness_f1 - f_min) / f_range
        else:
            f_min, f_max = self.fitness_f1.min(), self.fitness_f1.max()
            f_range = f_max - f_min if f_max != f_min else 1.0
            targets = (f_max - self.fitness_f1) / f_range

        targets_tensor = torch.from_numpy(targets).float().to(self.device).unsqueeze(1)

        _, probs = self.cnn(images_tensor)
        loss = self.cnn_criterion(probs, targets_tensor)

        self.cnn_optimizer.zero_grad()
        loss.backward()
        self.cnn_optimizer.step()

        return loss.item()

    def _dynamic_clustering(self) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
        n_clusters = min(self.n_clusters, self.population_size)
        center_indices = np.random.choice(self.population_size, size=n_clusters, replace=False)
        centers = self.population[center_indices].copy()

        for _ in range(15):
            dist_to_centers = cdist(self.population, centers, metric='euclidean')
            labels = np.argmin(dist_to_centers, axis=1)
            new_centers = []
            for k in range(n_clusters):
                cluster_members = np.where(labels == k)[0]
                if len(cluster_members) > 0:
                    new_centers.append(self.population[cluster_members].mean(axis=0))
                else:
                    new_centers.append(self.population[np.random.randint(self.population_size)])
            centers = np.array(new_centers)

        dist_to_centers = cdist(self.population, centers, metric='euclidean')
        labels = np.argmin(dist_to_centers, axis=1)

        clusters = []
        cluster_leaders = []
        for k in range(n_clusters):
            cluster_indices = np.where(labels == k)[0]
            if len(cluster_indices) > 0:
                clusters.append(cluster_indices)
                cluster_fitness = self.fitness_f1[cluster_indices]
                if self.minimize:
                    best_idx = cluster_indices[np.argmin(cluster_fitness)]
                else:
                    best_idx = cluster_indices[np.argmax(cluster_fitness)]
                cluster_leaders.append(best_idx)

        return clusters, np.array(cluster_leaders), centers

    def _adaptive_gender_switching(self) -> float:
        sorted_fitness = np.sort(self.fitness_f1)
        n = len(sorted_fitness)

        if self.minimize:
            F_a_init = np.mean(sorted_fitness[:n//3]) if n >= 3 else sorted_fitness[0]
            F_b_init = np.mean(sorted_fitness[n//3:2*n//3]) if n >= 3 else sorted_fitness[n//2]
            F_c_init = np.mean(sorted_fitness[2*n//3:]) if n >= 3 else sorted_fitness[-1]
        else:
            F_a_init = np.mean(sorted_fitness[2*n//3:]) if n >= 3 else sorted_fitness[-1]
            F_b_init = np.mean(sorted_fitness[n//3:2*n//3]) if n >= 3 else sorted_fitness[n//2]
            F_c_init = np.mean(sorted_fitness[:n//3]) if n >= 3 else sorted_fitness[0]

        F_threshold_1 = (F_a_init + F_b_init) / 2
        F_threshold_2 = (F_b_init + F_c_init) / 2

        for _ in range(self.max_threshold_iterations):
            if self.minimize:
                mask_a = self.fitness_f1 <= F_threshold_1
                mask_b = (self.fitness_f1 > F_threshold_1) & (self.fitness_f1 <= F_threshold_2)
                mask_c = self.fitness_f1 > F_threshold_2
            else:
                mask_a = self.fitness_f1 >= F_threshold_2
                mask_b = (self.fitness_f1 >= F_threshold_1) & (self.fitness_f1 < F_threshold_2)
                mask_c = self.fitness_f1 < F_threshold_1

            F_a = np.mean(self.fitness_f1[mask_a]) if np.any(mask_a) else F_threshold_1
            F_b = np.mean(self.fitness_f1[mask_b]) if np.any(mask_b) else (F_threshold_1 + F_threshold_2) / 2
            F_c = np.mean(self.fitness_f1[mask_c]) if np.any(mask_c) else F_threshold_2

            F_threshold_1_new = (F_a + F_b) / 2
            F_threshold_2_new = (F_b + F_c) / 2

            if abs(F_threshold_1_new - F_threshold_2_new) < self.epsilon:
                F_threshold_1 = F_threshold_1_new
                F_threshold_2 = F_threshold_2_new
                break

            F_threshold_1 = F_threshold_1_new
            F_threshold_2 = F_threshold_2_new

        F_threshold = (F_threshold_1 + F_threshold_2) / 2

        for i in range(self.population_size):
            if self.minimize:
                self.genders[i] = 'F' if self.fitness_f1[i] <= F_threshold else 'M'
            else:
                self.genders[i] = 'F' if self.fitness_f1[i] >= F_threshold else 'M'

        return F_threshold

    def _find_neighbors(self, fish_idx: int) -> np.ndarray:
        visual_scope = self.V_male if self.genders[fish_idx] == 'M' else self.V_female
        distances = np.linalg.norm(self.population - self.population[fish_idx], axis=1)
        neighbors = np.where(
            (distances <= visual_scope) & (np.arange(self.population_size) != fish_idx)
        )[0]
        return neighbors

    def _update_positions_cnn(self, clusters: List[np.ndarray], cluster_leaders: np.ndarray):
        """
        Update positions using CNN-extracted features.

        Paper equations:
        s_{i,j} = s_{i,j} + x1 * f_{i,j} * (N^{global}_j - N_{i,j})
        d_{i,j} = d_{i,j} + x2 * f_{i,j} * (N_{i,j} - N^{local}_j)

        where f_{i,j} is the j-th component of CNN feature vector F_CNN.
        """
        if self.minimize:
            global_best_idx = np.argmin(self.fitness_f1)
        else:
            global_best_idx = np.argmax(self.fitness_f1)
        global_best = self.population[global_best_idx]

        old_population = self.population.copy()
        new_population = self.population.copy()

        # Map CNN features to dimension space for modulation
        if self.use_cnn and self.cnn_features is not None:
            feat_dim = self.cnn_features.shape[1]
            if feat_dim >= self.dim:
                f_mapped = self.cnn_features[:, :self.dim]
                f_min = f_mapped.min(axis=1, keepdims=True)
                f_max = f_mapped.max(axis=1, keepdims=True)
                f_range = f_max - f_min
                f_range[f_range == 0] = 1.0
                f_mapped = (f_mapped - f_min) / f_range
            else:
                repeats = int(np.ceil(self.dim / feat_dim))
                f_mapped = np.tile(self.cnn_features, (1, repeats))[:, :self.dim]
                f_min = f_mapped.min(axis=1, keepdims=True)
                f_max = f_mapped.max(axis=1, keepdims=True)
                f_range = f_max - f_min
                f_range[f_range == 0] = 1.0
                f_mapped = (f_mapped - f_min) / f_range
        else:
            f_mapped = np.random.uniform(0.5, 1.0, (self.population_size, self.dim))

        for i in range(self.population_size):
            # Find local best (best neighbor)
            neighbors = self._find_neighbors(i)
            if len(neighbors) > 0:
                neighbor_fitness = self.fitness_f1[neighbors]
                if self.minimize:
                    best_neighbor_idx = neighbors[np.argmin(neighbor_fitness)]
                else:
                    best_neighbor_idx = neighbors[np.argmax(neighbor_fitness)]
                local_best = self.population[best_neighbor_idx]
            else:
                local_best = self.population[i]

            # CNN-guided step size update: s_{i,j} += x1 * f_{i,j} * (N^{global}_j - N_{i,j})
            self.step_sizes[i] += self.alpha * f_mapped[i] * (global_best - self.population[i])
            self.step_sizes[i] = np.clip(self.step_sizes[i], 0.001, self.range_min * 0.5)

            # CNN-guided direction update: d_{i,j} += x2 * f_{i,j} * (N_{i,j} - N^{local}_j)
            self.direction_vectors[i] += self.beta * f_mapped[i] * (self.population[i] - local_best)
            norm = np.linalg.norm(self.direction_vectors[i])
            if norm > 0:
                self.direction_vectors[i] /= norm

            # Position update using updated step and direction
            new_population[i] += self.step_sizes[i] * self.direction_vectors[i]

            # Cluster leader attraction
            for c_idx, cluster in enumerate(clusters):
                if i in cluster:
                    leader = self.population[cluster_leaders[c_idx]]
                    new_population[i] += self.w * (leader - self.population[i])
                    break

            # CNN probability-based exploration/exploitation balance
            if self.use_cnn and self.cnn_probabilities is not None:
                p = self.cnn_probabilities[i]
                if np.random.random() < p:
                    # Exploitation: move toward global best
                    new_population[i] += 0.1 * (global_best - new_population[i])
                else:
                    # Exploration: random perturbation
                    noise = np.random.normal(0, 0.1 * self.range_min, self.dim)
                    new_population[i] += noise

            new_population[i] = self._clip_to_bounds(new_population[i].reshape(1, -1))[0]

        self.population = new_population

    def _update_fitness(self):
        self.fitness_f1 = self._evaluate_fitness(self.population)
        self.fitness_f2 = self._inverse_fitness(self.fitness_f1)

        if self.minimize:
            best_idx = np.argmin(self.fitness_f1)
            if self.fitness_f1[best_idx] < self.best_fitness:
                self.best_fitness = self.fitness_f1[best_idx]
                self.best_solution = self.population[best_idx].copy()
        else:
            best_idx = np.argmax(self.fitness_f1)
            if self.fitness_f1[best_idx] > self.best_fitness:
                self.best_fitness = self.fitness_f1[best_idx]
                self.best_solution = self.population[best_idx].copy()

    def optimize(self, verbose: bool = True) -> Tuple[np.ndarray, float, dict]:
        self._initialize_population()
        self._update_fitness()

        if verbose:
            print(f"{'Iter':>6} | {'Best Fit':>12} | {'Avg Fit':>12} | {'Threshold':>12} | {'Genders (M/F)':>15} | {'CNN Loss':>10}")
            print("-" * 95)

        for iteration in range(self.max_iterations):
            cnn_loss = 0.0
            if self.use_cnn and iteration % self.cnn_update_freq == 0:
                cnn_loss = self._train_cnn(iteration)
                self.cnn_features, self.cnn_probabilities = self._extract_cnn_features()

            clusters, cluster_leaders, _ = self._dynamic_clustering()
            self._update_positions_cnn(clusters, cluster_leaders)
            self._update_fitness()
            F_threshold = self._adaptive_gender_switching()

            self.fitness_history.append(np.mean(self.fitness_f1))
            self.best_fitness_history.append(self.best_fitness)
            n_males = np.sum(self.genders == 'M')
            n_females = np.sum(self.genders == 'F')
            self.gender_history.append((n_males, n_females))
            self.threshold_history.append(F_threshold)
            self.cnn_loss_history.append(cnn_loss)

            if verbose and (iteration % 10 == 0 or iteration == self.max_iterations - 1):
                print(f"{iteration:>6} | {self.best_fitness:>12.4e} | {np.mean(self.fitness_f1):>12.4e} | "
                      f"{F_threshold:>12.4e} | {n_males:>3}/{n_females:<3} | {cnn_loss:>10.4f}")

        history = {
            'fitness_history': self.fitness_history,
            'best_fitness_history': self.best_fitness_history,
            'gender_history': self.gender_history,
            'threshold_history': self.threshold_history,
            'cnn_loss_history': self.cnn_loss_history
        }

        return self.best_solution, self.best_fitness, history


# ==================== BENCHMARK FUNCTIONS ====================

def sphere_function(x: np.ndarray) -> float:
    """Sphere function: f(x) = sum(x_i^2), global minimum at x=0, f(0)=0"""
    return np.sum(x**2)

def ackley_function(x: np.ndarray) -> float:
    """Ackley function, global minimum at x=0, f(0)=0"""
    a, b, c = 20, 0.2, 2 * np.pi
    n = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(c * x))
    return -a * np.exp(-b * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + a + np.exp(1)

def rastrigin_function(x: np.ndarray) -> float:
    """Rastrigin function, global minimum at x=0, f(0)=0"""
    A = 10
    n = len(x)
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def rosenbrock_function(x: np.ndarray) -> float:
    """Rosenbrock function, global minimum at x=1, f(1)=0"""
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def griewank_function(x: np.ndarray) -> float:
    """Griewank function, global minimum at x=0, f(0)=0"""
    n = len(x)
    sum_part = np.sum(x**2) / 4000
    prod_part = np.prod(np.cos(x / np.sqrt(np.arange(1, n + 1))))
    return sum_part - prod_part + 1


# ==================== ENGINEERING PROBLEMS ====================

def welded_beam_design(x: np.ndarray) -> float:
    """Welded beam design problem objective"""
    h, l, t, b = x[0], x[1], x[2], x[3]
    return 1.10471 * h**2 * l + 0.04811 * t * b * (14.0 + l)

def welded_beam_constraints(x: np.ndarray) -> float:
    """Welded beam design problem constraints (penalty)"""
    h, l, t, b = x[0], x[1], x[2], x[3]
    penalty = 0.0
    P, L = 6000, 14
    tau_prime = P / (np.sqrt(2) * h * l)
    M = P * (L + l / 2)
    R = np.sqrt(l**2 / 4 + ((h + t) / 2)**2)
    J = 2 * (h * l * np.sqrt(2) * (l**2 / 12 + ((h + t) / 2)**2))
    tau_double_prime = M * R / J
    tau = np.sqrt(tau_prime**2 + 2 * tau_prime * tau_double_prime * (l / (2 * R)) + tau_double_prime**2)
    if tau > 13600: penalty += (tau - 13600)**2
    sigma = 6 * P * L / (t**2 * b)
    if sigma > 30000: penalty += (sigma - 30000)**2
    if h > b: penalty += (h - b)**2 * 1000
    g4 = 0.1047 * h**2 + 0.04811 * t * b * (14.0 + l)
    if g4 > 5.0: penalty += (g4 - 5.0)**2 * 1000
    if h < 0.125: penalty += (0.125 - h)**2 * 1000
    E = 30e6
    delta = 4 * P * L**3 / (E * t**3 * b)
    if delta > 0.25: penalty += (delta - 0.25)**2
    return penalty

def tension_compression_spring(x: np.ndarray) -> float:
    """Tension/compression spring design problem objective"""
    d, D, N = x[0], x[1], x[2]
    return (N + 2) * D * d**2

def spring_constraints(x: np.ndarray) -> float:
    """Tension/compression spring design problem constraints (penalty)"""
    d, D, N = x[0], x[1], x[2]
    penalty = 0.0
    P, G = 1000, 11.5e6
    C = D / d
    K = (4 * C - 1) / (4 * C - 4) + 0.615 / C
    tau = 8 * K * P * D / (np.pi * d**3)
    if tau > 80000: penalty += (tau - 80000)**2
    delta = 8 * P * D**3 * N / (G * d**4)
    if delta > 0.25: penalty += (delta - 0.25)**2
    f = 1 / (2 * np.pi) * np.sqrt(G * d**4 / (8 * D**3 * N))
    if f < 100: penalty += (100 - f)**2
    if D <= d: penalty += (d - D)**2 * 1000
    return penalty


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    # Example 1: FFE-HOA
    print("\n[Example 1] FFE-HOA")
    print("-" * 60)
    np.random.seed(42)
    if TORCH_AVAILABLE:
        torch.manual_seed(42)

    hfoa_cnn = FFE_HOA(
        objective_func=sphere_function, dim=10,
        bounds=[(-5.12, 5.12)] * 10, population_size=50, max_iterations=100,
        alpha=0.2, beta=0.8, w=0.5, n_clusters=4,
        V_male=3.0, V_female=1.5, step_size_init=0.5,
        minimize=True, epsilon=1e-6, max_threshold_iterations=10,
        use_cnn=True, cnn_feature_dim=64, cnn_update_freq=5,
        cnn_learning_rate=1e-3, device='cpu'
    )
    best_sol, best_fit, history = hfoa_cnn.optimize(verbose=True)
    print(f"\nBest fitness (FFE-HOA): {best_fit:.10e}")
    print(f"Best solution (first 5 dims): {best_sol[:5]}")

    # Example 2: Compare with baseline (no CNN)
    print("\n" + "=" * 70)
    print("[Example 2] Comparison: HFOA (baseline)")
    print("-" * 60)
    np.random.seed(42)
    if TORCH_AVAILABLE:
        torch.manual_seed(42)

    hfoa_baseline = FFE_HOA(
        objective_func=sphere_function, dim=10,
        bounds=[(-5.12, 5.12)] * 10, population_size=50, max_iterations=100,
        alpha=0.6, beta=0.7, w=0.5, n_clusters=4,
        V_male=3.0, V_female=1.5, step_size_init=0.5,
        minimize=True, epsilon=1e-6, max_threshold_iterations=10,
        use_cnn=False,
        device='cpu'
    )
    best_sol_base, best_fit_base, history_base = hfoa_baseline.optimize(verbose=True)
    print(f"\nBest fitness (Baseline HFOA): {best_fit_base:.10e}")
    print(f"Best solution (first 5 dims): {best_sol_base[:5]}")

    print("\n" + "=" * 70)
    print("[Summary]")
    print(f"FFE-HOA Best Fitness: {best_fit:.10e}")
    print(f"Baseline HFOA Best Fitness:  {best_fit_base:.10e}")

    print("=" * 70)