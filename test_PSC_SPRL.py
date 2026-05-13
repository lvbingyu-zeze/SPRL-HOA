import math
import random
import matplotlib.pyplot as plt
import networkx as nx
import os
import numpy as np
from sklearn.cluster import SpectralClustering
from scipy.sparse.csgraph import connected_components
from collections import defaultdict, deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from datetime import datetime, timedelta

# ============================================================
# 1. Data Structures and Basic Definitions
# ============================================================

class Customer:
    """Customer class: contains location, demand, deadline, and dynamic status"""
    def __init__(self, customer_id, x, y, demand, deadline_hours):
        self.id = customer_id
        self.p = np.array([x, y], dtype=float)  # Location
        self.q = demand  # Demand quantity
        self.T_d = deadline_hours  # Deadline (hours)

        # Dynamic status
        self.a = 0  # Active status (0/1)
        self.order_time = None  # Order time
        self.accumulated_delay = 0.0  # Accumulated delay
        self.xi = 0.0  # Urgency

        # Historical data (for LSTM prediction)
        self.demand_history = deque(maxlen=20)  # Historical demand
        self.urgency_history = deque(maxlen=20)  # Historical urgency

    def place_order(self, current_time):
        """Customer places an order"""
        self.a = 1
        self.order_time = current_time
        self.accumulated_delay = 0.0

    def update(self, current_time, delivered=False):
        """Update customer status"""
        if self.a == 0:
            return

        if delivered:
            # Record history and reset
            self.demand_history.append(self.q)
            self.urgency_history.append(self.xi)
            self.a = 0
            self.accumulated_delay = 0.0
            return

        # Update accumulated delay
        elapsed = (current_time - self.order_time).total_seconds() / 3600.0
        if elapsed > self.T_d:
            self.accumulated_delay = elapsed - self.T_d
        else:
            self.accumulated_delay = 0.0

    def compute_urgency(self, current_time, w_a=0.6, w_b=0.4):
        """
        Compute urgency ξ_i(t) = w_a * ũ_i(t) + w_b * (1 - min(d̃_i(t)/24, 1))
        ũ_i(t): Normalized urgency based on deadline, increases as remaining time approaches 0
        d̃_i(t): Accumulated delay
        """
        if self.a == 0:
            self.xi = 0.0
            return 0.0

        elapsed = (current_time - self.order_time).total_seconds() / 3600.0
        remaining = max(0, self.T_d - elapsed)

        # Deadline urgency: less remaining time means more urgent
        u_tilde = 1.0 - min(remaining / self.T_d, 1.0) if self.T_d > 0 else 1.0

        # Delay penalty
        d_tilde = self.accumulated_delay
        delay_penalty = 1.0 - min(d_tilde / 24.0, 1.0)

        self.xi = w_a * u_tilde + w_b * delay_penalty
        return self.xi


class UAV:
    """UAV class"""
    def __init__(self, uav_id, warehouse_pos, max_payload=10, max_flight_time=2.0):
        self.id = uav_id
        self.p = np.array(warehouse_pos, dtype=float)  # 当前位置
        self.v = 50.0  # Linear velocity (km/h)
        self.omega = 1.0  # Angular velocity
        self.q_max = max_payload  # Maximum payload
        self.T_max = max_flight_time  # Maximum flight time (hours)
        self.warehouse = np.array(warehouse_pos, dtype=float)  # Assigned warehouse location
        self.assigned_customers = []  # Assigned customers
        self.current_load = 0.0
        self.total_distance = 0.0

    def reset(self):
        self.p = np.array(self.warehouse, dtype=float)
        self.assigned_customers = []
        self.current_load = 0.0
        self.total_distance = 0.0


class Warehouse:
    """Warehouse class"""
    def __init__(self, warehouse_id, x, y):
        self.id = warehouse_id
        self.p = np.array([x, y], dtype=float)
        self.uavs = []

    def add_uav(self, uav):
        self.uavs.append(uav)


class Obstacle:
    """Obstacle class"""
    def __init__(self, obs_id, x, y, radius):
        self.id = obs_id
        self.p = np.array([x, y], dtype=float)
        self.radius = radius


# ============================================================
# 2. LSTM Demand Prediction Network
# ============================================================

class DemandLSTM(nn.Module):
    """LSTM network to predict customer future demand patterns and urgency trends"""
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, output_size=2):
        super(DemandLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # Take the last time step
        out = self.fc(out)
        return out


class LSTMPredictor:
    """LSTM predictor wrapper class"""
    def __init__(self, model=None):
        self.model = model if model else DemandLSTM()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.training_data = []

    def predict(self, customer):
        """Predict customer future demand urgency"""
        if len(customer.demand_history) < 5:
            return customer.xi, customer.q

        # 构建序列
        seq = []
        for d, u in zip(customer.demand_history, customer.urgency_history):
            seq.append([d / 10.0, u])  # Normalization

        # Pad or truncate to length 10
        while len(seq) < 10:
            seq.insert(0, [0.0, 0.0])
        seq = seq[-10:]

        x = torch.FloatTensor(np.array(seq)).unsqueeze(0)  # (1, 10, 2)
        with torch.no_grad():
            pred = self.model(x)

        pred_demand = pred[0, 0].item() * 10.0
        pred_urgency = pred[0, 1].item()
        return pred_urgency, pred_demand

    def train_step(self, sequences, targets):
        """Training step"""
        if len(sequences) == 0:
            return 0.0

        x = torch.FloatTensor(np.array(sequences))
        y = torch.FloatTensor(np.array(targets))

        self.optimizer.zero_grad()
        pred = self.model(x)
        loss = self.criterion(pred, y)
        loss.backward()
        self.optimizer.step()

        return loss.item()


# ============================================================
# 3. Reinforcement Learning (PPO) Agent
# ============================================================

class ActorCritic(nn.Module):
    """Actor-Critic network for RL decision-making"""
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        # 共享特征提取层
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Actor: 输出动作概率
        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic: 输出状态价值
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        features = self.shared(state)
        action_probs = self.actor(features)
        value = self.critic(features)
        return action_probs, value


class PPOAgent:
    """PPO reinforcement learning agent"""
    def __init__(self, state_dim=10, action_dim=5):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device('cpu')

        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.0003)

        self.gamma = 0.99  # Discount factor
        self.eps_clip = 0.2  # PPO clipping parameter
        self.K_epochs = 4  # Number of update iterations per round

        self.memory = []  # Experience buffer

    def select_action(self, state):
        """Select action"""
        state = torch.FloatTensor(np.array(state)).unsqueeze(0)
        with torch.no_grad():
            action_probs, _ = self.policy(state)

        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob.item()

    def store_transition(self, state, action, reward, next_state, done, log_prob):
        """Store transition"""
        self.memory.append((state, action, reward, next_state, done, log_prob))

    def update(self):
        """Update policy"""
        if len(self.memory) < 10:
            return 0.0

        # Extract data
        states = torch.FloatTensor(np.array([t[0] for t in self.memory]))
        actions = torch.LongTensor([t[1] for t in self.memory])
        rewards = [t[2] for t in self.memory]
        next_states = torch.FloatTensor(np.array([t[3] for t in self.memory]))
        dones = [t[4] for t in self.memory]
        old_log_probs = torch.FloatTensor([t[5] for t in self.memory])

        # Compute returns
        returns = []
        G = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            if d:
                G = 0
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(np.array(returns))
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # 多次更新
        total_loss = 0
        for _ in range(self.K_epochs):
            action_probs, values = self.policy(states)
            dist = torch.distributions.Categorical(action_probs)
            log_probs = dist.log_prob(actions)

            # Compute advantages
            advantages = returns - values.squeeze().detach()

            # PPO loss
            ratios = torch.exp(log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic loss
            critic_loss = F.mse_loss(values.squeeze(), returns)

            loss = actor_loss + 0.5 * critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        self.memory = []
        return total_loss / self.K_epochs


# ============================================================
# 4. Priority-Aware Spectral Clustering (PSC)
# ============================================================

class PrioritySpectralClustering:
    """Priority-aware spectral clustering"""
    def __init__(self, alpha=0.4, beta=0.6, sigma=2000.0):
        self.alpha = alpha  # Spatial weight
        self.beta = beta    # Urgency权重
        self.sigma = sigma  # Gaussian kernel scale

    def compute_similarity_matrix(self, active_customers):
        """
        Compute similarity matrix W
        W_ij = exp(- (α·||p_i - p_j|| + β·|ξ_i - ξ_j|)² / (2σ²))
        """
        n = len(active_customers)
        W = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                ci = active_customers[i]
                cj = active_customers[j]

                # Spatial distance
                spatial_dist = np.linalg.norm(ci.p - cj.p)

                # Urgency差异
                urgency_diff = abs(ci.xi - cj.xi)

                # Combined distance
                combined = self.alpha * spatial_dist + self.beta * urgency_diff

                # Gaussian kernel similarity
                sim = math.exp(- (combined ** 2) / (2 * self.sigma ** 2))

                W[i, j] = sim
                W[j, i] = sim

        return W

    def cluster(self, active_customers, n_clusters=None):
        """Perform spectral clustering"""
        if len(active_customers) == 0:
            return {}, 0

        # Auto-determine cluster count: K = max(1, floor(|C^a(t)| / 5))
        if n_clusters is None:
            n_clusters = max(1, len(active_customers) // 5)

        n_clusters = min(n_clusters, len(active_customers))

        if n_clusters == 1 or len(active_customers) <= 2:
            return {0: [c.id for c in active_customers]}, 1

        W = self.compute_similarity_matrix(active_customers)

        # Use precomputed similarity matrix for spectral clustering
        sc = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            random_state=42,
            assign_labels='kmeans'
        )

        labels = sc.fit_predict(W)

        # Build partitions
        partitions = defaultdict(list)
        for idx, label in enumerate(labels):
            partitions[label].append(active_customers[idx].id)

        return dict(partitions), n_clusters


# ============================================================
# 5. Environment Simulation and Main System
# ============================================================

class DeliveryEnvironment:
    """Delivery environment simulator"""
    def __init__(self, customers, warehouses, obstacles, uavs_per_warehouse=3):
        self.customers = {c.id: c for c in customers}
        self.warehouses = warehouses
        self.obstacles = obstacles
        self.uavs = []
        self.current_time = datetime.now()

        # Initialize UAVs
        uav_id = 0
        for wh in warehouses:
            for _ in range(uavs_per_warehouse):
                uav = UAV(uav_id, wh.p, max_payload=10, max_flight_time=2.0)
                wh.add_uav(uav)
                self.uavs.append(uav)
                uav_id += 1

        # PSC clusterer
        self.psc = PrioritySpectralClustering(alpha=0.6, beta=0.4, sigma=100.0)

        # RL agent
        self.rl_agent = PPOAgent(state_dim=10, action_dim=5)

        # LSTM predictor
        self.lstm_predictor = LSTMPredictor()

        # Statistics
        self.total_deliveries = 0
        self.total_delay = 0.0
        self.episode_reward = 0.0

    def get_active_customers(self):
        """Get active customers"""
        return [c for c in self.customers.values() if c.a == 1]

    def get_state_for_rl(self):
        """Build RL state vector"""
        active = self.get_active_customers()

        state = np.zeros(10)
        if len(active) > 0:
            urgencies = [c.xi for c in active]
            demands = [c.q for c in active]

            state[0] = len(active) / 50.0  # Number of active customers
            state[1] = np.mean(urgencies) if urgencies else 0  # Average urgency
            state[2] = np.max(urgencies) if urgencies else 0   # Maximum urgency
            state[3] = np.std(urgencies) if len(urgencies) > 1 else 0  # Urgency标准差
            state[4] = np.mean(demands) if demands else 0  # Average demand
            state[5] = sum(demands) / (len(self.uavs) * 10.0)  # Load ratio
            state[6] = len([u for u in self.uavs if len(u.assigned_customers) == 0]) / len(self.uavs)  # Idle ratio

            # LSTM prediction features
            pred_urgencies = []
            for c in active[:3]:  # 取前3个
                pred_u, _ = self.lstm_predictor.predict(c)
                pred_urgencies.append(pred_u)
            state[7] = np.mean(pred_urgencies) if pred_urgencies else 0
            state[8] = self.total_delay / max(self.total_deliveries, 1)
            state[9] = self.current_time.hour / 24.0  # Time feature

        return state

    def apply_rl_action(self, action):
        """Apply RL action to adjust parameters"""
        # action 0-4 correspond to different (alpha, beta, sigma) combinations
        configs = [
            (0.7, 0.3, 80.0),   # Focus on spatial
            (0.6, 0.4, 100.0),  # Balanced
            (0.5, 0.5, 120.0),  # More balanced
            (0.4, 0.6, 150.0),  # Focus on urgency
            (0.3, 0.7, 200.0),  # Strong focus on urgency
        ]
        alpha, beta, sigma = configs[action]
        self.psc.alpha = alpha
        self.psc.beta = beta
        self.psc.sigma = sigma
        return alpha, beta, sigma

    def assign_clusters_to_uavs(self, partitions):
        """Assign cluster results to UAVs - considering nearest warehouse"""
        active = self.get_active_customers()
        active_dict = {c.id: c for c in active}

        # Reset UAVs
        for uav in self.uavs:
            uav.reset()

        # Find nearest warehouse and UAV for each cluster
        uav_idx = 0
        for cluster_id, customer_ids in partitions.items():
            if uav_idx >= len(self.uavs):
                break

            # Compute cluster center
            cluster_customers = [active_dict[cid] for cid in customer_ids if cid in active_dict]
            if len(cluster_customers) == 0:
                continue

            cluster_center = np.mean([c.p for c in cluster_customers], axis=0)

            # Find nearest warehouse
            nearest_wh = min(self.warehouses, key=lambda wh: np.linalg.norm(wh.p - cluster_center))

            # Find nearest idle UAV at that warehouse
            available_uavs = [u for u in self.uavs if u.id >= nearest_wh.id * 3 and u.id < (nearest_wh.id + 1) * 3 and len(u.assigned_customers) == 0]
            if len(available_uavs) == 0:
                # Use any idle UAV
                available_uavs = [u for u in self.uavs if len(u.assigned_customers) == 0]

            if len(available_uavs) == 0:
                continue

            uav = min(available_uavs, key=lambda u: np.linalg.norm(u.warehouse - cluster_center))

            # Sort customers by urgency, prioritize urgent deliveries
            cluster_customers.sort(key=lambda c: c.xi, reverse=True)

            total_demand = 0
            for c in cluster_customers:
                if total_demand + c.q <= uav.q_max:
                    # Check if flight time is feasible
                    route = [uav.warehouse] + [cc.p for cc in uav.assigned_customers + [c]] + [uav.warehouse]
                    total_dist = sum(np.linalg.norm(route[i] - route[i+1]) for i in range(len(route)-1))
                    flight_time = total_dist / uav.v

                    if flight_time <= uav.T_max:
                        uav.assigned_customers.append(c)
                        total_demand += c.q
                    else:
                        break  # Cannot add more

            uav.current_load = total_demand

    def compute_reward(self):
        """Compute reward function"""
        active = self.get_active_customers()

        if len(active) == 0:
            return 10.0  # All completed, large positive reward

        # Negative reward: sum of urgencies of undelivered customers
        urgency_penalty = sum(c.xi for c in active) * 0.00001

        # Negative reward: load imbalance
        loads = [u.current_load for u in self.uavs if len(u.assigned_customers) > 0]
        if len(loads) > 1:
            balance_penalty = np.std(loads) * 100
        else:
            balance_penalty = 0

        # Delay penalty
        delay_penalty = self.total_delay * 0.01

        reward = -urgency_penalty - balance_penalty - delay_penalty
        return reward / 10.0  # Scale

    def simulate_delivery(self, partitions):
        """Simulate delivery process"""
        active = self.get_active_customers()
        active_dict = {c.id: c for c in active}

        delivered = []

        for uav in self.uavs:
            if len(uav.assigned_customers) == 0:
                continue

            # Nearest-neighbor TSP path planning
            route = [uav.warehouse]
            unvisited = [c for c in uav.assigned_customers]

            while unvisited:
                current = route[-1]
                nearest = min(unvisited, key=lambda c: np.linalg.norm(current - c.p))
                route.append(nearest.p)
                unvisited.remove(nearest)

            route.append(uav.warehouse)

            # 计算飞行时间
            total_dist = 0
            for i in range(len(route) - 1):
                total_dist += np.linalg.norm(route[i] - route[i+1])
            flight_time = total_dist / uav.v  # 小时

            # Check if overtime
            if flight_time <= uav.T_max:
                for c in uav.assigned_customers:
                    c.update(self.current_time, delivered=True)
                    self.total_deliveries += 1
                    delivered.append(c.id)
            else:
                # Partial delivery: deliver in route order until overtime
                partial_dist = 0
                for i in range(1, len(route) - 1):
                    partial_dist += np.linalg.norm(route[i-1] - route[i])
                    partial_time = partial_dist / uav.v + np.linalg.norm(route[i] - uav.warehouse) / uav.v

                    if partial_time <= uav.T_max:
                        # 找到对应的客户
                        for c in uav.assigned_customers:
                            if np.allclose(c.p, route[i]):
                                c.update(self.current_time, delivered=True)
                                self.total_deliveries += 1
                                delivered.append(c.id)
                                break
                    else:
                        # Remaining customers delayed
                        for c in uav.assigned_customers:
                            if c.id not in delivered:
                                c.accumulated_delay += 0.5  # Half-hour delay
                                self.total_delay += 0.5
                        break

        return delivered

    def step(self, generate_orders=True):
        """Environment single-step advance"""
        # 1. 生成新订单（模拟）
        if generate_orders:
            for c in self.customers.values():
                if c.a == 0 and random.random() < 0.3:  # 30% probability of placing order
                    c.place_order(self.current_time)

        # 2. 更新所有客户状态
        for c in self.customers.values():
            c.compute_urgency(self.current_time)
            c.update(self.current_time)

        # 3. 获取RL状态并选择动作
        state = self.get_state_for_rl()
        action, log_prob = self.rl_agent.select_action(state)
        alpha, beta, sigma = self.apply_rl_action(action)

        # 4. 执行PSC聚类
        active = self.get_active_customers()
        partitions, k = self.psc.cluster(active)

        # 5. 分配给无人机
        self.assign_clusters_to_uavs(partitions)

        # 6. 模拟配送
        delivered = self.simulate_delivery(partitions)

        # 7. 计算奖励
        reward = self.compute_reward()
        self.episode_reward += reward

        # 8. 存储转移并更新RL
        next_state = self.get_state_for_rl()
        done = len(self.get_active_customers()) == 0
        self.rl_agent.store_transition(state, action, reward, next_state, done, log_prob)

        if len(self.rl_agent.memory) >= 20:
            loss = self.rl_agent.update()

        # 9. 更新时间
        self.current_time += timedelta(minutes=30)

        return {
            'active_customers': len(active),
            'clusters': k,
            'delivered': len(delivered),
            'alpha': alpha,
            'beta': beta,
            'sigma': sigma,
            'reward': reward,
            'partitions': partitions
        }

    def run_episode(self, max_steps=50, visualize=True):
        """Run a complete episode"""
        print("=" * 60)
        print("Starting Delivery Simulation Episode")
        print("=" * 60)

        history = []

        for step in range(max_steps):
            info = self.step()
            history.append(info)

            if step % 5 == 0:
                print(f"Step {step}: Active Customers={info['active_customers']}, "
                      f"Clusters={info['clusters']}, Delivered={info['delivered']}, "
                      f"α={info['alpha']:.2f}, β={info['beta']:.2f}, σ={info['sigma']:.1f}, "
                      f"Reward={info['reward']:.3f}")

            if info['active_customers'] == 0 and step > 10:
                print(f"All orders processed, ending early at step {step}")
                break

        # Train LSTM
        self._train_lstm()

        if visualize:
            self.visualize_results(history)

        return history

    def _train_lstm(self):
        """Train LSTM predictor"""
        sequences = []
        targets = []

        for c in self.customers.values():
            if len(c.demand_history) >= 5:
                seq = []
                for d, u in zip(c.demand_history, c.urgency_history):
                    seq.append([d / 10.0, u])
                while len(seq) < 10:
                    seq.insert(0, [0.0, 0.0])
                sequences.append(seq[-10:])

                # 目标：预测下一个需求紧急度
                if len(c.demand_history) > 0:
                    targets.append([
                        c.demand_history[-1] / 10.0,
                        c.urgency_history[-1]
                    ])

        if len(sequences) > 0:
            loss = self.lstm_predictor.train_step(sequences, targets)
            print(f"LSTM Training Loss: {loss:.4f}")

    def visualize_results(self, history):
        """Visualize simulation results"""
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        steps = range(len(history))

        # 1. 活跃客户数
        ax1 = axes[0, 0]
        ax1.plot(steps, [h['active_customers'] for h in history], 'b-o', markersize=3)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Active Customers')
        ax1.set_title('Active Customers Over Time')
        ax1.grid(True)

        # 2. 聚类数
        ax2 = axes[0, 1]
        ax2.plot(steps, [h['clusters'] for h in history], 'g-s', markersize=3)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Number of Clusters')
        ax2.set_title('Cluster Count (K) Over Time')
        ax2.grid(True)

        # 3. 每次配送数
        ax3 = axes[0, 2]
        ax3.plot(steps, [h['delivered'] for h in history], 'r-^', markersize=3)
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Deliveries')
        ax3.set_title('Deliveries Per Step')
        ax3.grid(True)

        # 4. RL参数变化
        ax4 = axes[1, 0]
        ax4.plot(steps, [h['alpha'] for h in history], 'b-', label='α (spatial)', alpha=0.7)
        ax4.plot(steps, [h['beta'] for h in history], 'r-', label='β (urgency)', alpha=0.7)
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Weight Value')
        ax4.set_title('RL-Adjusted Weights')
        ax4.legend()
        ax4.grid(True)

        # 5. 奖励曲线
        ax5 = axes[1, 1]
        rewards = [h['reward'] for h in history]
        ax5.plot(steps, rewards, 'm-', alpha=0.7)
        ax5.plot(steps, np.cumsum(rewards) / (np.arange(len(rewards)) + 1), 'k--', label='Cumulative Avg')
        ax5.set_xlabel('Step')
        ax5.set_ylabel('Reward')
        ax5.set_title('RL Reward Over Time')
        ax5.legend()
        ax5.grid(True)

        # 6. 最终聚类可视化
        ax6 = axes[1, 2]
        self._draw_cluster_visualization(ax6, history[-1]['partitions'] if history else {})
        ax6.set_title('Final Cluster Assignment')

        plt.tight_layout()
        plt.savefig('rl_psc_results.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Visualization results saved")

    def _draw_cluster_visualization(self, ax, partitions):
        """Draw clustering results"""
        active = self.get_active_customers()
        if len(active) == 0:
            ax.text(0.5, 0.5, 'No Active Customers', ha='center', va='center', transform=ax.transAxes)
            return

        colors = plt.cm.tab20(np.linspace(0, 1, max(len(partitions), 1)))

        # 绘制仓库
        for wh in self.warehouses:
            ax.scatter(wh.p[0], wh.p[1], s=300, marker='s', c='black', label='Warehouse')

        # 绘制障碍物
        for obs in self.obstacles:
            circle = plt.Circle(obs.p, obs.radius, color='gray', alpha=0.3)
            ax.add_patch(circle)

        # 绘制客户
        active_dict = {c.id: c for c in active}
        for i, (cluster_id, cids) in enumerate(partitions.items()):
            xs, ys, us = [], [], []
            for cid in cids:
                if cid in active_dict:
                    c = active_dict[cid]
                    xs.append(c.p[0])
                    ys.append(c.p[1])
                    us.append(c.xi)

            if len(xs) > 0:
                scatter = ax.scatter(xs, ys, s=[50 + u * 200 for u in us],
                                   c=[colors[i]], alpha=0.7,
                                   label=f'Cluster {cluster_id} ({len(xs)})')

        ax.legend(loc='upper left', fontsize=8)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')


# ============================================================
# 6. Data Reading (Compatible with Original TSP Format)
# ============================================================

def read_tsp_data(filename):
    """Read TSP data file, return city coordinates dictionary"""
    cities = {}
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                try:
                    city_id = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    cities[city_id] = (x, y)
                except ValueError:
                    continue
    return cities


def create_synthetic_data(n_customers=50, n_warehouses=3, n_obstacles=5, seed=42):
    """Create synthetic data for testing"""
    random.seed(seed)
    np.random.seed(seed)

    # 客户
    customers = []
    for i in range(n_customers):
        x = random.uniform(0, 1000)
        y = random.uniform(0, 1000)
        demand = random.uniform(1, 5)
        deadline = random.uniform(1, 4)
        customers.append(Customer(i, x, y, demand, deadline))

    # 仓库
    warehouses = []
    for i in range(n_warehouses):
        x = random.uniform(100, 900)
        y = random.uniform(100, 900)
        warehouses.append(Warehouse(i, x, y))

    # 障碍物
    obstacles = []
    for i in range(n_obstacles):
        x = random.uniform(200, 800)
        y = random.uniform(200, 800)
        radius = random.uniform(30, 80)
        obstacles.append(Obstacle(i, x, y, radius))

    return customers, warehouses, obstacles


# ============================================================
# 7. Main Program
# ============================================================

def main():
    print("=" * 60)
    print("Priority-Aware Spectral Clustering + RL + LSTM UAV Delivery System")
    print("=" * 60)

    # 创建数据
    customers, warehouses, obstacles = create_synthetic_data(
        n_customers=50, n_warehouses=3, n_obstacles=5, seed=42
    )

    print(f"\n数据集信息:")
    print(f"  客户数: {len(customers)}")
    print(f"  仓库数: {len(warehouses)}")
    print(f"  障碍物数: {len(obstacles)}")

    # 创建环境
    env = DeliveryEnvironment(customers, warehouses, obstacles, uavs_per_warehouse=3)
    print(f"  Total UAVs: {len(env.uavs)}")

    # 运行仿真
    history = env.run_episode(max_steps=50, visualize=True)

    # 输出统计
    print("\n" + "=" * 60)
    print("仿真统计")
    print("=" * 60)
    print(f"Total Deliveries: {env.total_deliveries}")
    print(f"Total Delay: {env.total_delay:.2f} hours")
    print(f"Cumulative Reward: {env.episode_reward:.2f}")
    print(f"Final Active Customers: {len(env.get_active_customers())}")

    # 多次训练RL
    print("\n" + "=" * 60)
    print("RL训练阶段 (10 episodes)")
    print("=" * 60)

    # Save agent for cross-episode training
    shared_agent = env.rl_agent

    for ep in range(10):
        # 重置环境
        customers, warehouses, obstacles = create_synthetic_data(
            n_customers=50, n_warehouses=3, n_obstacles=5, seed=ep
        )
        env = DeliveryEnvironment(customers, warehouses, obstacles, uavs_per_warehouse=3)
        # Reuse trained RL agent
        env.rl_agent = shared_agent

        history = env.run_episode(max_steps=50, visualize=False)
        print(f"Episode {ep+1}: Total Deliveries={env.total_deliveries}, Reward={env.episode_reward:.2f}")


if __name__ == "__main__":
    main()