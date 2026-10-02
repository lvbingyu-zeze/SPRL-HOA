#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PSC-SPRL: Priority-aware Spectral Clustering + State-Predictive RL
===================================================================
动态物流场景下的无人机订单分配（论文 Sec. 4.2, Algorithm 1）。

组件:
  1. Customer Priority : Eq.(srl3)  客户紧迫度  xi_i(t)
  2. PSC               : Eq.(srl1)(srl2)  优先级感知谱聚类（高斯核相似度 + 归一化拉普拉斯）
  3. SPRL              : Eq.(srl4) 奖励 + LSTM 增广状态 s'_t=[s_t,h_t] + 策略梯度(REINFORCE)

实验参数（Sec 5.1, Table I 及参数研究的最优配置）:
  - Att48 数据集: 1 仓库 + 47 客户, 初始 |C^a| = 25
  - 动态事件间隔 Δt = 100 s, 每事件新订单强度 5/10/15/20, 延迟订单随机取消
  - 10 架异构无人机: 5-10 kg (4,8,9), 11-20 kg (1,2,3,6), 21-30 kg (5,7,10)
  - PSC:  α=0.4, β=0.6, σ=0.5, K = max(1, |C^a|/5)     (Fig. ab / Table sigma)
  - 奖励权重: α1=1e-5, α2=100, α3=0.01, q*=0.8          (Table w1)
  - LSTM 历史窗口 L=80                                    (Table C2)
  - 综合成本 W1,W2,W3 = 0.3, 0.4, 0.3                    (Table I, Eq.(of))
  - 折扣因子 γ=0.99, T_max=100(事件数), T'_max=200(训练迭代)

依赖: 仅 numpy（LSTM/策略网络/截断BPTT 全部手写实现）。
用法:
  python psc_sprl.py                      # 论文默认参数（Att48, 强度5）
  python psc_sprl.py --episodes 60        # 缩短训练
  python psc_sprl.py --psc-only           # PSC-only 基线（Table C2 对比）
  python psc_sprl.py --intensity 10       # 订单强度 5/10/15/20
"""

import argparse
import numpy as np

# ----------------------------------------------------------------------
# Att48 数据集（48 个节点，坐标单位: 米；节点 0 为仓库，其余 47 个为客户）
# ----------------------------------------------------------------------
ATT48 = np.array([
    [6734, 1453], [2233, 10], [5530, 1424], [401, 841], [3082, 1644],
    [1608, 3327], [4935, 107], [879, 1117], [188, 1274], [3602, 331],
    [319, 1750], [4934, 1177], [6937, 1257], [429, 1653], [4826, 1696],
    [186, 1838], [2680, 1401], [4742, 34], [378, 1564], [2222, 291],
    [6586, 1000], [1835, 951], [3996, 160], [3227, 1387], [3176, 1058],
    [4141, 670], [4957, 1415], [5139, 80], [328, 1711], [3859, 1259],
    [4346, 713], [4074, 34], [4348, 1815], [4608, 328], [4344, 1864],
    [3963, 1995], [4860, 1174], [475, 913], [4545, 747], [3582, 2024],
    [4420, 1648], [4936, 143], [601, 950], [3289, 288], [3315, 123],
    [4463, 1428], [3057, 1829], [4444, 2011],
], dtype=float)

# ----------------------------------------------------------------------
# 全局参数（对应论文 Sec 5.1）
# ----------------------------------------------------------------------
PARAMS = dict(
    dt_event=100.0,        # Δt = 100 s 动态事件间隔
    deadline_lo=1800.0,    # 订单截止期下界 (s)
    deadline_hi=7200.0,    # 订单截止期上界 (s)
    urgency_horizon=7200.0,# ũ_i(t) 的归一化视界 H (s)
    w_a=0.5, w_b=0.5,      # Eq.(srl3) 紧迫度权重（论文未给出，取对称默认，可调）
    alpha_psc=0.4,         # Eq.(srl2) α（Fig.ab 最优）
    beta_psc=0.6,          # Eq.(srl2) β = 1 - α
    sigma=0.5,             # Eq.(srl2) σ（Table sigma 最优）
    k_div=5,               # Eq.(srl1): K = max(1, |C^a|/5)
    k_max=10,              # 聚类数上限（动作空间按固定维度构造）
    a1=1e-5, a2=100.0, a3=0.01,   # Eq.(srl4) 奖励权重（Table I / Table w1）
    q_star=0.8,            # 目标载荷利用率 q*
    W=(0.3, 0.4, 0.3),     # 综合成本 W1,W2,W3（Eq.(of)）
    e_rate=1.4,            # 能量成本系数 F_e = e_rate * 距离（与距离隐式相关）
    gamma=0.99,            # MDP 折扣因子
    L=80,                  # LSTM 历史窗口（Table C2 最优）
    T_max=100,             # 每回合事件数
    T_prime=200,           # T'_max 最大训练迭代（Algorithm 1）
    initial_active=25,     # 初始 |C^a|（Att48）
    uav_caps=(15, 18, 12, 8, 25, 16, 28, 6, 9, 22),  # 按论文 5-10/11-20/21-30 kg 分类
    cancel_prob=0.15,      # 延迟订单每事件被取消的概率
    reorder_cooldown=4,    # 客户再次下单所需的事件冷却数
    v_uav=10.0,            # UAV 巡航速度 v_u ∈ 4-20 m/s（取中值）
    endurance=(4000.0, 4800.0),  # T_u^max 单次任务续航 (s)
)

ACTIVE, SERVED, CANCELED = 0, 1, 2


# ----------------------------------------------------------------------
# 订单与无人机
# ----------------------------------------------------------------------
class Order:
    __slots__ = ("cid", "pos", "payload", "t_create", "deadline", "status", "cluster")

    def __init__(self, cid, pos, payload, t_create, deadline):
        self.cid, self.pos, self.payload = cid, pos, payload
        self.t_create, self.deadline = t_create, deadline
        self.status, self.cluster = ACTIVE, 0


class UAV:
    __slots__ = ("idx", "cap", "pos")

    def __init__(self, idx, cap, home):
        self.idx, self.cap = idx, cap
        self.pos = home.copy()


# ----------------------------------------------------------------------
# 动态物流环境
# ----------------------------------------------------------------------
class DynamicEnv:
    """Att48 + 随机事件（每 Δt=100s: 新订单到达 / 延迟订单取消）的滚动时域仿真。"""

    def __init__(self, coords=ATT48, intensity=5, initial_active=25, seed=0):
        self.rng = np.random.default_rng(seed)
        self.coords = coords.astype(float)
        self.home = coords[0].copy()
        self.n_cust = len(coords) - 1
        self.intensity = intensity
        self.initial_active = initial_active
        self.uavs = [UAV(i + 1, float(c), self.home)
                     for i, c in enumerate(PARAMS["uav_caps"])]
        self.scale = np.abs(coords).max()
        self.reset()

    # ---------- 生命周期 ----------
    def reset(self):
        self.t = 0.0
        self.orders = []
        self.last_order_event = np.full(self.n_cust, -PARAMS["reorder_cooldown"])
        self.total_waiting = 0.0     # 已服务订单累计等待（分钟）
        self.n_served = 0
        self.n_generated = 0
        self._spawn(self.initial_active, initial=True)
        return self.state_features(1, np.zeros(len(self.orders), int))

    def _spawn(self, k, initial=False):
        """从无活跃订单的客户中生成 k 个新订单（强度 5/10/15/20）。"""
        pool = [c for c in range(self.n_cust)
                if self.last_order_event[c] <= self._event_id() - PARAMS["reorder_cooldown"]
                and not any(o.cid == c and o.status == ACTIVE for o in self.orders)]
        if not pool:
            return
        chosen = self.rng.choice(pool, size=min(k, len(pool)), replace=False)
        for c in chosen:
            payload = float(self.rng.uniform(1.0, 8.0))
            dl = self.rng.uniform(PARAMS["deadline_lo"], PARAMS["deadline_hi"])
            self.orders.append(Order(c, self.coords[c + 1].copy(), payload, self.t, self.t + dl))
            self.last_order_event[c] = self._event_id()
            self.n_generated += 1

    def _event_id(self):
        return int(round(self.t / PARAMS["dt_event"]))

    def active_orders(self):
        return [o for o in self.orders if o.status == ACTIVE]

    # ---------- 动态事件（每 Δt=100s） ----------
    def step_event(self):
        self.t += PARAMS["dt_event"]
        self._spawn(self.intensity)
        # 延迟订单随机取消（论文: a subset of active orders is randomly canceled due to delays）
        for o in self.active_orders():
            if self.t > o.deadline and self.rng.random() < PARAMS["cancel_prob"]:
                o.status = CANCELED

    # ---------- 紧迫度 Eq.(srl3) ----------
    def urgency(self, o):
        rem = max(0.0, o.deadline - self.t)
        u_tilde = 1.0 - min(rem / PARAMS["urgency_horizon"], 1.0)   # 剩余时间→0 时趋近 1
        delay_h = max(0.0, self.t - o.deadline) / 3600.0            # 累计延迟（小时）
        return (PARAMS["w_a"] * u_tilde
                + PARAMS["w_b"] * (1.0 - min(delay_h / 24.0, 1.0)))

    # ---------- 成本项 ----------
    def tour_length(self, uav, cluster_ids):
        """UAV 从当前位置出发、按最近邻访问簇内订单的路径长度 F_d(τ_u)。"""
        pts = [(o.pos, o.payload) for o in self.active_orders() if o.cluster in cluster_ids]
        cur, dist = uav.pos.copy(), 0.0
        while pts:
            d = [np.linalg.norm(p - cur) for p, _ in pts]
            j = int(np.argmin(d))
            dist += d[j]
            cur = pts[j][0]
            pts.pop(j)
        return dist

    def _cluster_loads(self, uav_cluster):
        """每簇活跃订单总载荷 cp[c]，及每簇 UAV 数 nu[c]（载荷均摊给簇内 UAV）。"""
        cp, nu = {}, {}
        for c in uav_cluster.values():
            nu[c] = nu.get(c, 0) + 1
        for o in self.active_orders():
            cp[o.cluster] = cp.get(o.cluster, 0.0) + o.payload
        return cp, nu

    def costs(self, uav_cluster):
        """返回 (F_d, F_w, F_e, 载荷利用偏差)，按当前分配计算。"""
        F_d = F_w = util_pen = 0.0
        cp, nu = self._cluster_loads(uav_cluster)
        for u in self.uavs:
            if u.idx not in uav_cluster:
                continue                      # Algorithm 1: 仅前 min(|U|,K) 架参与
            c = uav_cluster[u.idx]
            F_d += self.tour_length(u, {c})
            load = cp.get(c, 0.0) / nu[c]     # 簇载荷均摊
            util_pen += (load / u.cap - PARAMS["q_star"]) ** 2
        for o in self.active_orders():
            F_w += (self.t - o.t_create) / 60.0        # 等待时间（分钟）
        F_e = PARAMS["e_rate"] * F_d                    # 能量与距离隐式相关
        return F_d, F_w, F_e, util_pen

    def comprehensive_cost(self, F_d, F_w, F_e):
        """综合成本 F = W1·F_d + W2·F_e + W3·F_w (Eq.(of), W=0.3/0.4/0.3)。"""
        w1, w2, w3 = PARAMS["W"]
        return w1 * F_d + w2 * F_e + w3 * F_w

    def reward(self, uav_cluster):
        """Eq.(srl4): R = -(α1·ΣF_d + α2·Σ(q_u/q_max - q*)² + α3·ΣF_w)。"""
        F_d, F_w, _, util_pen = self.costs(uav_cluster)
        return -(PARAMS["a1"] * F_d + PARAMS["a2"] * util_pen + PARAMS["a3"] * F_w)

    # ---------- 服务（事件末执行） ----------
    def serve(self, uav_cluster):
        """每架 UAV 执行一次配送任务：紧迫度优先 + 载荷/续航(T_u^max)双重约束。
        订单需在创建后的事件才可被服务（UAV 需要飞行时间），由此产生等待/延迟。"""
        v = PARAMS["v_uav"]
        for u in self.uavs:
            if u.idx not in uav_cluster:
                continue                      # 未指派的 UAV 不执行任务
            endurance = PARAMS["endurance"][0] + (u.idx - 1) * (
                (PARAMS["endurance"][1] - PARAMS["endurance"][0]) / 9.0)  # 4000-4800 s
            cap_left, t_trip = u.cap, 0.0
            cur = u.pos.copy()
            mine = [o for o in self.active_orders()
                    if o.cluster == uav_cluster[u.idx] and o.t_create < self.t]
            mine.sort(key=lambda o: -self.urgency(o))          # 紧迫者优先
            for o in mine:
                leg = float(np.linalg.norm(o.pos - cur))
                if o.payload <= cap_left and t_trip + leg / v <= endurance:
                    o.status = SERVED
                    cap_left -= o.payload
                    t_trip += leg / v
                    self.total_waiting += (self.t - o.t_create) / 60.0
                    self.n_served += 1
                    cur = o.pos.copy()
                    u.pos = o.pos.copy()
            u.pos = self.home.copy()                            # 返回仓库，容量恢复

    # ---------- 观测特征（固定维度，供 LSTM/策略使用） ----------
    def state_features(self, K, labels, uav_cluster=None):
        uav_cluster = uav_cluster or {}
        act = self.active_orders()
        n = len(act)
        cp, nu = self._cluster_loads(uav_cluster)
        f = []
        for u in self.uavs:  # S^u: 无人机位置 + 载荷利用
            c = uav_cluster.get(u.idx, -1)
            load = cp.get(c, 0.0) / nu[c] if c in nu else 0.0
            f += [u.pos[0] / self.scale, u.pos[1] / self.scale, load / u.cap]
        urg = np.array([self.urgency(o) for o in act]) if n else np.zeros(1)
        for c in range(PARAMS["k_max"]):  # S^c: 簇级特征（质心/规模，不足补零）
            if c < K:
                pts = np.array([o.pos for o in act if o.cluster == c]) if n else np.zeros((0, 2))
                if len(pts):
                    f += [pts[:, 0].mean() / self.scale, pts[:, 1].mean() / self.scale, len(pts) / 50.0]
                else:
                    f += [0.0, 0.0, 0.0]
            else:
                f += [0.0, 0.0, 0.0]
        delay = np.array([min(max(0.0, self.t - o.deadline) / 3600.0 / 24.0, 1.0) for o in act]) if n else np.zeros(1)
        f += [n / 50.0, urg.mean(), (1 - urg.mean()), delay.mean(), self.t / (PARAMS["T_max"] * PARAMS["dt_event"])]
        return np.array(f, dtype=np.float64)

    # ---------- 动作空间: Assign(u,c) | Reassign(i,c) | Noop ----------
    N_MAX = 50  # 动作维度按最大活跃订单数构造

    def n_actions(self):
        return 10 * PARAMS["k_max"] + self.N_MAX * PARAMS["k_max"] + 1

    def action_mask(self, K, n_act):
        m = np.zeros(self.n_actions(), dtype=bool)
        m[:10 * PARAMS["k_max"]] = False
        for u in range(10):
            m[u * PARAMS["k_max"]: u * PARAMS["k_max"] + K] = True          # Assign
        base = 10 * PARAMS["k_max"]
        for i in range(min(n_act, self.N_MAX)):
            m[base + i * PARAMS["k_max"]: base + i * PARAMS["k_max"] + K] = True  # Reassign
        m[-1] = True                                                         # Noop
        return m

    def execute(self, a, K, n_act):
        """执行动作并返回描述。"""
        if a == self.n_actions() - 1:
            return "Noop"
        if a < 10 * PARAMS["k_max"]:
            u, c = a // PARAMS["k_max"], a % PARAMS["k_max"]
            return ("Assign", u, c)
        a -= 10 * PARAMS["k_max"]
        i, c = a // PARAMS["k_max"], a % PARAMS["k_max"]
        return ("Reassign", i, c)


# ----------------------------------------------------------------------
# PSC: 优先级感知谱聚类（Eq. srl1 / srl2）
# ----------------------------------------------------------------------
def kmeans(X, k, rng, iters=50):
    n = len(X)
    if k <= 1 or n == 0:
        return np.zeros(n, dtype=int)
    k = min(k, n)
    # k-means++ 初始化
    centers = [X[rng.integers(n)]]
    for _ in range(k - 1):
        d2 = np.min([((X - c) ** 2).sum(1) for c in centers], axis=0)
        probs = d2 / (d2.sum() + 1e-12)
        centers.append(X[rng.choice(n, p=probs)])
    C = np.array(centers)
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(2)
        new = d.argmin(1)
        if (new == labels).all():
            break
        labels = new
        for j in range(k):
            if (labels == j).any():
                C[j] = X[labels == j].mean(0)
    return labels


def priority_spectral_clustering(positions, urgency, rng,
                                  alpha=PARAMS["alpha_psc"],
                                  beta=PARAMS["beta_psc"],
                                  sigma=PARAMS["sigma"],
                                  k_div=PARAMS["k_div"]):
    """Eq.(srl1): K = max(1, |C^a|/5)；Eq.(srl2): 高斯核相似度；谱聚类。"""
    n = len(positions)
    K = min(max(1, n // k_div), PARAMS["k_max"])           # Eq.(srl1)
    if n <= 1:
        return np.zeros(n, dtype=int), max(K, 1)
    # 成对相似度（距离归一化到 [0,1]，与 |Δξ| 可比）
    diff = positions[:, None, :] - positions[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    dnorm = dist / (dist.max() + 1e-12)
    s = alpha * dnorm + beta * np.abs(urgency[:, None] - urgency[None, :])  # Eq.(srl2)
    W = np.exp(-(s ** 2) / (2.0 * sigma ** 2))
    np.fill_diagonal(W, 0.0)
    # 度矩阵 / 拉普拉斯
    d = W.sum(1)
    Dm05 = np.diag(1.0 / np.sqrt(d + 1e-12))
    L_sym = np.eye(n) - Dm05 @ W @ Dm05                     # 对称归一化拉普拉斯
    # 取 K 个最小特征值对应特征向量，行归一化后 k-means
    _, vecs = np.linalg.eigh(L_sym)
    U = vecs[:, :K]
    U = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12)
    labels = kmeans(U, K, rng)
    return labels, K


# ----------------------------------------------------------------------
# SPRL: LSTM 增广状态 + 策略梯度（纯 NumPy，含截断 BPTT）
# ----------------------------------------------------------------------
class Adam:
    def __init__(self, params, lr=3e-4):
        self.p, self.lr = params, lr
        self.m = [np.zeros_like(x) for x in params]
        self.v = [np.zeros_like(x) for x in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for p, g, m, v in zip(self.p, grads, self.m, self.v):
            m *= 0.9; m += 0.1 * g
            v *= 0.999; v += 0.001 * g * g
            mh = m / (1 - 0.9 ** self.t)
            vh = v / (1 - 0.999 ** self.t)
            p += self.lr * mh / (np.sqrt(vh) + 1e-8)


class SPRLPolicy:
    """
    策略 π_θ(a_t | s'_t), s'_t = [s_t, h_t]：
      - LSTM 编码最近 L=80 个历史状态 → h_t（近似 POMDP 的信念状态）
      - MLP 输出动作分布，REINFORCE 更新（∇J = E[Q ∇log π]）
    """

    def __init__(self, state_dim, n_actions, hidden=32, mlp_hidden=128,
                 L=PARAMS["L"], lr=3e-4, seed=0):
        rng = np.random.default_rng(seed)
        self.L, self.nA = L, n_actions
        self.H = hidden
        # LSTM 参数（门顺序: i, f, o, g）
        s = 0.1
        self.Wx = rng.normal(0, s, (state_dim, 4 * hidden))
        self.Wh = rng.normal(0, s, (hidden, 4 * hidden))
        self.b = np.zeros(4 * hidden); self.b[hidden:2 * hidden] = 1.0
        # 策略 MLP
        d_in = state_dim + hidden
        self.W1 = rng.normal(0, np.sqrt(2.0 / d_in), (d_in, mlp_hidden))
        self.b1 = np.zeros(mlp_hidden)
        self.W2 = rng.normal(0, np.sqrt(2.0 / mlp_hidden), (mlp_hidden, n_actions))
        self.b2 = np.zeros(n_actions)
        self.params = [self.Wx, self.Wh, self.b, self.W1, self.b1, self.W2, self.b2]
        self.opt = Adam(self.params, lr)
        self.baseline = {}     # 按时间步的 EMA 基线 b_t（消除 G_t 的步位偏置）
        self._reset()

    def _reset(self):
        self.h = np.zeros(self.H)
        self.c = np.zeros(self.H)
        self.cells = []     # LSTM 截断 BPTT 缓存（最多 L 步）
        self.rollout = []   # (feat, z1, probs, action, mask) 每步 MLP 缓存

    def reset(self):
        self._reset()

    def act(self, s, mask, explore=True):
        # --- LSTM 一步前向（截断缓存窗口 L） ---
        x = s
        pre = x @ self.Wx + self.h @ self.Wh + self.b
        i, f, o, g = np.split(pre, 4)
        i, f, o, g = 1/(1+np.exp(-i)), 1/(1+np.exp(-f)), 1/(1+np.exp(-o)), np.tanh(g)
        c_new = f * self.c + i * g
        tanhc = np.tanh(c_new)
        h_new = o * tanhc
        self.cells.append(dict(x=x, h_prev=self.h, c_prev=self.c, i=i, f=f, o=o,
                               g=g, tanhc=tanhc, h=h_new))
        if len(self.cells) > self.L:
            self.cells.pop(0)
        self.h, self.c = h_new, c_new
        # --- 策略 MLP ---
        feat = np.concatenate([s, h_new])
        z1 = feat @ self.W1 + self.b1
        a1 = np.maximum(z1, 0)
        logits = a1 @ self.W2 + self.b2
        logits = np.where(mask, logits, -1e9)
        logits -= logits.max()
        p = np.exp(logits); p /= p.sum()
        a = int(rng_choice(p)) if explore else int(np.argmax(p + (~mask) * -1e9))
        self.rollout.append((feat, z1, p, a, mask))
        return a

    def update(self, rewards, gamma=PARAMS["gamma"]):
        """REINFORCE: ∇θ J = Σ_t (G_t - b) ∇θ log π(a_t|s'_t)，梯度经 h_t 回传 LSTM（截断 BPTT）。"""
        T = len(rewards)
        if T == 0:
            return 0.0
        # 折扣回报 G_t
        G = np.zeros(T)
        g = 0.0
        for t in reversed(range(T)):
            g = rewards[t] + gamma * g
            G[t] = g
        # 按时间步的 EMA 基线：adv_t = G_t - b_t（消除折扣回报随步位的系统偏置）
        for t in range(T):
            b = self.baseline.get(t)
            self.baseline[t] = G[t] if b is None else 0.9 * b + 0.1 * G[t]
        adv = np.array([G[t] - self.baseline[t] for t in range(T)])
        sd = adv.std() + 1e-8
        adv = np.clip(adv / sd, -3.0, 3.0)
        # MLP 反向：对每步求 dfeat（含 dh_t 分量），累积参数梯度
        gW1 = np.zeros_like(self.W1); gb1 = np.zeros_like(self.b1)
        gW2 = np.zeros_like(self.W2); gb2 = np.zeros_like(self.b2)
        dh_mlp = [None] * T
        for t, (feat, z1, p, a, mask) in enumerate(self.rollout):
            dlogits = -p.copy()
            dlogits[a] += 1.0                       # ∇ log softmax(a)
            dlogits *= adv[t]
            gb2 += dlogits
            gW2 += np.outer(np.maximum(z1, 0), dlogits)
            da1 = dlogits @ self.W2.T
            dz1 = da1 * (z1 > 0)
            gb1 += dz1
            gW1 += np.outer(feat, dz1)
            dh_mlp[t] = dz1 @ self.W1.T             # [ds_t(丢弃), dh_t]
        # LSTM 截断 BPTT（从最新往回最多 L 步）
        gWx = np.zeros_like(self.Wx); gWh = np.zeros_like(self.Wh)
        gb = np.zeros_like(self.b)
        dh = np.zeros(self.H); dc = np.zeros(self.H)
        n_cells = len(self.cells)
        for k in range(n_cells - 1, -1, -1):
            t = k - (n_cells - T)                   # 对应 rollout 步索引
            if 0 <= t < T:
                dh = dh + dh_mlp[t][ -self.H:]      # 该步策略损失对 h_t 的梯度
            cell = self.cells[k]
            i, f, o, g = cell["i"], cell["f"], cell["o"], cell["g"]
            tanhc = cell["tanhc"]
            dc_tot = dc + dh * o * (1 - tanhc ** 2)
            dpo = dh * tanhc * (o * (1 - o))
            dpf = dc_tot * cell["c_prev"] * (f * (1 - f))
            dpi = dc_tot * g * (i * (1 - i))
            dpg = dc_tot * i * (1 - g ** 2)
            dpre = np.concatenate([dpi, dpf, dpo, dpg])
            gWx += np.outer(cell["x"], dpre)
            gWh += np.outer(cell["h_prev"], dpre)
            gb += dpre
            dh = dpre @ self.Wh.T
            dc = dc_tot * f
            dh += 0  # 截断: 不继续穿越更早的窗口边界
        grads = [gWx, gWh, gb, gW1, gb1, gW2, gb2]   # ∇J 的上升方向（Adam 为 p += lr·g）
        self.opt.step(grads)
        self._reset()
        return float(G.mean())


def rng_choice(p):
    r = np.random.random()
    acc = 0.0
    for j, pj in enumerate(p):
        acc += pj
        if r <= acc:
            return j
    return len(p) - 1


# ----------------------------------------------------------------------
# Algorithm 1: PSC-SPRL
# ----------------------------------------------------------------------
def run_episode(env, policy, rng, n_events, refine_steps=3, psc_only=False,
                collect_metrics=True, greedy=False):
    """按 Algorithm 1 执行一回合：PSC 初始分配 → UAV 按载重降序指派 → SPRL 精调 → 服务。"""
    env.reset()
    policy.reset()
    metrics = dict(Fd_init=[], Fd_final=[], Fw=[], F=[], reward=[])
    t = 0
    while t < n_events:                       # 滚动时域
        env.step_event()                      # 动态事件: 新订单 / 取消
        act = env.active_orders()
        if not act:
            t += 1
            continue
        pos = np.array([o.pos for o in act])
        urg = np.array([env.urgency(o) for o in act])
        labels, K = priority_spectral_clustering(pos, urg, rng)   # PSC
        for j, o in enumerate(act):
            o.cluster = int(labels[j])
        # Algorithm 1: 仅前 min(|U|,K) 架（按最大载重降序）参与初始指派
        uav_cluster = {}
        for rank, u in enumerate(sorted(env.uavs, key=lambda u: -u.cap)[:min(len(env.uavs), K)]):
            uav_cluster[u.idx] = rank
        Fd0, Fw, Fe, _ = env.costs(uav_cluster)
        Fd_init = Fd0
        # --- RL 精调 ---
        if not psc_only:
            for _ in range(refine_steps):
                s = env.state_features(K, labels, uav_cluster)
                mask = env.action_mask(K, len(act))
                a = policy.act(s, mask, explore=not greedy)
                res = env.execute(a, K, len(act))
                if res[0] == "Assign":
                    uav_cluster[res[1] + 1] = res[2]
                elif res[0] == "Reassign":
                    i, c = res[1], res[2]
                    if i < len(act):
                        act[i].cluster = c
                metrics["reward"].append(env.reward(uav_cluster))   # Eq.(srl4)
        Fd1, Fw, Fe, _ = env.costs(uav_cluster)
        if collect_metrics:
            metrics["Fd_init"].append(Fd_init)
            metrics["Fd_final"].append(Fd1)
            metrics["Fw"].append(Fw)
            metrics["F"].append(env.comprehensive_cost(Fd1, Fw, Fe))
        env.serve(uav_cluster)
        t += 1
    served_ratio = env.n_served / max(env.n_generated, 1)
    metrics["served"] = served_ratio
    return metrics


def summarize(metrics):
    fd_i = np.mean(metrics["Fd_init"]) if metrics["Fd_init"] else 0.0
    fd_f = np.mean(metrics["Fd_final"]) if metrics["Fd_final"] else 0.0
    red = (fd_i - fd_f) / (fd_i + 1e-9) * 100
    return dict(Fd_init=fd_i, Fd_final=fd_f, reduction=red,
                Fw=np.mean(metrics["Fw"]) if metrics["Fw"] else 0.0,
                F=np.mean(metrics["F"]) if metrics["F"] else 0.0,
                served=metrics["served"])


def main():
    ap = argparse.ArgumentParser(description="PSC-SPRL (Att48, paper parameters)")
    ap.add_argument("--episodes", type=int, default=PARAMS["T_prime"],
                    help="训练迭代 T'_max（默认 200）")
    ap.add_argument("--events", type=int, default=PARAMS["T_max"],
                    help="每回合事件数 T_max（默认 100）")
    ap.add_argument("--intensity", type=int, default=5, choices=[5, 10, 15, 20],
                    help="每事件新订单强度（默认 5）")
    ap.add_argument("--refine", type=int, default=3, help="每事件 RL 精调步数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--psc-only", action="store_true", help="PSC-only 基线")
    ap.add_argument("--window", type=int, default=PARAMS["L"], help="LSTM 窗口 L")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    env = DynamicEnv(intensity=args.intensity,
                    initial_active=PARAMS["initial_active"], seed=args.seed)
    dummy = env.reset()
    policy = SPRLPolicy(len(dummy), env.n_actions(), L=args.window,
                         seed=args.seed + 1)

    if args.psc_only:
        m = run_episode(env, policy, rng, args.events, psc_only=True)
        s = summarize(m)
        print(f"\n===== PSC-only 基线 (Att48, intensity={args.intensity}, seed={args.seed}) =====")
    else:
        print(f"PSC-SPRL 训练: episodes={args.episodes}, events/ep={args.events}, "
              f"intensity={args.intensity}, L={args.window}, "
              f"α={PARAMS['alpha_psc']}, β={PARAMS['beta_psc']}, σ={PARAMS['sigma']}, "
              f"(α1,α2,α3)=({PARAMS['a1']},{PARAMS['a2']},{PARAMS['a3']})")
        curve = []
        np.random.seed(args.seed)
        for ep in range(args.episodes):
            m = run_episode(env, policy, rng, args.events, refine_steps=args.refine)
            policy.update(m["reward"])                 # 策略梯度更新（Algorithm 1 while 循环体）
            mr = np.mean(m["reward"]) if m["reward"] else 0.0
            curve.append(mr)
            if (ep + 1) % max(1, args.episodes // 10) == 0:
                print(f"  episode {ep+1:4d}/{args.episodes}  mean reward = {mr:9.2f}  "
                      f"served ratio = {m['served']*100:5.1f}%")
        # 最终评估（贪心策略）
        env2 = DynamicEnv(intensity=args.intensity,
                         initial_active=PARAMS["initial_active"], seed=args.seed + 100)
        m = run_episode(env2, policy, rng, args.events, refine_steps=args.refine,
                        collect_metrics=True, greedy=True)
        s = summarize(m)
        print(f"\n===== PSC-SPRL 结果 (Att48, intensity={args.intensity}, seed={args.seed}) =====")
        print(f"训练奖励曲线: {['%.1f' % c for c in curve[:5]]} ... "
              f"{['%.1f' % c for c in curve[-3:]]}")

    print("-" * 62)
    print(f"{'指标':<28}{'数值':>15}")
    print("-" * 62)
    print(f"{'F_d 初始 (仅 PSC)  ↓':<30}{s['Fd_init']:>14.2f} m")
    print(f"{'F_d 最终 (PSC+SPRL) ↓':<30}{s['Fd_final']:>14.2f} m")
    print(f"{'F_d 降幅 ↑':<30}{s['reduction']:>13.2f}%")
    print(f"{'F_w 平均等待成本 ↓':<30}{s['Fw']:>14.2f} min")
    print(f"{'F  综合成本 (W=0.3/0.4/0.3) ↓':<30}{s['F']:>14.2f}")
    print(f"{'Served Ratio 送达率 ↑':<30}{s['served']*100:>13.2f}%")
    print("-" * 62)


if __name__ == "__main__":
    main()
