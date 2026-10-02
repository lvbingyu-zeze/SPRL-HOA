#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FFE-HOA: CNN 引导的适应度特征提取人工鱼群算法（路径规划，论文 Sec. 4.3, Algorithm 2）
============================================================================
流水线（SPRL-HOA 的路径规划阶段）:
  PSC 聚类分配(复用 psc_sprl) → FFE-HOA 规划各 UAV 簇内路径 τ_u

Algorithm 2 各步骤实现:
  1. 初始种群: N=50 条人工鱼, 二进制编码 N_i ∈ {0,1}^D。
     D = Σ_u n_u² —— 每架 UAV 一个 n_u×n_u 簇内节点邻接矩阵块（节点 0 为仓库；
     论文 D=|U|·|C|² 的簇分解可解码变体，分配由 PSC-SPRL 决定）
  2. 双适应度: F1 = Eq.(of) 综合成本(W=0.3/0.4/0.3) + 障碍惩罚常数; F2 = 1/F1
  3. 鱼群移动 Eq.(hfoa1): N_ij ← N_ij + s_ij·d_ij（潜变量 + 二值化）
  4. K-means 将种群聚为 L 个子种群（增加多样性）
  5. 子种群内人工鱼学习（visual scope 内追尾/聚群行为）
  6. CNN 引导步长/方向更新 Eq.(hfoa2): x1=0.6, x2=0.7
  7. 自适应性别切换: 阈值由适应度分布迭代收敛（λ1=0.1, λ2=0.9 分位数初始化）
  8. 迭代 T''_max=300 次, 输出各 UAV 路径 τ_u（含障碍绕行, 保证可行性）

CNN（Att48 最优配置, Table C3 / Table II）:
  输入 3 通道 H×W 图像: 二进制解 / 3×3 邻域聚合 / 归一化适应度(F2)
  conv1: 1→32, k=3, p=1 → ReLU → maxpool 2×2 s2
  conv2: 32→64, k=5, p=2 → ReLU → maxpool 2×2 s2
  flatten → FC-100 ReLU → FC-1 sigmoid → 概率 p_i（种群分类的数据驱动依据）
  在线训练: BCE(p_i, 适应度优于种群中位数) —— 前向/反向传播全部手写（torch 不可用）

实验设置（Sec 5.1）:
  - Att48: 1 仓库 + 47 客户, 初始 |C^a|=25, PSC 聚类(Eq. srl1/srl2)
  - 10 架异构 UAV: 5-10kg(4,8,9), 11-20kg(1,2,3,6), 21-30kg(5,7,10), 按载重降序指派
  - 10 个圆柱形障碍物, 半径 10-100 m 整数均匀随机, 避开仓库与客户点
  - v_u=10 m/s; W1,W2,W3=0.3,0.4,0.3; N=50; T''_max=300; λ1=0.1, λ2=0.9

用法:
  python ffe_hoa.py                    # 论文默认参数
  python ffe_hoa.py --active 45        # 45 活跃订单
  python ffe_hoa.py --iters 100        # 缩短迭代
  python ffe_hoa.py --config berlin52  # 其他数据集 CNN 配置(Table II)
依赖: numpy（与 psc_sprl.py 同目录）
"""

import argparse
import time
import numpy as np

from psc_sprl import (ATT48, PARAMS, priority_spectral_clustering, kmeans, Adam)

# ----------------------------------------------------------------------
# Table II / Table C3: 各数据集 CNN 最优配置（channels, kernels, paddings）
# ----------------------------------------------------------------------
CNN_CONFIGS = {
    "att48":    dict(channels=[1, 32, 64],     kernels=[3, 5], pads=[1, 2]),
    "berlin52": dict(channels=[1, 32, 64],     kernels=[5, 2], pads=[2, 1]),
    "eil76":    dict(channels=[1, 16, 32, 64], kernels=[3, 3, 3], pads=[1, 1, 1]),
    "bier127":  dict(channels=[1, 16, 32],     kernels=[3, 3], pads=[1, 1]),
    "ch130":    dict(channels=[1, 16, 32, 64], kernels=[3, 3, 3], pads=[1, 1, 1]),
}

EPS = 1e-9


# ======================================================================
# 手写 CNN 层（float32, tensordot/BLAS 加速）
# ======================================================================
class Conv2D:
    def __init__(self, cin, cout, k, pad, rng):
        self.cin, self.cout, self.k, self.pad = cin, cout, k, pad
        self.W = rng.normal(0, np.sqrt(2.0 / (cin * k * k)),
                            (cout, cin, k, k)).astype(np.float32)
        self.b = np.zeros(cout, np.float32)
        self.Z = None        # 前向缓存(ReLU 反传)

    def forward(self, X):
        B, C, H, W = X.shape
        k, p = self.k, self.pad
        Xp = np.pad(X, ((0, 0), (0, 0), (p, p), (p, p))) if p else X
        Ho, Wo = H + 2 * p - k + 1, W + 2 * p - k + 1
        out = np.zeros((B, self.cout, Ho, Wo), np.float32)
        for ky in range(k):
            for kx in range(k):
                patch = Xp[:, :, ky:ky + Ho, kx:kx + Wo]
                out += np.tensordot(patch, self.W[:, :, ky, kx],
                                    axes=([1], [1])).transpose(0, 3, 1, 2)
        out += self.b.reshape(1, -1, 1, 1)
        self.Xp, self.X = Xp, X
        self.Z = out
        return out

    def backward(self, dZ):
        """返回 (dW, db, dX)。"""
        B, Co, Ho, Wo = dZ.shape
        k, p = self.k, self.pad
        dW = np.zeros_like(self.W)
        db = dZ.sum(axis=(0, 2, 3))
        for ky in range(k):
            for kx in range(k):
                # dW[o,c] = Σ_{b,h,w} Xp[b,c,·,·]·dZ[b,o,·,·] → 转置为 (cout, cin)
                dW[:, :, ky, kx] = np.tensordot(
                    self.Xp[:, :, ky:ky + Ho, kx:kx + Wo], dZ,
                    axes=([0, 2, 3], [0, 2, 3])).T
        dXp = np.zeros_like(self.Xp)
        for ky in range(k):
            for kx in range(k):
                # dXp[b,c,ky+h,kx+w] += Σ_o W[o,c,·,·]·dZ[b,o,·,·] → 转置回 (B,cin,Ho,Wo)
                dXp[:, :, ky:ky + Ho, kx:kx + Wo] += np.tensordot(
                    dZ, self.W[:, :, ky, kx],
                    axes=([1], [0])).transpose(0, 3, 1, 2)
        if p:
            dX = dXp[:, :, p:p + self.X.shape[2], p:p + self.X.shape[3]]
        else:
            dX = dXp
        return dW, db, dX


class MaxPool2:
    """2×2, stride 2 最大池化。"""

    def forward(self, X):
        B, C, H, W = X.shape
        H2, W2 = H // 2, W // 2
        self.shape_in = X.shape
        self.hw2 = (H2, W2)
        Xr = X[:, :, :H2 * 2, :W2 * 2].reshape(B, C, H2, W2, 4)
        self.idx = Xr.argmax(axis=-1)                  # 窗内最大元素位置
        return np.take_along_axis(Xr, self.idx[..., None], axis=-1)[..., 0]

    def backward(self, dY):
        B, C, H2, W2 = dY.shape
        dXr = np.zeros((B, C, H2, W2, 4), np.float32)
        b, c, h, w = np.meshgrid(np.arange(B), np.arange(C),
                                 np.arange(H2), np.arange(W2), indexing="ij")
        dXr[b, c, h, w, self.idx] = dY
        dX = np.zeros(self.shape_in, np.float32)       # 奇数尺寸的末行/末列梯度丢弃
        dX[:, :, :H2 * 2, :W2 * 2] = dXr.reshape(B, C, H2 * 2, W2 * 2)
        return dX


class FFE_CNN:
    """Table C3 最优配置 CNN + 输出头(FC-100 ReLU → FC-1 sigmoid)。
    前向: [conv→ReLU→pool]×n → flatten → FC100 ReLU → FC1 sigmoid。"""

    def __init__(self, cfg, rng, lr=1e-3):
        # 输入为 3 通道(解/邻域聚合/适应度), 第一层 cin=3, 其余按 Table 配置
        chans = [3] + cfg["channels"][1:]
        self.convs = [Conv2D(chans[i], chans[i + 1],
                             cfg["kernels"][i], cfg["pads"][i], rng)
                      for i in range(len(cfg["kernels"]))]
        self.pools = [MaxPool2() for _ in self.convs]
        self.rng = rng
        self.W1 = self.b1 = self.W2 = self.b2 = None
        self._adam = None
        self.lr = lr

    def _init_head(self, flat_dim):
        self.W1 = self.rng.normal(0, np.sqrt(2.0 / flat_dim),
                                  (flat_dim, 100)).astype(np.float32)
        self.b1 = np.zeros(100, np.float32)
        self.W2 = self.rng.normal(0, np.sqrt(2.0 / 100),
                                  (100, 1)).astype(np.float32)
        self.b2 = np.zeros(1, np.float32)
        self._adam = Adam([self.W1, self.b1, self.W2, self.b2], lr=self.lr)

    def forward(self, X):
        A = X.astype(np.float32)
        for conv, pool in zip(self.convs, self.pools):
            Z = conv.forward(A)
            A = pool.forward(np.maximum(Z, 0))
        # 全局平均池化(GAP)替代直接 flatten: 决策周期内 H×W 随 |C^a| 变化,
        # GAP 使特征维度固定 (=末层通道数), F_CNN 训练一次即可跨事件复用(Algorithm 3)
        B, C = A.shape[0], A.shape[1]
        spatial = A.shape[2] * A.shape[3]
        flat = A.mean(axis=(2, 3))                              # (B, C)
        if self.W1 is None:
            self._init_head(flat.shape[1])
        z1 = flat @ self.W1 + self.b1
        h1 = np.maximum(z1, 0)
        z2 = h1 @ self.W2 + self.b2
        p = 1.0 / (1.0 + np.exp(-z2[:, 0]))
        self.cache = dict(A=A, flat=flat, z1=z1, h1=h1, p=p, spatial=spatial)
        return flat, p

    def train(self, X, y, epochs=3):
        """在线 BCE 训练: y=1 表示该鱼适应度优于种群中位数。"""
        for _ in range(epochs):
            flat, p = self.forward(X)
            B = max(len(p), 1)
            dz2 = ((p - y) / B).astype(np.float32)[:, None]        # BCE+sigmoid
            flat, z1, h1 = self.cache["flat"], self.cache["z1"], self.cache["h1"]
            dW2 = h1.T @ dz2
            db2 = dz2.sum(0)
            dh1 = dz2 @ self.W2.T
            dz1 = dh1 * (z1 > 0)
            dW1 = flat.T @ dz1
            db1 = dz1.sum(0)
            dflat = dz1 @ self.W1.T
            # GAP 反传: 梯度均匀分配到每个空间位置, 广播到池化输出形状
            dA = np.broadcast_to(
                (dflat / self.cache["spatial"])[:, :, None, None],
                self.cache["A"].shape)
            # 反传穿过 [pool → ReLU → conv] 块（从最后一层到第一层）
            for i in reversed(range(len(self.convs))):
                dA_pool_in = self.pools[i].backward(dA)            # pool 输入(=ReLU 输出)
                dZ = dA_pool_in * (self.convs[i].Z > 0)            # ReLU mask
                dW, db, dX = self.convs[i].backward(dZ)
                dA = dX                                           # 上一层 pool 输出
            self._adam.step([-dW1, -db1, -dW2, -db2])             # 梯度下降
        return self.forward(X)[1]


# ======================================================================
# 场景: Att48 + 10 个圆柱障碍物 + PSC 聚类分配
# ======================================================================
class Obstacles:
    """10 个圆柱形静态障碍物, 半径 10-100 m 整数随机（避开仓库/客户点）。"""

    def __init__(self, rng, bbox, home, pts, n_obs=10, margin=10):
        self.margin = margin
        self.circles = []
        lo, hi = bbox
        guard = 0
        while len(self.circles) < n_obs and guard < 20000:
            guard += 1
            r = int(rng.integers(10, 101))
            c = rng.uniform(lo, hi, 2)
            if np.linalg.norm(c - home) < r + 150:
                continue
            if any(np.linalg.norm(c - p) < r + margin for p in pts):
                continue
            self.circles.append((c, float(r)))

    def seg(self, A, B):
        """线段 A→B 障碍感知长度: 穿越圆柱则经切向外侧点绕行。返回 (长度, 违规数, 绕行点)。"""
        best_len, best_viol, best_mid = float(np.linalg.norm(B - A)), 0, None
        for c, r in self.circles:
            ab = B - A
            t = np.clip(np.dot(c - A, ab) / (np.dot(ab, ab) + EPS), 0, 1)
            p0 = A + t * ab
            dist = np.linalg.norm(c - p0)
            if dist < r:
                n = (p0 - c) / (dist + EPS)
                mid = c + n * (r + self.margin)
                detour = float(np.linalg.norm(mid - A) + np.linalg.norm(B - mid))
                if best_mid is None or detour < best_len:
                    best_len, best_viol, best_mid = detour, 1, mid
        return best_len, best_viol, best_mid


class Scenario:
    """Att48 静态规划实例: |C^a| 活跃订单 + PSC 聚类 + UAV 指派 + 距离矩阵。"""

    def __init__(self, seed=0, n_active=25, K_override=None):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.home = ATT48[0].copy()
        cust = ATT48[1:]
        idx = rng.choice(len(cust), n_active, replace=False)
        self.order_pos = cust[idx].copy()
        self.payload = rng.uniform(1.0, 8.0, n_active)
        urgency = rng.uniform(0.0, 1.0, n_active)
        labels, K = priority_spectral_clustering(self.order_pos, urgency, rng)
        if K_override and K_override != K:
            # 参数研究允许覆盖 K（Table C1: Att48 最优 K=6）：直接 kmeans 重聚类
            labels, K = kmeans(self.order_pos, K_override, rng), K_override
        self.K = K
        # UAV: 按最大载重降序指派前 min(|U|,K) 架（Algorithm 1）
        self.uav_caps = dict(enumerate(PARAMS["uav_caps"], start=1))
        self.uav_ids = list(self.uav_caps.keys())
        sorted_uavs = sorted(self.uav_ids, key=lambda u: -self.uav_caps[u])
        self.uav_cluster = {u: c for c, u in
                            enumerate(sorted_uavs[:min(len(sorted_uavs), K)])}
        # 障碍物（避开所有节点）
        lo, hi = ATT48.min(0), ATT48.max(0)
        self.obs = Obstacles(rng, (lo, hi), self.home,
                             np.vstack([self.home, self.order_pos]))
        # 每架 UAV 的节点表(节点 0 = 仓库)与预计算障碍感知距离矩阵
        self.blocks = {}
        for u in self.uav_ids:
            members = (np.where(labels == self.uav_cluster[u])[0]
                       if u in self.uav_cluster else np.array([], int))
            nodes = [self.home] + [self.order_pos[m] for m in members]
            n = len(nodes)
            DM = np.zeros((n, n))
            VI = np.zeros((n, n), int)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        DM[i, j], VI[i, j], _ = self.obs.seg(nodes[i], nodes[j])
            self.blocks[u] = dict(members=members, nodes=nodes, DM=DM, VI=VI)
        # 编码维度 D = Σ_u n_u²（每 UAV 一个 n_u×n_u 邻接块）
        self.block_off = {}
        off = 0
        for u in self.uav_ids:
            n = len(self.blocks[u]["nodes"])
            self.block_off[u] = (off, n)
            off += n * n
        self.D = off
        self.cmax = max(len(self.blocks[u]["nodes"]) for u in self.uav_ids)


# ======================================================================
# FFE-HOA（Algorithm 2）
# ======================================================================
class FFEHOA:
    def __init__(self, scen, cfg_name="att48", N=50, L=5, T_max=300,
                 x1=0.6, x2=0.7, lam1=0.1, lam2=0.9, seed=0,
                 cnn_every=10, cnn_epochs=3, penalty=5000.0, v_uav=10.0,
                 verbose=True, cnn_frozen=False):
        self.sc, self.N, self.L, self.T = scen, N, L, T_max
        self.x1, self.x2, self.lam1, self.lam2 = x1, x2, lam1, lam2
        self.penalty, self.v = penalty, v_uav
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose
        self.D = scen.D
        # 种群: 潜变量 X∈[0,1], 二进制解 N = (X>0.5)
        self.X = self.rng.uniform(0, 1, (N, self.D)).astype(np.float32)
        self.s = self.rng.uniform(0.01, 0.1, (N, self.D)).astype(np.float32)
        self.d = self.rng.choice([-1.0, 1.0], (N, self.D)).astype(np.float32)
        self.gender = self.rng.integers(0, 2, N)
        self.cnn = FFE_CNN(CNN_CONFIGS[cfg_name], self.rng)
        self.cnn_every, self.cnn_epochs = cnn_every, cnn_epochs
        self.cnn_frozen = cnn_frozen
        self.feats = None
        self.p = np.full(N, 0.5)
        self.best_F1, self.best_X = np.inf, None

    # ---------- 解码: 二进制 → 各 UAV 访问序列 ----------
    def decode(self, x):
        tours, F_d, F_w, viol = {}, 0.0, 0.0, 0
        for u in self.sc.uav_ids:
            off, n = self.sc.block_off[u]
            M = x[off:off + n * n].reshape(n, n)
            blk = self.sc.blocks[u]
            DM, VI = blk["DM"], blk["VI"]
            seq, remaining, cur = [0], set(range(1, n)), 0
            t_arr = 0.0
            while remaining:
                allowed = [j for j in remaining if M[cur, j] > 0.5]
                pool = allowed if allowed else list(remaining)
                dists = [DM[cur, j] for j in pool]
                nxt = pool[int(np.argmin(dists))]
                t_arr += DM[cur, nxt] / self.v
                F_w += t_arr / 60.0                      # 累计等待(分钟)
                viol += int(VI[cur, nxt])
                F_d += DM[cur, nxt]
                seq.append(nxt)
                remaining.remove(nxt)
                cur = nxt
            tours[u] = seq
        return tours, F_d, F_w, viol

    # ---------- 双适应度 ----------
    def evaluate(self, X=None):
        X = self.X if X is None else X
        F1 = np.zeros(len(X))
        for i, x in enumerate(X):
            _, F_d, F_w, viol = self.decode(x)
            F_e = PARAMS["e_rate"] * F_d
            w1, w2, w3 = PARAMS["W"]
            F1[i] = w1 * F_d + w2 * F_e + w3 * F_w + self.penalty * viol
        return F1, 1.0 / (F1 + 1.0)

    # ---------- CNN 输入图像: 3 通道(解/邻域聚合/适应度) ----------
    def make_images(self, F2):
        cm = max(self.sc.cmax, 4)                        # 最小画布 4: 保证两次 2×2 池化后维度≥1
        H = len(self.sc.uav_ids) * cm                   # 每个 UAV 占 cm 行
        W = cm
        B = self.N
        imgs = np.zeros((B, 3, H, W), np.float32)
        for i in range(B):
            img = np.zeros((H, W), np.float32)
            for u in self.sc.uav_ids:
                off, n = self.sc.block_off[u]
                r0 = (u - 1) * cm
                block = (self.X[i, off:off + n * n] > 0.5).astype(np.float32)
                img[r0:r0 + n, :n] = block.reshape(n, n)
            imgs[i, 0] = img
            p = np.pad(img, 1)                          # 通道2: 3×3 邻域聚合
            agg = np.zeros((H, W), np.float32)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    agg += p[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
            imgs[i, 1] = agg / 9.0
            imgs[i, 2] = F2[i] / (F2.max() + EPS)       # 通道3: 归一化适应度
        return imgs

    # ---------- 自适应性别切换阈值(迭代收敛) ----------
    def gender_threshold(self, F1s):
        t1, t2 = np.quantile(F1s, self.lam1), np.quantile(F1s, self.lam2)
        for _ in range(20):
            if abs(t1 - t2) < 1e-6:
                break
            a, b_, c_ = F1s[F1s < t1], F1s[(F1s >= t1) & (F1s <= t2)], F1s[F1s > t2]
            fa = a.mean() if len(a) else t1
            fb = b_.mean() if len(b_) else (t1 + t2) / 2
            fc = c_.mean() if len(c_) else t2
            t1n, t2n = (fa + fb) / 2, (fb + fc) / 2
            if abs(t1n - t1) < 1e-9 and abs(t2n - t2) < 1e-9:
                t1, t2 = t1n, t2n
                break
            t1, t2 = t1n, t2n
        return (t1 + t2) / 2

    # ---------- CNN 特征 → 每鱼每维调制因子 f_{i,j} ----------
    def _feature_modulation(self):
        F = self.feats
        mu, sd = F.mean(0, keepdims=True), F.std(0, keepdims=True) + EPS
        f01 = 1.0 / (1.0 + np.exp(-(F - mu) / sd))       # (B, CH) ∈ (0,1)
        B, CH = f01.shape
        return np.tile(f01, (1, (self.D + CH - 1) // CH))[:, :self.D].astype(np.float32)

    # ---------- 主循环 (Algorithm 2) ----------
    def run(self):
        ET = []
        for it in range(self.T):
            t0 = time.time()
            F1, F2 = self.evaluate()                     # ① 双适应度评估
            bi = int(np.argmin(F1))
            if F1[bi] < self.best_F1:
                self.best_F1, self.best_X = F1[bi], self.X[bi].copy()
            # ⑥(预备) CNN 刷新: 特征提取 + 在线训练 + 概率 p_i
            if self.feats is None or it % self.cnn_every == 0:
                imgs = self.make_images(F2)
                self.feats = self.cnn.forward(imgs)[0]
                if self.cnn_frozen:                      # 冻结模式: 仅前向提取特征
                    self.p = self.cnn.cache["p"]
                else:                                    # 在线模式: BCE 训练
                    y = (F1 <= np.median(F1)).astype(np.float32)
                    self.p = self.cnn.train(imgs, y, epochs=self.cnn_epochs)
            # ② 鱼群移动 Eq.(hfoa1): N += s·d → 二值化
            self.X = np.clip(self.X + self.s * self.d, 0.0, 1.0)
            F1, F2 = self.evaluate()                     # ③ 重新评估
            bi = int(np.argmin(F1))
            if F1[bi] < self.best_F1:
                self.best_F1, self.best_X = F1[bi], self.X[bi].copy()
            gbin = (self.best_X > 0.5).astype(np.float32)
            # ④ K-means 聚为 L 个子种群
            labels = kmeans(self.X, self.L, self.rng, iters=15)
            # ⑤+⑦ 子种群内学习 + 性别切换
            V = {0: 0.6, 1: 0.35}                         # V_male / V_female
            dnorm = np.sqrt(self.D)
            for sub in range(self.L):
                ids = np.where(labels == sub)[0]
                if len(ids) == 0:
                    continue
                thr = self.gender_threshold(F1[ids])
                for i in ids:
                    if F1[i] > thr:                      # 性别切换
                        self.gender[i] = 1 - self.gender[i]
                    dd = np.linalg.norm(self.X[ids] - self.X[i], axis=1) / dnorm
                    nb = [j for k, j in enumerate(ids)
                          if j != i and dd[k] < V[self.gender[i]]]
                    if nb:                               # 追尾: 移向最优邻居
                        bnb = nb[int(np.argmin(F1[nb]))]
                        self.X[i] = np.clip(
                            self.X[i] + 0.3 * (self.X[bnb] - self.X[i]), 0, 1)
                    else:                                 # 聚群失败: 随机游动
                        self.X[i] = np.clip(
                            self.X[i] + self.rng.normal(0, 0.05, self.D), 0, 1)
                    if self.p[i] < 0.5:                   # 弱个体向全局最优靠拢
                        self.X[i] = np.clip(
                            self.X[i] + 0.1 * (self.best_X - self.X[i]), 0, 1)
            # ⑥ Eq.(hfoa2): CNN 特征引导步长/方向更新
            fmod = self._feature_modulation()
            for sub in range(self.L):
                ids = np.where(labels == sub)[0]
                if len(ids) == 0:
                    continue
                lbest = ids[int(np.argmin(F1[ids]))]
                lbin = (self.X[lbest] > 0.5).astype(np.float32)
                for i in ids:
                    Nij = (self.X[i] > 0.5).astype(np.float32)
                    self.s[i] = np.clip(
                        self.s[i] + self.x1 * fmod[i] * (gbin - Nij), 1e-4, 0.5)
                    self.d[i] = np.clip(
                        self.d[i] + self.x2 * fmod[i] * (Nij - lbin), -2.0, 2.0)
            ET.append(time.time() - t0)
            if self.verbose and (it + 1) % max(1, self.T // 10) == 0:
                print(f"  iter {it+1:4d}/{self.T}  best F1 = {self.best_F1:12.2f}  "
                      f"ET = {ET[-1]*1000:6.1f} ms")
        return self.best_X, self.best_F1, np.array(ET)


# ----------------------------------------------------------------------
# 基线: 纯最近邻贪心（解码 fallback 即 NN）
# ----------------------------------------------------------------------
def nn_baseline(scen, v=10.0, penalty=5000.0):
    hoa = FFEHOA.__new__(FFEHOA)
    hoa.sc, hoa.v, hoa.penalty = scen, v, penalty
    x = np.zeros(scen.D, np.float32)
    _, F_d, F_w, viol = hoa.decode(x)
    F_e = PARAMS["e_rate"] * F_d
    w1, w2, w3 = PARAMS["W"]
    return w1 * F_d + w2 * F_e + w3 * F_w + penalty * viol, F_d, F_w, viol


def main():
    ap = argparse.ArgumentParser(description="FFE-HOA (Att48, paper parameters)")
    ap.add_argument("--active", type=int, default=25, help="初始 |C^a|（默认 25）")
    ap.add_argument("--iters", type=int, default=300, help="T''_max（默认 300）")
    ap.add_argument("--K", type=int, default=None,
                    help="覆盖簇数 K（Table C1: Att48 最优 6）")
    ap.add_argument("--pop", type=int, default=50, help="种群 N（默认 50）")
    ap.add_argument("--sub", type=int, default=5, help="子种群数 L")
    ap.add_argument("--config", default="att48", choices=list(CNN_CONFIGS),
                    help="CNN 配置（Table II, 默认 att48）")
    ap.add_argument("--cnn-every", type=int, default=10, help="CNN 刷新间隔")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    scen = Scenario(seed=args.seed, n_active=args.active, K_override=args.K)
    print(f"FFE-HOA 路径规划: Att48, |C^a|={args.active}, K={scen.K}, "
          f"障碍物={len(scen.obs.circles)}个(半径10-100m), N={args.pop}, "
          f"T''_max={args.iters}, λ=(0.1, 0.9), x1=0.6, x2=0.7")
    print(f"编码维度 D={scen.D}, CNN 配置[{args.config}]: {CNN_CONFIGS[args.config]}")

    t0 = time.time()
    hoa = FFEHOA(scen, cfg_name=args.config, N=args.pop, L=args.sub,
                 T_max=args.iters, seed=args.seed, cnn_every=args.cnn_every)
    best_X, best_F1, ET = hoa.run()
    wall = time.time() - t0

    tours, F_d, F_w, viol = hoa.decode(best_X)
    F_e = PARAMS["e_rate"] * F_d
    w1, w2, w3 = PARAMS["W"]
    F = w1 * F_d + w2 * F_e + w3 * F_w

    F1_0, _ = hoa.evaluate(hoa.rng.uniform(0, 1, (hoa.N, hoa.D)).astype(np.float32))
    F_nn, Fd_nn, Fw_nn, v_nn = nn_baseline(scen)

    print(f"\n===== FFE-HOA 结果 (Att48, |C^a|={args.active}, K={scen.K}) =====")
    print("-" * 60)
    print(f"{'指标':<34}{'数值':>16}")
    print("-" * 60)
    print(f"{'F_d 路径总长 ↓':<36}{F_d:>15.2f} m")
    print(f"{'F_e 能量成本 ↓':<36}{F_e:>15.2f}")
    print(f"{'F_w 等待成本 ↓':<36}{F_w:>15.2f} min")
    print(f"{'F  综合成本 ↓':<36}{F:>15.2f}")
    print(f"{'随机初始种群最优 F1 ↓':<36}{F1_0.min():>15.2f}")
    print(f"{'最近邻基线 F ↓':<36}{F_nn:>15.2f}")
    print(f"{'绕行障碍次数(输出路径可行)':<32}{viol:>12d}")
    print(f"{'ET_min / avg / max (s)':<36}"
          f"{ET.min():>6.2f} / {ET.mean():>5.2f} / {ET.max():>5.2f}")
    print(f"{'总运行时间 (s)':<36}{wall:>16.1f}")
    print("-" * 60)
    print("各 UAV 路径 τ_u:")
    for u in sorted(scen.uav_cluster, key=lambda u: scen.uav_cluster[u]):
        blk = scen.blocks[u]
        seq = tours[u]
        names = ["仓库"] + [f"订单{int(blk['members'][k - 1])}" for k in seq[1:]]
        print(f"  UAV {u:2d} (cap={scen.uav_caps[u]:>2d}kg, 簇"
              f"{scen.uav_cluster[u]}): {' → '.join(names)}")
    for u in scen.uav_ids:
        if u not in scen.uav_cluster:
            print(f"  UAV {u:2d} (cap={scen.uav_caps[u]:>2d}kg): 未指派")


if __name__ == "__main__":
    main()
