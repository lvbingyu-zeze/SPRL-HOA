#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPRL-HOA: Algorithm 3 总装（PSC-SPRL 订单分配 + FFE-HOA 路径规划）
====================================================================
Algorithm 3 流程:
  0. 预训练: π_θ 由 Algorithm 1 训练, F_CNN 由 Algorithm 2 训练（各一次）
  1. while C^a ≠ ∅ and t < T_max:
     a. Algorithm 1 (PSC-SPRL) 求最优订单分配 DO
     b. 未指派 UAV 按最大载重降序排序（已指派者从排序池移除）
     c. 对每个指派 UAV, Algorithm 2 (FFE-HOA) 规划簇内路径 τ_u
     d. 执行配送并更新系统状态（载荷/续航约束, 障碍感知距离）
     e. 完成的订单移出 C^a; t += 1
  2. 返回全部路径 τ

OAT 权重敏感性分析（Sec "SPRL-HOA", Fig. weight）:
  - 初始权重 (W1,W2,W3) = (0.34, 0.46, 0.20) = EW(等权) 与 AHP 的平均
  - 一次一因子(OAT), 10% 增量, 重归一化
  - 论文全局最优: (0.30, 0.40, 0.30) —— 作为本流水线默认运行参数

参数（论文参数研究结论）:
  PSC:  α=0.4, β=0.6, σ=0.5, K=max(1,|C^a|/5)
  SPRL: (α1,α2,α3)=(1e-5,100,0.01), q*=0.8, LSTM L=80, γ=0.99
  FFE-HOA: CNN 1→32(k3,p1)→32→64(k5,p2), x1=0.6, x2=0.7, λ=(0.1,0.9), N=50
  权重: W=(0.30,0.40,0.30); T_max=100 事件; Δt=100s; 订单强度 5/10/15/20
  场景: Att48 + 10 圆柱障碍物(半径10-100m) + 10 架异构 UAV(5-30kg)

用法:
  python sprl_hoa.py                     # 完整流水线(论文参数, W=0.3/0.4/0.3)
  python sprl_hoa.py --events 30         # 快速演示
  python sprl_hoa.py --oat               # W1-W3 OAT 敏感性分析(Fig. weight 复现)
  python sprl_hoa.py --full              # 论文全量训练(T'_max=200 episodes)
依赖: numpy; 同目录 psc_sprl.py, ffe_hoa.py
"""

import argparse
import time
import numpy as np

from psc_sprl import (ATT48, PARAMS, ACTIVE, SERVED, DynamicEnv, SPRLPolicy,
                      run_episode, priority_spectral_clustering)
from ffe_hoa import (FFEHOA, FFE_CNN, CNN_CONFIGS, Obstacles, Scenario,
                     Conv2D, MaxPool2, EPS)

W_INIT = (0.34, 0.46, 0.20)     # EW/AHP 平均初始权重（OAT 起点）
W_OPT = (0.30, 0.40, 0.30)      # 论文全局最优权重（流水线默认）


# ======================================================================
# 工具: 克隆预训练 CNN（Algorithm 3: F_CNN 训练一次, 各事件复用）
# ======================================================================
def clone_cnn(cnn, rng):
    new = FFE_CNN.__new__(FFE_CNN)
    new.rng, new.lr = rng, cnn.lr
    new.convs = []
    for c in cnn.convs:
        c2 = Conv2D.__new__(Conv2D)
        c2.cin, c2.cout, c2.k, c2.pad = c.cin, c.cout, c.k, c.pad
        c2.W, c2.b, c2.Z = c.W.copy(), c.b.copy(), None
        new.convs.append(c2)
    new.pools = [MaxPool2() for _ in new.convs]
    new.W1 = None if cnn.W1 is None else cnn.W1.copy()
    new.b1 = None if cnn.b1 is None else cnn.b1.copy()
    new.W2 = None if cnn.W2 is None else cnn.W2.copy()
    new.b2 = None if cnn.b2 is None else cnn.b2.copy()
    new._adam = None
    return new


# ======================================================================
# 预训练（Algorithm 3 第 2 行）
# ======================================================================
def pretrain_policy(seed, episodes, events, intensity):
    """Algorithm 1: 训练策略网络 π_θ。"""
    env = DynamicEnv(intensity=intensity,
                     initial_active=PARAMS["initial_active"], seed=seed)
    dummy = env.reset()
    policy = SPRLPolicy(len(dummy), env.n_actions(), L=PARAMS["L"], seed=seed + 1)
    rng = np.random.default_rng(seed)
    np.random.seed(seed)                                  # rng_choice 全局随机源
    for ep in range(episodes):
        m = run_episode(env, policy, rng, events, refine_steps=3)
        policy.update(m["reward"])                        # 策略梯度更新
    return policy


def pretrain_cnn(seed, cfg, iters, n_active=25):
    """Algorithm 2: 训练特征提取网络 F_CNN。"""
    scen = Scenario(seed=seed, n_active=n_active)
    hoa = FFEHOA(scen, cfg_name=cfg, N=50, L=5, T_max=iters, seed=seed,
                 verbose=False)
    hoa.run()
    return hoa.cnn


# ======================================================================
# Algorithm 1 调用: 当前时刻订单分配 DO
# ======================================================================
def allocate(env, policy, rng, greedy=True, refine_steps=3):
    """PSC 聚类 + SPRL 精调 → 订单分配 DO (uav_cluster) 与簇标签。"""
    act = env.active_orders()
    if not act:
        return {}, 0, act
    pos = np.array([o.pos for o in act])
    urg = np.array([env.urgency(o) for o in act])
    labels, K = priority_spectral_clustering(pos, urg, rng)   # PSC
    for j, o in enumerate(act):
        o.cluster = int(labels[j])
    # 未指派 UAV 按最大载重降序, 前 min(|U|,K) 架参与初始指派
    uav_cluster = {}
    for rank, u in enumerate(sorted(env.uavs, key=lambda u: -u.cap)
                             [:min(len(env.uavs), K)]):
        uav_cluster[u.idx] = rank
    if policy is not None:
        for _ in range(refine_steps):                        # SPRL 精调
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
    return uav_cluster, K, act


# ======================================================================
# 实时规划实例: 从动态环境当前状态构建（与 ffe_hoa.Scenario 同构）
# ======================================================================
class LiveScenario:
    def __init__(self, env, uav_cluster, obstacles):
        self.home = env.home.copy()
        self.obs = obstacles
        self.uav_caps = dict(enumerate(PARAMS["uav_caps"], start=1))
        self.uav_ids = list(self.uav_caps)
        self.uav_cluster = dict(uav_cluster)
        act = env.active_orders()
        self.blocks = {}
        for u in self.uav_ids:
            if u in uav_cluster:
                members = [k for k, o in enumerate(act)
                           if o.cluster == uav_cluster[u]]
            else:
                members = []
            nodes = [self.home] + [act[k].pos for k in members]
            n = len(nodes)
            DM = np.zeros((n, n))
            VI = np.zeros((n, n), int)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        DM[i, j], VI[i, j], _ = self.obs.seg(nodes[i], nodes[j])
            self.blocks[u] = dict(members=members, nodes=nodes, DM=DM, VI=VI)
        self.block_off = {}
        off = 0
        for u in self.uav_ids:
            n = len(self.blocks[u]["nodes"])
            self.block_off[u] = (off, n)
            off += n * n
        self.D = off
        self.cmax = max(1, max(len(self.blocks[u]["nodes"])
                               for u in self.uav_ids))


# ======================================================================
# 执行配送: 按规划路径 τ_u 服务订单（载荷 + 续航 + 障碍感知距离）
# ======================================================================
def execute_delivery(env, act, scen, tours):
    v = PARAMS["v_uav"]
    span = PARAMS["endurance"][1] - PARAMS["endurance"][0]
    n_served = 0
    for u in scen.uav_ids:
        if u not in scen.uav_cluster:
            continue
        blk = scen.blocks[u]
        cap = env.uavs[u - 1].cap
        endurance = PARAMS["endurance"][0] + (u - 1) * span / 9.0   # 4000-4800s
        cur, dist = 0, 0.0
        for k in tours[u][1:]:
            m = blk["members"][k - 1]
            o = act[m]
            if o.status != ACTIVE:
                continue
            leg = blk["DM"][cur, k]
            if o.payload <= cap and (dist + leg) / v <= endurance:
                cap -= o.payload
                dist += leg
                o.status = SERVED
                env.total_waiting += (env.t - o.t_create) / 60.0
                env.n_served += 1
                n_served += 1
                cur = k
    return n_served


# ======================================================================
# Algorithm 3 主循环: SPRL-HOA 滚动时域
# ======================================================================
def run_sprl_hoa(env, policy, cnn, obstacles, rng, n_events, cfg="att48",
                 ffe_iters=15, seed=0, verbose=True):
    metrics = dict(F=[], Fd=[], Fw=[], F_ff=[], F_nn=[], n_active=[], n_served=[])
    tau = {}
    w1, w2, w3 = PARAMS["W"]
    for t in range(n_events):
        env.step_event()                                  # 动态事件: 新订单/取消
        act = env.active_orders()
        if not act:
            metrics["n_active"].append(0)
            continue
        # (a) Algorithm 1: 订单分配 DO
        uav_cluster, K, act = allocate(env, policy, rng, greedy=True)
        if not uav_cluster:
            metrics["n_active"].append(len(act))
            continue
        # (b) 已指派 UAV 已含于 DO; 其余按载重降序闲置
        # (c) Algorithm 2: 每个指派 UAV 的簇内路径 τ_u
        scen = LiveScenario(env, uav_cluster, obstacles)
        hoa = FFEHOA(scen, cfg_name=cfg, N=50, L=5, T_max=ffe_iters,
                     seed=seed * 1000 + t, verbose=False,
                     cnn_every=max(1, ffe_iters // 3), penalty=0.0)
        # 注: DM 已含障碍绕行长度(路径天然可行), 流水线中惩罚项冗余且会使
        # 搜索目标偏离报告指标 F, 故置 0; 搜索目标与报告 F 完全一致
        hoa.cnn = clone_cnn(cnn, np.random.default_rng(seed * 1000 + t))
        hoa.cnn_frozen = True                             # 复用预训练 F_CNN
        hoa.X[0] = 0.0    # 精英种子: 全 0 编码解码为最近邻路径, 保证输出 ≥ NN
        best_X, _, _ = hoa.run()
        tours, F_d, F_w, viol = hoa.decode(best_X)
        F_e = PARAMS["e_rate"] * F_d
        F = w1 * F_d + w2 * F_e + w3 * F_w
        # NN 基线（同一分配下的最近邻路径, 供对比）
        _, Fd_nn, Fw_nn, _ = hoa.decode(np.zeros(scen.D, np.float32))
        F_nn = w1 * Fd_nn + w2 * PARAMS["e_rate"] * Fd_nn + w3 * Fw_nn
        # (d) 执行配送并更新系统状态
        n_srv = execute_delivery(env, act, scen, tours)
        # (e) 完成订单已在 SERVED 中移出活跃集合
        tau = tours
        last_act = act
        last_scen = scen
        metrics["F"].append(F)
        metrics["Fd"].append(F_d)
        metrics["Fw"].append(F_w)
        metrics["F_ff"].append(F)
        metrics["F_nn"].append(F_nn)
        metrics["n_active"].append(len(act))
        metrics["n_served"].append(n_srv)
        if verbose and (t + 1) % max(1, n_events // 10) == 0:
            print(f"  t={t+1:3d}  |C^a|={len(act):3d}  F={F:10.2f}  "
                  f"NN基线F={F_nn:10.2f}  本事件服务={n_srv:3d}  "
                  f"累计送达率={env.n_served/max(env.n_generated,1)*100:5.1f}%")
    return metrics, tau, last_scen, last_act


# ======================================================================
# OAT 权重敏感性分析（Fig. weight 复现）
# ======================================================================
def oat_study(seed=0, iters=60, n_active=25, cfg="att48"):
    """一次一因子: 初始权重 (0.34,0.46,0.20), 10% 增量, 重归一化。
    在同一 Att48 静态实例上评估各权重组合的总成本 F。"""
    base = np.array(W_INIT)
    mults = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]           # ±10% 步进
    scen = Scenario(seed=seed, n_active=n_active)          # 固定同一实例

    def eval_weights(w):
        PARAMS["W"] = tuple(float(x) for x in w)
        hoa = FFEHOA(scen, cfg_name=cfg, N=50, L=5, T_max=iters,
                     seed=seed, verbose=False)              # 同种子 → 仅权重不同
        best_X, _, _ = hoa.run()
        _, F_d, F_w, viol = hoa.decode(best_X)
        F_e = PARAMS["e_rate"] * F_d
        F = w[0] * F_d + w[1] * F_e + w[2] * F_w + 5000.0 * viol
        return F, F_d, F_w

    rows = []
    print("OAT 敏感性分析: 初始权重 (W1,W2,W3) = (0.34, 0.46, 0.20) [EW/AHP 平均]")
    print(f"{'变化因子':<8}{'倍率':>6}   {'(W1, W2, W3)':<26}{'F ↓':>12}"
          f"{'F_d':>12}{'F_w':>10}")
    print("-" * 76)
    for k, name in enumerate(["W1", "W2", "W3"]):
        for m in mults:
            v = base.copy()
            v[k] = base[k] * m
            v = v / v.sum()                                # 重归一化
            F, F_d, F_w = eval_weights(v)
            rows.append((name, m, v.copy(), F))
            print(f"{name:<8}{m:>6.1f}x  ({v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f})   "
                  f"{F:>12.2f}{F_d:>12.2f}{F_w:>10.2f}")
    # 论文全局最优权重
    F_opt, Fd_opt, Fw_opt = eval_weights(W_OPT)
    rows.append(("paper", 1.0, np.array(W_OPT), F_opt))
    print(f"{'论文最优':<8}{1.0:>6.1f}x  ({W_OPT[0]:.3f}, {W_OPT[1]:.3f}, "
          f"{W_OPT[2]:.3f})   {F_opt:>12.2f}{Fd_opt:>12.2f}{Fw_opt:>10.2f}")
    best = min(rows, key=lambda r: r[3])
    print("-" * 76)
    print(f"OAT 搜索最优: ({best[2][0]:.3f}, {best[2][1]:.3f}, {best[2][2]:.3f}) "
          f"F={best[3]:.2f}  [变化因子={best[0]}, 倍率={best[1]}x]")
    print(f"论文报告全局最优: (0.30, 0.40, 0.30)  F={F_opt:.2f}")
    PARAMS["W"] = W_OPT                                    # 恢复论文最优
    return rows


# ======================================================================
# 主入口
# ======================================================================
def main():
    ap = argparse.ArgumentParser(description="SPRL-HOA (Algorithm 3, paper parameters)")
    ap.add_argument("--events", type=int, default=100, help="T_max 事件数（默认 100）")
    ap.add_argument("--episodes", type=int, default=60, help="π_θ 预训练回合数")
    ap.add_argument("--cnn-iters", type=int, default=30, help="F_CNN 预训练迭代")
    ap.add_argument("--ffe-iters", type=int, default=15, help="每事件 FFE-HOA 迭代")
    ap.add_argument("--intensity", type=int, default=5, choices=[5, 10, 15, 20],
                    help="每事件订单强度（默认 5）")
    ap.add_argument("--config", default="att48", choices=list(CNN_CONFIGS),
                    help="CNN 配置（默认 att48）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--full", action="store_true",
                    help="论文全量训练(T'_max=200 episodes, CNN 300 iters)")
    ap.add_argument("--oat", action="store_true",
                    help="运行 W1-W3 OAT 敏感性分析(Fig. weight)")
    ap.add_argument("--oat-iters", type=int, default=60, help="OAT 每配置迭代数")
    args = ap.parse_args()

    if args.oat:
        oat_study(seed=args.seed, iters=args.oat_iters, cfg=args.config)
        return

    episodes = 200 if args.full else args.episodes
    cnn_iters = 300 if args.full else args.cnn_iters

    print("=" * 68)
    print("SPRL-HOA (Algorithm 3): PSC-SPRL 订单分配 + FFE-HOA 路径规划")
    print(f"Att48, 强度={args.intensity}, W={PARAMS['W']}, T_max={args.events}, "
          f"障碍物=10 个圆柱(10-100m), 10 架异构 UAV")
    print("=" * 68)

    # ---- 预训练 π_θ (Algorithm 1) 与 F_CNN (Algorithm 2) ----
    t0 = time.time()
    print(f"[1/3] 预训练策略网络 π_θ (PSC-SPRL, {episodes} episodes)...")
    policy = pretrain_policy(args.seed, episodes, args.events, args.intensity)
    print(f"      完成, 耗时 {time.time()-t0:.1f}s")
    t1 = time.time()
    print(f"[2/3] 预训练特征网络 F_CNN (FFE-HOA, {cnn_iters} iters)...")
    cnn = pretrain_cnn(args.seed, args.config, cnn_iters)
    print(f"      完成, 耗时 {time.time()-t1:.1f}s")

    # ---- Algorithm 3 滚动时域主循环 ----
    t2 = time.time()
    print(f"[3/3] 滚动时域执行 (T_max={args.events} 事件, Δt=100s)...")
    env = DynamicEnv(intensity=args.intensity,
                     initial_active=PARAMS["initial_active"],
                     seed=args.seed + 500)
    rng = np.random.default_rng(args.seed)
    lo, hi = ATT48.min(0), ATT48.max(0)
    obstacles = Obstacles(rng, (lo, hi), env.home,
                          np.vstack([env.home] + [o.pos for o in env.orders]))
    metrics, tau, scen, last_act = run_sprl_hoa(env, policy, cnn, obstacles, rng,
                                                args.events, cfg=args.config,
                                                ffe_iters=args.ffe_iters,
                                                seed=args.seed)
    wall = time.time() - t2

    # ---- 汇总 ----
    F = np.array(metrics["F"]) if metrics["F"] else np.zeros(1)
    Fd = np.array(metrics["Fd"]) if metrics["Fd"] else np.zeros(1)
    Fw = np.array(metrics["Fw"]) if metrics["Fw"] else np.zeros(1)
    Fnn = np.array(metrics["F_nn"]) if metrics["F_nn"] else np.zeros(1)
    served = env.n_served / max(env.n_generated, 1)
    print(f"\n{'='*68}\nSPRL-HOA 结果 (Att48, intensity={args.intensity}, "
          f"seed={args.seed})\n{'-'*68}")
    print(f"{'指标':<36}{'数值':>16}")
    print("-" * 68)
    print(f"{'执行事件数':<38}{len(metrics['F']):>13d}")
    print(f"{'生成订单总数 / 送达数':<34}{env.n_generated:>8d} / {env.n_served}")
    print(f"{'Served Ratio 送达率 ↑':<36}{served*100:>13.2f}%")
    print(f"{'平均 |C^a| 活跃订单':<36}{np.mean(metrics['n_active']):>13.1f}")
    print(f"{'F_d 平均路径成本 ↓':<36}{Fd.mean():>15.2f} m")
    print(f"{'F_w 平均等待成本 ↓':<36}{Fw.mean():>15.2f} min")
    print(f"{'F  综合成本 (W=0.30/0.40/0.30) ↓':<36}{F.mean():>15.2f}")
    print(f"{'NN 基线 F (同分配) ↓':<36}{Fnn.mean():>15.2f}")
    print(f"{'FFE-HOA 相对 NN 改善 ↑':<36}"
          f"{(1-F.mean()/(Fnn.mean()+EPS))*100:>13.2f}%")
    print(f"{'平均累计等待 (分钟/事件)':<34}{env.total_waiting/max(1,len(metrics['F'])):>11.1f}")
    print(f"{'主循环耗时 (s)':<36}{wall:>16.1f}")
    print("-" * 68)
    print("末事件各 UAV 路径 τ_u:")
    for u in sorted(scen.uav_cluster, key=lambda u: scen.uav_cluster[u]):
        blk = scen.blocks[u]
        seq = tau.get(u, [0])
        names = ["仓库"] + [f"订单{blk['members'][k - 1]}" for k in seq[1:]]
        print(f"  UAV {u:2d} (cap={scen.uav_caps[u]:>2d}kg, 簇"
              f"{scen.uav_cluster[u]}): {' → '.join(names)}")
    for u in scen.uav_ids:
        if u not in scen.uav_cluster:
            print(f"  UAV {u:2d} (cap={scen.uav_caps[u]:>2d}kg): 未指派(闲置)")


if __name__ == "__main__":
    main()
