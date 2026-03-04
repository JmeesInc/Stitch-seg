#!/usr/bin/env python3
"""
ONNX Runtime プロファイル JSON の解析スクリプト

Usage:
    python analyze_profile.py onnxruntime_profile__2026-02-08_15-08-34.json

推論ボトルネックを以下の観点で解析:
  1. 全体のサマリー (Session / Kernel 分類)
  2. オペレータ (op_name) ごとの合計時間 Top-N
  3. 個別ノード (name) ごとの時間 Top-N
  4. モジュール (prefix) 別の合計時間 → inferencer_toonnx_only.py のどこが重いか
  5. CPU vs GPU (provider) 実行の割合
  6. fence_before / fence_after の overhead 解析
  7. メモリ (output_size / activation_size) 上位ノード
"""

import json
import sys
import re
from collections import defaultdict
from pathlib import Path


def load_profile(path: str) -> list:
    print(f"Loading {path} ...")
    with open(path) as f:
        data = json.load(f)
    print(f"  -> {len(data)} events loaded")
    return data


def classify_events(events: list):
    """イベントをカテゴリ別に分類"""
    session_events = []
    node_kernel = []       # *_kernel_time
    node_fence_before = [] # *_fence_before
    node_fence_after = []  # *_fence_after
    other = []

    for ev in events:
        cat = ev.get("cat", "")
        name = ev.get("name", "")
        if cat == "Session":
            session_events.append(ev)
        elif cat == "Node":
            if name.endswith("_kernel_time"):
                node_kernel.append(ev)
            elif name.endswith("_fence_before"):
                node_fence_before.append(ev)
            elif name.endswith("_fence_after"):
                node_fence_after.append(ev)
            else:
                other.append(ev)
        else:
            other.append(ev)

    return session_events, node_kernel, node_fence_before, node_fence_after, other


def strip_suffix(name: str) -> str:
    """_kernel_time, _fence_before, _fence_after を除去してノード名を返す"""
    for suffix in ("_kernel_time", "_fence_before", "_fence_after"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def get_module_prefix(node_name: str) -> str:
    """
    ノード名からモジュールプレフィックスを抽出する。
    例:
      /extractor/block1/conv/Conv      -> extractor
      /matcher/transformers.0/...       -> matcher
      /masking_model/encoder/...        -> masking_model
      /masking_model2/...               -> masking_model2
      /Div                              -> (top-level)
    """
    name = strip_suffix(node_name)
    # 先頭の / を除去
    if name.startswith("/"):
        name = name[1:]
    parts = name.split("/")
    if len(parts) >= 2:
        return parts[0]
    return "(top-level)"


def get_submodule_prefix(node_name: str, depth: int = 2) -> str:
    """depth 階層までのプレフィックスを返す"""
    name = strip_suffix(node_name)
    if name.startswith("/"):
        name = name[1:]
    parts = name.split("/")
    if len(parts) > depth:
        return "/".join(parts[:depth])
    return "/".join(parts)


def format_us(us: float) -> str:
    """マイクロ秒を読みやすい文字列に変換"""
    if us >= 1_000_000:
        return f"{us / 1_000_000:.3f} s"
    elif us >= 1_000:
        return f"{us / 1_000:.3f} ms"
    else:
        return f"{us:.1f} µs"


def format_bytes(b: float) -> str:
    if b >= 1e9:
        return f"{b / 1e9:.2f} GB"
    elif b >= 1e6:
        return f"{b / 1e6:.2f} MB"
    elif b >= 1e3:
        return f"{b / 1e3:.2f} KB"
    else:
        return f"{b:.0f} B"


def print_separator(title: str):
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")


def analyze(profile_path: str):
    events = load_profile(profile_path)
    session_events, node_kernel, node_fence_before, node_fence_after, other = classify_events(events)

    # =========================================================================
    # 1. 全体サマリー
    # =========================================================================
    print_separator("1. 全体サマリー")

    total_kernel_us = sum(ev.get("dur", 0) for ev in node_kernel)
    total_fence_before_us = sum(ev.get("dur", 0) for ev in node_fence_before)
    total_fence_after_us = sum(ev.get("dur", 0) for ev in node_fence_after)

    # session events
    for sev in session_events:
        print(f"  {sev['name']}: {format_us(sev.get('dur', 0))}")

    print(f"\n  カーネル実行ノード数: {len(node_kernel)}")
    print(f"  合計カーネル時間:     {format_us(total_kernel_us)}")
    print(f"  合計 fence_before:    {format_us(total_fence_before_us)}")
    print(f"  合計 fence_after:     {format_us(total_fence_after_us)}")
    print(f"  fence overhead 合計:  {format_us(total_fence_before_us + total_fence_after_us)}")

    # =========================================================================
    # 2. Provider (CPU vs GPU) 別の合計時間
    # =========================================================================
    print_separator("2. 実行プロバイダ別の合計時間 (CPU vs GPU)")

    provider_time = defaultdict(float)
    provider_count = defaultdict(int)
    for ev in node_kernel:
        prov = ev.get("args", {}).get("provider", "unknown")
        provider_time[prov] += ev.get("dur", 0)
        provider_count[prov] += 1

    for prov, t in sorted(provider_time.items(), key=lambda x: -x[1]):
        pct = t / total_kernel_us * 100 if total_kernel_us > 0 else 0
        print(f"  {prov}: {format_us(t)} ({pct:.1f}%)  [{provider_count[prov]} ops]")

    # =========================================================================
    # 3. op_name 別の合計時間 Top-30
    # =========================================================================
    print_separator("3. オペレータ (op_name) 別の合計時間 Top-30")

    op_time = defaultdict(float)
    op_count = defaultdict(int)
    for ev in node_kernel:
        op = ev.get("args", {}).get("op_name", "unknown")
        op_time[op] += ev.get("dur", 0)
        op_count[op] += 1

    top_ops = sorted(op_time.items(), key=lambda x: -x[1])[:30]
    print(f"  {'Op Name':<30} {'Total Time':>12} {'Count':>8} {'Avg':>12} {'% of Total':>10}")
    print(f"  {'-'*30} {'-'*12} {'-'*8} {'-'*12} {'-'*10}")
    for op, t in top_ops:
        avg = t / op_count[op] if op_count[op] > 0 else 0
        pct = t / total_kernel_us * 100 if total_kernel_us > 0 else 0
        print(f"  {op:<30} {format_us(t):>12} {op_count[op]:>8} {format_us(avg):>12} {pct:>9.1f}%")

    # =========================================================================
    # 4. 個別ノード Top-50 (最も時間がかかったノード)
    # =========================================================================
    print_separator("4. 個別ノード (name) 実行時間 Top-50")

    node_sorted = sorted(node_kernel, key=lambda x: -x.get("dur", 0))[:50]
    print(f"  {'#':>3} {'Node Name':<70} {'Time':>12} {'Op':>15} {'Provider':>10}")
    print(f"  {'-'*3} {'-'*70} {'-'*12} {'-'*15} {'-'*10}")
    for i, ev in enumerate(node_sorted):
        name = strip_suffix(ev.get("name", ""))
        dur = ev.get("dur", 0)
        op = ev.get("args", {}).get("op_name", "?")
        prov = ev.get("args", {}).get("provider", "?")
        pct = dur / total_kernel_us * 100 if total_kernel_us > 0 else 0
        # Truncate long names
        if len(name) > 70:
            name = "..." + name[-67:]
        print(f"  {i+1:>3} {name:<70} {format_us(dur):>12} {op:>15} {prov:>10}")

    # =========================================================================
    # 5. モジュール (prefix) 別の合計時間
    #    → inferencer_toonnx_only.py のどこが重いかに直結
    # =========================================================================
    print_separator("5. モジュール別の合計時間 (inferencer_toonnx_only.py のボトルネック)")

    module_time = defaultdict(float)
    module_count = defaultdict(int)
    for ev in node_kernel:
        mod = get_module_prefix(ev.get("name", ""))
        module_time[mod] += ev.get("dur", 0)
        module_count[mod] += 1

    top_modules = sorted(module_time.items(), key=lambda x: -x[1])
    print(f"\n  {'Module':<35} {'Total Time':>12} {'Count':>8} {'% of Total':>10}")
    print(f"  {'-'*35} {'-'*12} {'-'*8} {'-'*10}")
    for mod, t in top_modules:
        pct = t / total_kernel_us * 100 if total_kernel_us > 0 else 0
        print(f"  {mod:<35} {format_us(t):>12} {module_count[mod]:>8} {pct:>9.1f}%")

    # モジュール→コード対応表
    print(f"\n  --- モジュール → inferencer_toonnx_only.py コード対応 ---")
    mapping = {
        "extractor":      "L158: self.extractor(frame_u / 255.0)  — ALIKED 特徴抽出",
        "matcher":        "L166: self.matcher(kpts, descs)  — LightGlue マッチング",
        "masking_model":  "L111: self.masking_model(input_img)  — ツールマスク推定 (UNet1)",
        "masking_model2": "L112: self.masking_model2(input_img)  — ツールマスク推定 (UNet2)",
        "(top-level)":    "前処理・後処理テンソル演算 (warp, blend, homography等)",
    }
    for mod, _ in top_modules:
        desc = mapping.get(mod, f"(サブモジュール: {mod})")
        pct = module_time[mod] / total_kernel_us * 100 if total_kernel_us > 0 else 0
        print(f"    {mod:<25} → {desc}  [{pct:.1f}%]")

    # =========================================================================
    # 6. サブモジュール (depth=2) 別の詳細
    # =========================================================================
    print_separator("6. サブモジュール (depth=2) 別の時間 Top-40")

    submod_time = defaultdict(float)
    submod_count = defaultdict(int)
    for ev in node_kernel:
        sm = get_submodule_prefix(ev.get("name", ""), depth=2)
        submod_time[sm] += ev.get("dur", 0)
        submod_count[sm] += 1

    top_submods = sorted(submod_time.items(), key=lambda x: -x[1])[:40]
    print(f"  {'Submodule':<50} {'Total Time':>12} {'Count':>8} {'% of Total':>10}")
    print(f"  {'-'*50} {'-'*12} {'-'*8} {'-'*10}")
    for sm, t in top_submods:
        pct = t / total_kernel_us * 100 if total_kernel_us > 0 else 0
        print(f"  {sm:<50} {format_us(t):>12} {submod_count[sm]:>8} {pct:>9.1f}%")

    # =========================================================================
    # 7. CPUExecutionProvider で実行されているノード Top-30
    #    → これらは GPU に移すべき可能性がある
    # =========================================================================
    print_separator("7. CPU で実行されているノード Top-30 (GPU移行候補)")

    cpu_nodes = [ev for ev in node_kernel if ev.get("args", {}).get("provider", "") == "CPUExecutionProvider"]
    cpu_sorted = sorted(cpu_nodes, key=lambda x: -x.get("dur", 0))[:30]
    total_cpu = sum(ev.get("dur", 0) for ev in cpu_nodes)

    if cpu_nodes:
        print(f"  CPU ノード数: {len(cpu_nodes)}, 合計時間: {format_us(total_cpu)}")
        print(f"\n  {'#':>3} {'Node Name':<60} {'Time':>12} {'Op':>15}")
        print(f"  {'-'*3} {'-'*60} {'-'*12} {'-'*15}")
        for i, ev in enumerate(cpu_sorted):
            name = strip_suffix(ev.get("name", ""))
            dur = ev.get("dur", 0)
            op = ev.get("args", {}).get("op_name", "?")
            if len(name) > 60:
                name = "..." + name[-57:]
            print(f"  {i+1:>3} {name:<60} {format_us(dur):>12} {op:>15}")
    else:
        print("  CPU で実行されたノードはありません。")

    # =========================================================================
    # 8. fence_before / fence_after が大きいノード Top-20
    #    → GPU sync overhead / data transfer bottleneck
    # =========================================================================
    print_separator("8. fence_before / fence_after overhead Top-20")

    fence_map = defaultdict(lambda: {"before": 0, "after": 0, "kernel": 0})
    for ev in node_fence_before:
        key = strip_suffix(ev.get("name", ""))
        fence_map[key]["before"] += ev.get("dur", 0)
    for ev in node_fence_after:
        key = strip_suffix(ev.get("name", ""))
        fence_map[key]["after"] += ev.get("dur", 0)
    for ev in node_kernel:
        key = strip_suffix(ev.get("name", ""))
        fence_map[key]["kernel"] += ev.get("dur", 0)

    # Sort by total fence overhead
    fence_sorted = sorted(fence_map.items(), key=lambda x: -(x[1]["before"] + x[1]["after"]))[:20]
    print(f"  {'Node Name':<55} {'fence_before':>12} {'fence_after':>12} {'kernel':>12} {'overhead%':>10}")
    print(f"  {'-'*55} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")
    for name, d in fence_sorted:
        fb = d["before"]
        fa = d["after"]
        kern = d["kernel"]
        total = fb + fa + kern
        overhead_pct = (fb + fa) / total * 100 if total > 0 else 0
        if len(name) > 55:
            name = "..." + name[-52:]
        print(f"  {name:<55} {format_us(fb):>12} {format_us(fa):>12} {format_us(kern):>12} {overhead_pct:>9.1f}%")

    # =========================================================================
    # 9. メモリ (output_size) 上位ノード Top-20
    # =========================================================================
    print_separator("9. メモリ (output_size) 上位ノード Top-20")

    mem_nodes = []
    for ev in node_kernel:
        osize = ev.get("args", {}).get("output_size", "0")
        try:
            osize = int(osize)
        except (ValueError, TypeError):
            osize = 0
        if osize > 0:
            mem_nodes.append((ev, osize))

    mem_sorted = sorted(mem_nodes, key=lambda x: -x[1])[:20]
    print(f"  {'Node Name':<55} {'Output Size':>12} {'Op':>15} {'Time':>12}")
    print(f"  {'-'*55} {'-'*12} {'-'*15} {'-'*12}")
    for ev, osize in mem_sorted:
        name = strip_suffix(ev.get("name", ""))
        op = ev.get("args", {}).get("op_name", "?")
        dur = ev.get("dur", 0)
        if len(name) > 55:
            name = "..." + name[-52:]
        print(f"  {name:<55} {format_bytes(osize):>12} {op:>15} {format_us(dur):>12}")

    # =========================================================================
    # 10. 実行タイムライン (推論ごとのサマリー)
    # =========================================================================
    print_separator("10. 推論イテレーション分析")
    
    # kernel events をタイムスタンプ順にソート
    kernel_sorted_ts = sorted(node_kernel, key=lambda x: x.get("ts", 0))
    
    if len(kernel_sorted_ts) > 0:
        first_ts = kernel_sorted_ts[0].get("ts", 0)
        last_ev = kernel_sorted_ts[-1]
        last_ts = last_ev.get("ts", 0) + last_ev.get("dur", 0)
        wall_time = last_ts - first_ts
        
        print(f"  最初のカーネル ts: {format_us(first_ts)}")
        print(f"  最後のカーネル ts: {format_us(last_ts)}")
        print(f"  全体ウォール時間:  {format_us(wall_time)}")
        print(f"  カーネル合計時間:  {format_us(total_kernel_us)}")
        print(f"  GPU使用率 (概算):  {total_kernel_us / wall_time * 100:.1f}%" if wall_time > 0 else "")

        # 推論セッション（大きなギャップで分割）を検出
        # session_initialization 後の最初のノードから始めて、大きな gap (> 100ms) で分割
        gaps = []
        for i in range(1, len(kernel_sorted_ts)):
            prev_end = kernel_sorted_ts[i - 1].get("ts", 0) + kernel_sorted_ts[i - 1].get("dur", 0)
            curr_start = kernel_sorted_ts[i].get("ts", 0)
            gap = curr_start - prev_end
            if gap > 100_000:  # > 100ms gap
                gaps.append((i, gap))

        if gaps:
            print(f"\n  推論イテレーション間のギャップ (> 100ms): {len(gaps)} 個検出")
            # Build iteration boundaries
            boundaries = [0] + [g[0] for g in gaps] + [len(kernel_sorted_ts)]
            n_iters = len(boundaries) - 1
            print(f"  推定イテレーション数: {n_iters}")
            
            # Analyze each iteration
            print(f"\n  {'Iter':>5} {'# Ops':>8} {'Kernel Time':>14} {'Wall Time':>14}")
            print(f"  {'-'*5} {'-'*8} {'-'*14} {'-'*14}")
            for it in range(min(n_iters, 20)):  # Show first 20 iterations
                start_idx = boundaries[it]
                end_idx = boundaries[it + 1]
                iter_events = kernel_sorted_ts[start_idx:end_idx]
                iter_kernel = sum(ev.get("dur", 0) for ev in iter_events)
                iter_wall = (iter_events[-1].get("ts", 0) + iter_events[-1].get("dur", 0)) - iter_events[0].get("ts", 0)
                print(f"  {it+1:>5} {len(iter_events):>8} {format_us(iter_kernel):>14} {format_us(iter_wall):>14}")
            
            if n_iters > 20:
                print(f"  ... ({n_iters - 20} iterations omitted)")

    # =========================================================================
    # 11. ボトルネックまとめ
    # =========================================================================
    print_separator("11. ボトルネック分析まとめ")

    print("\n  [モジュール別ランキング]")
    for i, (mod, t) in enumerate(top_modules[:5]):
        pct = t / total_kernel_us * 100 if total_kernel_us > 0 else 0
        desc = mapping.get(mod, mod)
        print(f"    #{i+1}  {mod:<25} {format_us(t):>12} ({pct:.1f}%)  — {desc}")

    print("\n  [最も遅い個別ノード Top-10]")
    for i, ev in enumerate(node_sorted[:10]):
        name = strip_suffix(ev.get("name", ""))
        dur = ev.get("dur", 0)
        op = ev.get("args", {}).get("op_name", "?")
        prov = ev.get("args", {}).get("provider", "?")
        pct = dur / total_kernel_us * 100 if total_kernel_us > 0 else 0
        if len(name) > 55:
            name = "..." + name[-52:]
        print(f"    #{i+1}  {name:<55} {format_us(dur):>12} ({pct:.1f}%) [{op}, {prov}]")

    if total_cpu > 0:
        cpu_pct = total_cpu / total_kernel_us * 100
        print(f"\n  [CPU実行 overhead]")
        print(f"    CPU合計: {format_us(total_cpu)} ({cpu_pct:.1f}%) — GPU移行で高速化の余地あり")

    fence_total = total_fence_before_us + total_fence_after_us
    if fence_total > 0:
        print(f"\n  [Fence (sync) overhead]")
        print(f"    合計: {format_us(fence_total)} — CPU↔GPU同期のコスト")

    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <profile.json>")
        sys.exit(1)
    analyze(sys.argv[1])
