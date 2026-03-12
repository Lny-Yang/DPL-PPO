# -*- coding: utf-8 -*-
"""
批量测试多个SB3 PPO checkpoint模型（自动发现全部模型）
- 逐个调用同目录的 test_phase1_SB3.py 进行评测
- 仅收集本次批量运行目录下的结果，生成对比报告
- 选出成功率最高的模型（并列时按平均奖励最高打破）
"""
import subprocess
import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
from datetime import datetime

def test_checkpoint(test_script: Path, model_path: Path, save_dir: Path, episodes: int = 50) -> bool:
    """测试单个SB3 checkpoint"""
    print(f"\n{'='*80}")
    print(f"测试模型: {model_path}")
    print(f"结果目录: {save_dir}")
    print(f"{'='*80}")

    cmd = [
        sys.executable,
        str(test_script),
        "--model", str(model_path.with_suffix('')),  # 传不带.zip的基名，SB3会自动补
        "--episodes", str(episodes),
        "--no_render",
        "--no_show_plot",
        "--save_dir", str(save_dir)
    ]

    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 测试失败: {e}")
        return False

def collect_results(log_dir: Path):
    """收集本次批量目录中的测试结果"""
    results = []
    if not log_dir.exists():
        print(f"⚠️ 测试结果文件夹不存在: {log_dir}")
        return results

    for result_file in log_dir.glob("phase1_test_results_*.json"):
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            model_name = result_file.stem.replace("phase1_test_results_", "")

            num_episodes = len(data.get('episode_rewards', []))
            if num_episodes == 0:
                continue

            success_rate = data.get('success_count', 0) / num_episodes
            collision_rate = data.get('collision_count', 0) / num_episodes
            timeout_rate = data.get('timeout_count', 0) / num_episodes
            boundary_collision_rate = data.get('boundary_collision_count', 0) / num_episodes
            physical_collision_rate = data.get('physical_collision_count', 0) / num_episodes

            avg_reward = float(np.mean(data.get('episode_rewards', [0.0])))
            std_reward = float(np.std(data.get('episode_rewards', [0.0])))
            avg_length = float(np.mean(data.get('episode_lengths', [0])))

            avg_min_depth = float(np.mean(data.get('min_depths', []))) if data.get('min_depths') else 0.0
            avg_goal_distance = float(np.mean(data.get('goal_distances', []))) if data.get('goal_distances') else 0.0

            reward_components = data.get('reward_components', {})
            avg_success_reward = float(np.mean(reward_components.get('success', [0.0])))
            avg_crash_reward = float(np.mean(reward_components.get('crash', [0.0])))
            avg_dense_reward = float(np.mean(reward_components.get('dense', [0.0])))

            results.append({
                'model': model_name,
                'episodes': num_episodes,
                'success_rate': success_rate,
                'collision_rate': collision_rate,
                'timeout_rate': timeout_rate,
                'boundary_collision_rate': boundary_collision_rate,
                'physical_collision_rate': physical_collision_rate,
                'avg_reward': avg_reward,
                'std_reward': std_reward,
                'avg_length': avg_length,
                'avg_min_depth': avg_min_depth,
                'avg_goal_distance': avg_goal_distance,
                'avg_success_reward': avg_success_reward,
                'avg_crash_reward': avg_crash_reward,
                'avg_dense_reward': avg_dense_reward
            })
        except Exception as e:
            print(f"⚠️ 处理文件 {result_file} 时出错: {e}")
            continue

    return results

def generate_comparison_report(results, out_dir: Path):
    """生成对比报告并返回最佳模型信息"""
    if not results:
        print("❌ 没有可用的测试结果")
        return None

    df = pd.DataFrame(results)

    # 主排序：成功率；副排序：平均奖励
    df = df.sort_values(['success_rate', 'avg_reward'], ascending=[False, False]).reset_index(drop=True)

    print("\n" + "="*140)
    print("📊 SB3 PPO模型对比分析报告（按成功率→平均奖励排序）")
    print("="*140)

    print("\n🏆 按成功率排序:")
    print("-"*140)
    print(f"{'模型名称':<45} | {'回合':>5} | {'成功率':>8} | {'碰撞率':>8} | {'超时率':>8} | {'平均奖励':>10} | {'奖励标准差':>10} | {'平均步数':>8}")
    print("-"*140)
    for _, row in df.iterrows():
        print(f"{row['model']:<45} | "
              f"{row['episodes']:>5d} | "
              f"{row['success_rate']:>7.1%} | "
              f"{row['collision_rate']:>7.1%} | "
              f"{row['timeout_rate']:>7.1%} | "
              f"{row['avg_reward']:>10.2f} | "
              f"{row['std_reward']:>10.2f} | "
              f"{row['avg_length']:>8.1f}")
    print("-"*140)

    print("\n📈 详细分析:")
    print("-"*140)
    print(f"{'模型名称':<45} | {'边界碰撞':>9} | {'物理碰撞':>9} | {'最小深度':>9} | {'目标距离':>9}")
    print("-"*140)
    for _, row in df.iterrows():
        print(f"{row['model']:<45} | "
              f"{row['boundary_collision_rate']:>8.1%} | "
              f"{row['physical_collision_rate']:>8.1%} | "
              f"{row['avg_min_depth']:>8.2f}m | "
              f"{row['avg_goal_distance']:>8.2f}m")
    print("-"*140)

    print("\n💰 奖励分量分析:")
    print("-"*140)
    print(f"{'模型名称':<45} | {'成功奖励':>10} | {'碰撞惩罚':>10} | {'密集奖励':>10}")
    print("-"*140)
    for _, row in df.iterrows():
        print(f"{row['model']:<45} | "
              f"{row['avg_success_reward']:>10.2f} | "
              f"{row['avg_crash_reward']:>10.2f} | "
              f"{row['avg_dense_reward']:>10.2f}")
    print("-"*140)

    best_model = df.iloc[0]
    print(f"\n🏆 最佳模型（按成功率，其次按平均奖励）: {best_model['model']}")
    print(f"   成功率: {best_model['success_rate']:.1%}")
    print(f"   平均奖励: {best_model['avg_reward']:.2f} ± {best_model['std_reward']:.2f}")
    print(f"   平均步数: {best_model['avg_length']:.1f}")

    # 保存报告
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "model_comparison_report.csv"
    df.to_csv(report_path, index=False, encoding='utf-8-sig')
    print(f"\n📄 详细对比报告已保存: {report_path}")

    summary_cols = ['model', 'episodes', 'success_rate', 'collision_rate', 'timeout_rate', 'avg_reward', 'avg_length']
    simplified_report_path = out_dir / "model_comparison_summary.csv"
    df[summary_cols].to_csv(simplified_report_path, index=False, encoding='utf-8-sig')
    print(f"📄 简化报告已保存: {simplified_report_path}")
    print("="*140)

    return best_model

def main():
    print("🚀 开始批量测试SB3 PPO模型（自动发现全部checkpoint）")

    # 以脚本目录为基准，确保路径正确
    script_dir = Path(__file__).parent.resolve()
    test_script = script_dir / "test_phase1_SB3.py"

    if not test_script.exists():
        print(f"❌ 未找到测试脚本: {test_script}")
        return

    # 模型目录：D3PO/muti_formation-D3PO/agent/model_SB3
    model_dir = script_dir / "agent" / "model_SB3"
    if not model_dir.exists():
        print(f"❌ 模型目录不存在: {model_dir}")
        return

    # 批次结果目录：D3PO/muti_formation-D3PO/agent/log_SB3/batch_YYYYmmdd_HHMMSS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_dir = script_dir / "agent" / "log_SB3" / f"batch_{timestamp}"
    run_log_dir.mkdir(parents=True, exist_ok=True)

    # 自动发现全部模型：leader_phase1_*.zip
    checkpoints_zip = sorted(model_dir.glob("leader_phase1_*.zip"))
    if not checkpoints_zip:
        print(f"❌ 未找到任何模型zip文件（期望: {model_dir}/leader_phase1_*.zip）")
        return

    checkpoints = [p for p in checkpoints_zip]  # 传入时会去掉后缀
    print(f"找到 {len(checkpoints)} 个模型待测试：")
    print([p.stem for p in checkpoints])

    # 逐个测试
    episodes_per_model = 200  # 可按需调整
    tested = 0
    for i, checkpoint in enumerate(checkpoints, 1):
        print(f"\n进度: [{i}/{len(checkpoints)}]")
        ok = test_checkpoint(test_script, checkpoint, run_log_dir, episodes=episodes_per_model)
        if ok:
            tested += 1

    if tested == 0:
        print("❌ 所有模型测试均失败或无结果")
        return

    # 收集并分析本次批量目录中的结果
    print("\n" + "="*80)
    print("📊 收集测试结果...")
    print("="*80)
    results = collect_results(run_log_dir)

    if results:
        best = generate_comparison_report(results, out_dir=run_log_dir)
        if best is not None:
            # 推导最佳模型的实际checkpoint路径
            best_name = best['model']  # 如 leader_phase1_episode_85000
            best_ckpt = model_dir / f"{best_name}.zip"
            print(f"\n✅ 本次批量测试最佳模型checkpoint: {best_ckpt}")
    else:
        print("❌ 未找到测试结果JSON")
        print(f"请检查结果是否保存到: {run_log_dir}")

if __name__ == '__main__':
    main()