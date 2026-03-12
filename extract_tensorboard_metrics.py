"""
从TensorBoard日志中提取训练指标

用于提取之前训练的详细指标数据
"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse

try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    print("❌ 需要安装tensorboard: pip install tensorboard")
    exit(1)

# 设置中文字体和图表参数
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False  # 用普通连字符替代Unicode负号
plt.rcParams['font.family'] = 'sans-serif'

import warnings
import logging
# 完全抑制字体相关的警告
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*glyph.*')
warnings.filterwarnings('ignore', message='.*Glyph.*')
warnings.filterwarnings('ignore', message='.*substituting.*')
# 降低matplotlib的日志级别
logging.getLogger('matplotlib').setLevel(logging.ERROR)

def extract_tensorboard_data(log_dir, use_latest=True, run_name=None):
    """从TensorBoard日志中提取数据
    
    Args:
        log_dir: TensorBoard日志根目录
        use_latest: 是否使用最新的run（如果为False，会合并所有runs）
        run_name: 指定要提取的run名称（如'PPO_1'），为None时根据use_latest决定
    """
    
    # 查找所有run目录
    run_dirs = [d for d in log_dir.iterdir() if d.is_dir() and d.name.startswith('PPO_')]
    
    if not run_dirs:
        print(f"❌ 未找到TensorBoard run目录: {log_dir}")
        return None
    
    print(f"找到 {len(run_dirs)} 个训练run:")
    for run_dir in sorted(run_dirs):
        event_files = list(run_dir.glob("events.out.tfevents.*"))
        if event_files:
            latest_time = max(f.stat().st_mtime for f in event_files)
            print(f"  - {run_dir.name}: {len(event_files)} 个事件文件 (最后更新: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(latest_time))})")
    
    # 确定要使用的run
    if run_name:
        selected_run = log_dir / run_name
        if not selected_run.exists():
            print(f"❌ 指定的run不存在: {run_name}")
            return None
        print(f"\n📊 使用指定run: {run_name}")
    elif use_latest:
        # 使用事件文件的最新时间，而不是目录的修改时间
        def get_latest_event_time(run_dir):
            event_files = list(run_dir.glob("events.out.tfevents.*"))
            if event_files:
                return max(f.stat().st_mtime for f in event_files)
            return 0
        
        selected_run = max(run_dirs, key=get_latest_event_time)
        print(f"\n📊 使用最新run: {selected_run.name}")
    else:
        print(f"\n📊 合并所有runs的数据")
        selected_run = None
    
    # 提取数据
    if selected_run:
        return _extract_single_run(selected_run)
    else:
        return _extract_all_runs(run_dirs)

def _extract_single_run(run_dir):
    """提取单个run的数据"""
    ea = event_accumulator.EventAccumulator(str(run_dir))
    ea.Reload()
    
    # 提取所有可用的标量
    tags = ea.Tags()['scalars']
    print(f"\n可用的指标标签: {len(tags)}个")
    
    metrics_data = {}
    for tag in tags:
        try:
            events = ea.Scalars(tag)
            metrics_data[tag] = {
                'steps': [e.step for e in events],
                'values': [e.value for e in events],
                'wall_times': [e.wall_time for e in events]
            }
        except Exception as e:
            print(f"⚠️  无法提取 {tag}: {e}")
    
    return metrics_data

def _extract_all_runs(run_dirs):
    """合并所有runs的数据"""
    all_metrics = {}
    
    for run_dir in sorted(run_dirs):
        ea = event_accumulator.EventAccumulator(str(run_dir))
        ea.Reload()
        
        tags = ea.Tags()['scalars']
        
        for tag in tags:
            try:
                events = ea.Scalars(tag)
                if tag not in all_metrics:
                    all_metrics[tag] = {
                        'steps': [],
                        'values': [],
                        'wall_times': []
                    }
                
                all_metrics[tag]['steps'].extend([e.step for e in events])
                all_metrics[tag]['values'].extend([e.value for e in events])
                all_metrics[tag]['wall_times'].extend([e.wall_time for e in events])
            except Exception as e:
                print(f"⚠️  无法从{run_dir.name}提取 {tag}: {e}")
    
    print(f"\n合并后可用的指标标签: {len(all_metrics)}个")
    return all_metrics

def convert_to_training_metrics(tensorboard_data):
    """将TensorBoard数据转换为训练指标格式"""
    
    if not tensorboard_data:
        return []
    
    # 获取步数列表（使用最长的序列）
    all_steps = []
    for tag, data in tensorboard_data.items():
        all_steps.extend(data['steps'])
    unique_steps = sorted(set(all_steps))
    
    print(f"\n总共 {len(unique_steps)} 个训练步数记录")
    
    # 为每个步数构建指标字典
    training_metrics = []
    
    for step in unique_steps:
        metrics = {
            'step': int(step),
            'train': {},
            'time': {},
            'rollout': {}
        }
        
        # 提取该步数的所有指标
        for tag, data in tensorboard_data.items():
            if step in data['steps']:
                idx = data['steps'].index(step)
                value = data['values'][idx]
                
                # 分类存储
                if tag.startswith('train/'):
                    key = tag.replace('train/', '')
                    metrics['train'][key] = float(value)
                elif tag.startswith('time/'):
                    key = tag.replace('time/', '')
                    metrics['time'][key] = float(value)
                elif tag.startswith('rollout/'):
                    key = tag.replace('rollout/', '')
                    metrics['rollout'][key] = float(value)
                else:
                    metrics[tag] = float(value)
        
        if metrics['train']:  # 只添加有训练数据的记录
            training_metrics.append(metrics)
    
    return training_metrics

def plot_tensorboard_metrics(metrics_data, save_path, show_all=False):
    """绘制TensorBoard指标
    
    Args:
        metrics_data: 提取的指标数据
        save_path: 保存路径（基础路径，会生成两个文件）
        show_all: 是否显示所有指标（默认只显示训练相关指标）
    """
    
    # 分类提取指标
    train_tags = [tag for tag in metrics_data.keys() if tag.startswith('train/')]
    rollout_tags = [tag for tag in metrics_data.keys() if tag.startswith('rollout/')]
    time_tags = [tag for tag in metrics_data.keys() if tag.startswith('time/')]
    custom_tags = [tag for tag in metrics_data.keys() if not any(tag.startswith(p) for p in ['train/', 'rollout/', 'time/'])]
    
    print(f"\n📊 指标分类:")
    print(f"  - 训练指标 (train/): {len(train_tags)}")
    print(f"  - Rollout指标 (rollout/): {len(rollout_tags)}")
    print(f"  - 时间指标 (time/): {len(time_tags)}")
    print(f"  - 自定义指标: {len(custom_tags)}")
    
    # 打印所有标签
    if custom_tags:
        print(f"\n🎯 自定义指标标签:")
        for tag in custom_tags:
            n_points = len(metrics_data[tag]['steps'])
            print(f"  - {tag}: {n_points} 个数据点")
    
    # 分别绘制自定义指标和训练指标
    save_dir = Path(save_path).parent
    base_name = Path(save_path).stem
    
    # 1. 绘制自定义指标
    if custom_tags:
        custom_save_path = save_dir / f"{base_name}_custom.png"
        _plot_metrics_group(metrics_data, custom_tags, custom_save_path, "自定义指标")
    
    # 2. 绘制训练指标
    if train_tags:
        train_save_path = save_dir / f"{base_name}_training.png"
        _plot_metrics_group(metrics_data, train_tags, train_save_path, "训练指标 (SB3)")

def _plot_metrics_group(metrics_data, tags, save_path, title):
    """绘制一组指标
    
    Args:
        metrics_data: 所有指标数据
        tags: 要绘制的指标标签列表
        save_path: 保存路径
        title: 图表标题
    """
    if not tags:
        return
    
    # 创建图表（自动调整布局）
    n_plots = len(tags)
    n_cols = min(2, n_plots)  # 最多2列
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    if n_plots == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    for i, tag in enumerate(tags):
        if i >= len(axes):
            break
            
        ax = axes[i]
        data = metrics_data[tag]
        
        # 绘制曲线
        ax.plot(data['steps'], data['values'], alpha=0.8, linewidth=1.5)
        ax.set_xlabel('训练步数', fontsize=10)
        ax.set_ylabel('值', fontsize=10)
        
        # 简化标签显示
        display_name = tag.replace('train/', '').replace('custom/', '')
        ax.set_title(display_name, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 对损失类指标使用对数尺度
        if 'loss' in tag.lower():
            try:
                ax.set_yscale('log')
            except:
                pass
        
        # 添加统计信息
        values = data['values']
        if values:
            stats_text = f"均值: {np.mean(values):.3f}\n范围: [{min(values):.3f}, {max(values):.3f}]"
            ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
                   fontsize=8, verticalalignment='bottom', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # 隐藏多余的子图
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ {title}图表已保存: {save_path}")
    plt.close()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='从TensorBoard日志中提取训练指标')
    parser.add_argument('--log_dir', type=str, 
                       default='agent/log_SB3/tensorboard',
                       help='TensorBoard日志目录')
    parser.add_argument('--run', type=str, default=None,
                       help='指定要提取的run名称（如PPO_1），不指定则使用最新run')
    parser.add_argument('--merge_all', action='store_true',
                       help='合并所有runs的数据（默认只使用最新run）')
    parser.add_argument('--show_all', action='store_true',
                       help='绘制所有指标（默认只显示训练和自定义指标）')
    parser.add_argument('--output_dir', type=str, default='agent/log_SB3',
                       help='输出目录')
    
    args = parser.parse_args()
    
    log_dir = Path(args.log_dir)
    
    if not log_dir.exists():
        print(f"❌ TensorBoard日志目录不存在: {log_dir}")
        return
    
    print("="*80)
    print("从TensorBoard日志中提取训练指标")
    print("="*80)
    
    # 提取数据
    tensorboard_data = extract_tensorboard_data(
        log_dir, 
        use_latest=not args.merge_all,
        run_name=args.run
    )
    
    if not tensorboard_data:
        return
    
    # 转换格式
    training_metrics = convert_to_training_metrics(tensorboard_data)
    
    # 保存为JSON
    output_dir = Path(args.output_dir)
    json_path = output_dir / "training_metrics_from_tensorboard.json"
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(training_metrics, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 训练指标已保存: {json_path}")
    print(f"   共 {len(training_metrics)} 条记录")
    
    # 绘制图表
    plot_path = output_dir / "training_metrics_from_tensorboard.png"
    plot_tensorboard_metrics(tensorboard_data, plot_path, show_all=args.show_all)
    
    # 打印详细统计信息
    print("\n" + "="*80)
    print("所有可用指标:")
    print("="*80)
    for tag in sorted(tensorboard_data.keys()):
        n_points = len(tensorboard_data[tag]['steps'])
        values = tensorboard_data[tag]['values']
        if values:
            print(f"  - {tag}:")
            print(f"      数据点: {n_points}")
            print(f"      范围: [{min(values):.4f}, {max(values):.4f}]")
            print(f"      均值: {np.mean(values):.4f}")
    
    print("="*80)
    print("提取完成！")
    print("="*80)

if __name__ == '__main__':
    main()
