"""
Data structure representing one environment timestep.
"""

from __future__ import annotations

from dataclasses import dataclass
# dataclass 可以自动生成 __init__、__repr__、__eq__ 等方法，减少样板代码，适合用来定义简单的数据容器。
import numpy as np


@dataclass(slots=True)
class Transition:
    """
   @dataclass 装饰器将 Transition 类标记为一个数据类。
slots=True 是 Python 3.10+ 引入的特性，用于为数据类创建 __slots__，从而 减少内存占用 并提升属性访问速度。
通常数据类的每个实例都有一个 __dict__ 字典来存储属性，占用较多内存。
使用 slots=True 后，属性存储在固定槽位中，不再有 __dict__，对于需要创建大量过渡样本的场景（例如强化学习中的 replay buffer）可以显著节省内存。
注意：使用 slots=True 时，不能动态添加新属性，但这对只存储固定字段的 Transition 来说没有问题。
    """

    observations: dict[str, np.ndarray]
"""
类型注解：dict[str, np.ndarray]，表示一个字典，键为智能体 ID（字符串），值为该智能体在当前时刻的观测（NumPy 数组）。
对应 PettingZoo step 或 reset 返回的 observations 字典。
这是每个智能体的 局部观测，将作为 Actor 网络的输入。
"""
    global_state: np.ndarray
"""
类型注解：np.ndarray，表示当前时刻的 全局状态。
它是通过拼接所有智能体的观测（或其他方式）构建的集中式状态。
将作为 Critic 网络的输入，用于估计状态价值。
"""
    actions: dict[str, int]
"""
类型注解：dict[str, int]，表示每个智能体在当前时刻选择的动作。
键为智能体 ID，值为离散动作的整数索引（因为当前环境用离散动作空间）。
这些动作会根据 Actor 输出的 Categorical 分布采样得到。
"""
    log_probs: dict[str, float]
"""
类型注解：dict[str, float]，表示每个智能体选择的动作在旧策略下的对数概率 
在 PPO 中，计算重要性采样比率时需要对比 旧策略 和 新策略 的对数概率，因此必须记录采样时的对数概率。
"""
    rewards: dict[str, float]
"""
类型注解：dict[str, float]，表示每个智能体在当前时刻获得的即时奖励。
注意：在多智能体环境中，不同智能体可能获得不同奖励，也可能共享相同奖励（取决于环境设定）。
"""
    values: dict[str, float]
"""
类型注解：dict[str, float]，表示当前时刻 Critic 对全局状态的价值估计 V(st) 。
在 GAE 和值函数更新中需要用到这些估计值。
这里按智能体 ID 存储，通常所有智能体共享同一个集中式 Critic，因此该值可能对所有智能体相同（如果采用全局状态价值，则每个智能体的 value 相同），但通过字典分别存储，便于后续按智能体索引。
"""
    next_observations: dict[str, np.ndarray]
"""
类型注解：dict[str, np.ndarray]，表示执行动作后、获得下一时刻的 局部观测。
用于计算 TD 误差中的 V(s_{t+1})（Critic 需要下一状态的价值）以及作为后续时间步的输入。
"""
    next_global_state: np.ndarray
"""
类型注解：np.ndarray，表示下一时刻的全局状态。
用于 Critic 更新时计算下一状态的价值 V(s t+1) ，或在 GAE 中计算 bootstrapped 价值。
"""
    terminations: dict[str, bool]
"""
类型注解：dict[str, bool]，表示每个智能体在当前时间步是否因达到终止条件而结束（例如完成任务、掉坑等）。
在 PettingZoo 中，terminations 表示环境自然结束（如目标达成），一旦为 True，则该智能体后序时间步不再有有效数据。
"""
    truncations: dict[str, bool]
"""
类型注解：dict[str, bool]，表示每个智能体在当前时间步是否因达到最大步数等外部条件而被 截断（即 episode 被提前结束）。
PettingZoo 遵循 Gymnasium 的 terminations/truncations 约定：
termination：因环境自身条件结束（成功或失败）。
truncation：因时间限制等外部条件结束（例如达到 max_cycles）。
在计算价值目标时，需要区分两者：终止时将未来价值置零；截断时仍使用下一状态价值进行 bootstrap。
"""
