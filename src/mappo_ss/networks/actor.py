"""
Shared Actor Network for MAPPO.

The actor receives a local observation and outputs a categorical
distribution over discrete actions.“共享”指所有智能体使用同一个 Actor 网络（参数共享），这是 MARL 中常用的做法，可以加快训练并处理智能体数量变化。
"""

from __future__ import annotations
# 启用延迟注解求值，使类型注解更灵活，兼容旧版本 Python

import torch
import torch.nn as nn
from torch.distributions import Categorical

from mappo_ss.networks.initialization import init_layer
from mappo_ss.networks.mlp import MLP
"""
torch：PyTorch 主库。
torch.nn as nn：神经网络模块，提供各种层和容器。
Categorical：从 torch.distributions 导入的分类分布类，用于构建和采样离散动作的概率分布。
init_layer：自定义的层初始化函数，用于对网络层进行特定的权重初始化。
MLP：自定义的多层感知机模块，用于提取特征。下面会用到。
"""


class Actor(nn.Module):
    """
    Shared policy network.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        action_dim: int,
    ):
        super().__init__()
"""
构造函数接收三个参数：
obs_dim：观测向量的维度。
hidden_dim：隐藏层维度（特征提取器的输出维度）。
action_dim：离散动作空间的维度（可选动作的数量）。
调用 super().__init__() 初始化基类 nn.Module。
"""
        self.feature_extractor = MLP(
            input_dim=obs_dim,
            hidden_dim=hidden_dim,
        )
"""
特征提取器
创建一个 MLP 实例作为特征提取器。
input_dim 设为观测维度 obs_dim。
hidden_dim 设为隐藏层维度 hidden_dim。
这个 MLP 负责将原始观测映射到一个特征向量。通常它会包含若干线性层和激活函数（例如 ReLU、Tanh）。具体实现未知，但可以推测输出维度也是 hidden_dim（因为随后接的线性层输入维度是 hidden_dim）。
"""
        # Small gain is standard for policy output layers.
        self.policy_head = init_layer(
            nn.Linear(hidden_dim, action_dim),
            gain=0.01,
        )
"""
策略输出层
创建一个线性层 nn.Linear(hidden_dim, action_dim)，将特征向量映射为动作 logits。
使用 init_layer 对该线性层进行初始化，增益 gain=0.01。
小增益（0.01）是策略输出层的常见做法（如 PPO、A3C 等），它使得初始输出接近均匀分布，避免一开始策略过于确定，有利于探索。
如果增益为 1，则初始化权重为标准正态分布，可能导致初始策略偏向某些动作，不利于早期探索。
这个 policy_head 输出的 logits 会用于构建 Categorical 分布。
"""
    def forward(self, obs: torch.Tensor) -> Categorical:
        """
        forward 定义了数据流动过程，输入观测张量 obs，输出一个 Categorical 分布对象。
输入 obs 的形状可以是任意前导维度，最后一维必须是 obs_dim，例如 (batch_size, obs_dim) 或 (1, obs_dim)。
        """

        features = self.feature_extractor(obs)
# 将观测 obs 送入特征提取器 self.feature_extractor，得到特征向量 features。通常形状为 (..., hidden_dim)。
        logits = self.policy_head(features)
#将特征 features 通过策略输出线性层 self.policy_head，得到动作 logits。形状为 (..., action_dim)。
        return Categorical(logits=logits)
        """
使用 logits 构建一个 torch.distributions.Categorical 分布。
该分布可以用来：
采样动作：distribution.sample()
计算动作的对数概率：distribution.log_prob(action)
计算熵：distribution.entropy()
在 MAPPO 训练中，Actor 的前向通常返回这个分布（而不是直接输出动作），以便后续进行重要性采样和策略梯度计算。
        """
