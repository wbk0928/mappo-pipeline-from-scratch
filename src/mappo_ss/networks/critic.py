"""
Centralized Critic for MAPPO.

The critic receives the global state and predicts
the state value V(s).   在 CTDE（集中训练分布执行）框架下，Critic 在训练时可以获取全局信息，因此输入是全局状态，而不是局部观测。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mappo_ss.networks.initialization import init_layer
from mappo_ss.networks.mlp import MLP
# init_layer：自定义的层初始化函数，用于对网络层进行特定的权重初始化（通常是正交初始化）。
# MLP：自定义的多层感知机模块，用作特征提取器。

class Critic(nn.Module):
    """
    Centralized value network.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.feature_extractor = MLP(
            input_dim=state_dim,
            hidden_dim=hidden_dim,
        )
        """
特征提取层
创建 MLP 实例作为特征提取器。
输入维度为 state_dim，输出维度为 hidden_dim。
该 MLP 通常包含多个线性层和非线性激活函数，用于从全局状态中提取有用的特征表示。
        """

        # Standard PPO/MAPPO initialization for value head
        self.value_head = init_layer(
            nn.Linear(hidden_dim, 1),
            gain=1.0,
        )
        """
价值输出层
创建一个线性层 nn.Linear(hidden_dim, 1)，将特征映射为一个标量价值。
使用 init_layer 初始化该层，增益 gain=1.0。
注释说明这是 PPO/MAPPO 中价值头（value head）的标准初始化。
增益为 1.0 通常对应正交初始化的默认增益，保持输出方差与输入相近，有利于稳定价值估计。
        """

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
       定义前向传播，输入 state（形状 (..., state_dim)），输出预测的价值 V(s) 。
输出的形状与输入的前导维度相同，去掉最后一维，例如输入 (batch_size, state_dim) 输出 (batch_size,)。
        """

        features = self.feature_extractor(state)
# 将全局状态 state 送入特征提取器，获得特征表示 features，形状为 (..., hidden_dim)。
        value = self.value_head(features)
# 将特征通过线性层 value_head，得到价值估计 value，形状为 (..., 1)。
        return value.squeeze(-1)
        """
使用 squeeze(-1) 去掉最后一维，使输出形状变为 (...,)。
这样保证输出的价值是一个标量（或批量数据下的向量），方便后续计算损失时与目标值直接比较。
        """
