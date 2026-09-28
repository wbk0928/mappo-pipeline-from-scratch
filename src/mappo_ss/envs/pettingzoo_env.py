"""
Generic PettingZoo environment wrapper.

Currently supports:
    - simple_spread_v3 (MPE2)

Designed for CTDE (Centralized Training, Decentralized Execution).设计目标是为了 CTDE（集中训练、分散执行）模式
"""

from __future__ import annotations 
# 启用延迟注解求值

from typing import Any
# 导入Any类型，用于标注不确定类型的数据

import numpy as np
from gymnasium.spaces import Discrete
# 从Gymnasium库导入Discrete空间类，用于检查动作空间是否为离散类型
from mpe2 import simple_spread_v3

from mappo_ss.config.config import Config
# 导入项目内的Config 类，它是一个配置对象，存放环境参数、训练超参数等。


class PettingZooEnv:
    """Generic wrapper for PettingZoo parallel environments."""

    def __init__(self, cfg: Config):

        env_cfg = cfg["environment"]

        env_name = env_cfg["env_name"]

        if env_name == "simple_spread_v3":
            self.env = simple_spread_v3.parallel_env(
                N=env_cfg["num_agents"],
                max_cycles=env_cfg["max_cycles"],
                continuous_actions=env_cfg["continuous_actions"],
                render_mode=env_cfg["render_mode"],
            )
        """
        如果环境名是 "simple_spread_v3"，则调用 simple_spread_v3.parallel_env 创建并行环境实例。
N：智能体数量。
max_cycles：每个 episode 的最大步数。
continuous_actions：是否使用连续动作（这里应为 False，因为代码只支持离散动作）。
render_mode：渲染模式，例如 "human" 或 None。
        """
        else:
            raise ValueError(f"Unsupported environment: {env_name}")
            # 否则抛出 ValueError 异常，表示不支持该环境。

        self.agent_ids: list[str] = []
        self.num_agents: int = 0

        self.obs_dim: int = 0
        self.action_dim: int = 0
        self.state_dim: int = 0

    """
    初始化一些实例属性，稍后在 reset 阶段会被填充：
agent_ids：智能体的 ID 列表（字符串）。
num_agents：智能体数量。
obs_dim：单个智能体的观测维度。
action_dim：动作空间的维度（对于离散动作是动作个数）。
state_dim：集中式全局状态的维度（这里定义为所有智能体观测的简单拼接）。
    """
    def reset(
        self,
        seed: int | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """
        重置环境到初始状态。
参数 seed：随机种子（可选），用于可复现性。
返回一个元组 (observations, infos)，其中：
observations 是一个字典，键是智能体 ID，值是该智能体的观测 numpy 数组。
infos 是一个辅助信息字典。
        """

        observations, infos = self.env.reset(seed=seed)
        # 调用底层 PettingZoo 环境的 reset 方法，获得初始观测和额外信息
        
        self.agent_ids = list(self.env.agents)
        self.num_agents = len(self.agent_ids)
        # 获取当前环境下所有智能体的 ID，并记录数量

        first_agent = self.agent_ids[0]

        self.obs_dim = observations[first_agent].shape[0]
        # 从第一个智能体的观测数组的形状中提取观测维度

        action_space = self.env.action_space(first_agent)

        if not isinstance(action_space, Discrete):
            raise RuntimeError(
                "Current MAPPO implementation supports only discrete actions."
            )
            # 检查动作空间是否为 Discrete 类型。如果不满足，则抛出运行时错误，因为当前 MAPPO 算法只支持离散动作

        self.action_dim = action_space.n
        # 对于离散动作空间，action_space.n 表示可用的动作数量，将其赋给 self.action_dim

        self.state_dim = self.obs_dim * self.num_agents
        # 设置全局状态维度：假设全局状态是所有智能体观测的简单拼接，因此维度等于 obs_dim * num_agents

        return observations, infos

    def step(self, actions):

        return self.env.step(actions)

    def get_global_state(
        self,
        observations: dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        该方法用于构建集中式全局状态（centralized state）。
输入 observations：包含所有智能体当前观测的字典。
输出：一个一维 numpy 数组，它是所有智能体观测的简单拼接。
        Build centralized state by concatenating
        every agent observation.方法在 CTDE 训练中用于给 Critic 网络提供全局信息
        """

        return np.concatenate(
            [observations[agent] for agent in self.agent_ids],
            axis=0,
        )
        """
        使用列表推导式按 self.agent_ids 的顺序取出每个智能体的观测，然后调用 np.concatenate(..., axis=0) 将它们拼接为一个长向量。
例如，若有 3 个智能体，每个观测维度为 10，则返回的形状为 (30,)。
这种全局状态构造假设观测是一维的，并且拼接顺序固定（与 agent_ids 一致），以保证 Critic 输入的一致性。
        """

    def close(self):

        self.env.close()
