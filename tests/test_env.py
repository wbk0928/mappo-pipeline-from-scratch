from mappo_ss.config.config import load_config
from mappo_ss.envs.pettingzoo_env import PettingZooEnv

cfg = load_config("configs/simple_spread.yaml")

env = PettingZooEnv(cfg)

obs, infos = env.reset(seed=42)

print("=" * 60)
print("Agents")
print(env.agent_ids)

print("=" * 60)
print("Number of agents")
print(env.num_agents)

print("=" * 60)
print("Observation dimension")
print(env.obs_dim)

print("=" * 60)
print("Action dimension")
print(env.action_dim)

print("=" * 60)
print("Global state dimension")
print(env.state_dim)

state = env.get_global_state(obs)

print("=" * 60)
print("Global state shape")
print(state.shape)

env.close()