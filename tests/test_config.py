from pathlib import Path

from mappo_ss.config.config import load_config

cfg = load_config(Path("configs/simple_spread.yaml"))

print("=" * 60)
print(cfg["environment"])

print("=" * 60)
print(cfg["network"])

print("=" * 60)
print(cfg["algorithm"])

print("=" * 60)
print(cfg["training"])