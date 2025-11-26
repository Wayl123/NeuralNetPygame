from gymnasium.envs.registration import register

register (
  id="gymnasium_env/MarbleGame-v0",
  entry_point="gymnasium_env.envs:MarbleGameEnv",
)