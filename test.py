import gymnasium
import gymnasium_env
env = gymnasium.make('gymnasium_env/MarbleGame-v0', render_mode="human")
env.reset()
env.render()