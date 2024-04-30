import gym
import imitation
import torch
from torch import nn, optim
from imitation.algorithms import bc
from imitation.data import rollout
from imitation.util import networks

# Set up MiniGrid environment
env_name = "MiniGrid-BlockedUnlockPickup-v0"
env = gym.make(env_name)

# Expert demonstrations
expert_data = rollout.generate_expert_rollouts(
    env_name=env_name,
    n_trajectories=100,  # adjust this number as needed
    save_path=None  # or specify a file path to save demonstrations
)

# Define behavior cloning model
obs_space = env.observation_space
act_space = env.action_space
acmodel = networks.build_mlp(
    input_size=obs_space.shape[0],
    output_size=act_space.n,
    hidden_sizes=[64, 64],  # adjust hidden layer sizes as needed
    output_activation=nn.Identity  # Identity activation for discrete actions
)

# Define behavior cloning algorithm
bc_trainer = bc.BC(
    acmodel=acmodel,
    optimizer=optim.Adam(acmodel.parameters(), lr=3e-4),  # adjust learning rate as needed
    expert_data=expert_data
)

# Train behavior cloning model
bc_trainer.train(n_iters=1000)  # adjust number of training iterations as needed

# Save trained model
torch.save(acmodel.state_dict(), 'bc_model.pt')

# Optionally, test the trained model
def test_model(model, env, n_episodes=10):
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = model(torch.tensor(obs, dtype=torch.float32)).argmax().item()
            obs, reward, done, _ = env.step(action)
            env.render()
            if done:
                break
    env.close()

# Load the trained model
acmodel.load_state_dict(torch.load('bc_model.pt'))

# Test the model
test_model(acmodel, env)
