import os
import sys
import gym
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt

# Make environment


class PolicyNetwork(nn.Module):
    
    def __init__(self, n_actions, n_states, hidden_size):
        super(PolicyNetwork, self).__init__()
        self.n_states = n_states
        self.hidden_size = hidden_size
        self.n_actions = n_actions
        self.fc1 = nn.Linear(self.n_states, self.hidden_size)
        self.fc1_activation = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size, self.n_actions)
        self.pi = nn.Softmax(dim = 0)
    
    def forward(self, input):

        X = self.fc1(input)
        out = self.fc2(self.fc1_activation(X))
        output = self.pi(out)
        return output

class Agent:

    def __init__(self, n_actions, n_states, hidden_size, gamma = 0.99, learning_rate = 0.001):

        self.gamma = gamma
        self.n_actions = n_actions
        self.policy = PolicyNetwork(n_actions,n_states,hidden_size)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr = learning_rate)
        self.rewards_history = []
        self.action_history = []
        self.state_history = []

    def choose_action(self, observation):
        
        state = torch.tensor(observation, dtype=torch.float32)
        probs = self.policy(state)
        action_probs = Categorical(probs = probs)
        action = action_probs.sample()

        return action.detach().numpy().item()

    def store_transition(self, observation, action, reward):
        self.state_history.append(observation)
        self.action_history.append(action)
        self.rewards_history.append(reward)

    def learn(self):
        
        actions = torch.tensor(self.action_history, dtype=torch.float32)
        rewards = torch.tensor(self.rewards_history, dtype=torch.float32)
        mc_returns = np.zeros_like(rewards)
        for t in range(len(rewards)):
            G = 0
            for k in range(t, len(rewards)):
                G += (self.gamma**(k - t))*rewards[k]
            mc_returns[t] = G

        loss= 0
        for idx, (g,state) in enumerate(zip(mc_returns, self.state_history)):
            state = torch.tensor(state, dtype = torch.float32)
            probs = self.policy(state)
            action_probs = Categorical(probs = probs)
            log_probs = action_probs.log_prob(actions[idx])
            loss += -g*torch.squeeze(log_probs) 

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.state_history = []
        self.action_history = []
        self.rewards_history = []
    

if __name__ == '__main__':

    env = gym.make('CartPole-v1')
    agent = Agent(n_actions = env.action_space.n, n_states = 4, hidden_size = 100, gamma = 0.99, learning_rate = 0.003)
    score_history = []
    n_episodes = 2000

    for i in range(n_episodes):
        done = False
        score = 0
        observation = env.reset()

        while not done:
            action = agent.choose_action(observation)
            next_observation, reward, done, info = env.step(action)
            agent.store_transition(observation, action, reward)
            observation = next_observation
            score += reward
            # env.render()

        score_history.append(score)
        agent.learn()
        

        avg_score = np.mean(score_history[-100:])
        print('episode ', i, 'score  %.1f' % score, 'avg score %.1f' % avg_score)
    
    plt.figure(2)
    plt.plot(np.arange(n_episodes),score_history)
    plt.show()

    

    










