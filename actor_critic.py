import os
import sys
import gym
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt

class ActorNetwork(nn.Module):
    
    def __init__(self, n_actions, n_states, hidden_size):
        super(ActorNetwork, self).__init__()
        self.n_states = n_states
        self.hidden_size = hidden_size
        self.n_actions = n_actions
        self.fc1 = nn.Linear(self.n_states, self.hidden_size[0])
        self.fc1_activation = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size[0], self.hidden_size[1])
        self.fc2_activation = nn.ReLU()
        self.fc3 = nn.Linear(self.hidden_size[1], self.n_actions)
        self.pi = nn.Softmax(dim = 0)
    
    def forward(self, input):

        X = self.fc1(input)
        out = self.fc2(self.fc1_activation(X))
        out = self.fc3(self.fc2_activation(out))
        output = self.pi(out)
        return output

class CriticNetwork(nn.Module):

    def __init__(self, n_states, hidden_size, n_outputs = 1):
        super(CriticNetwork, self).__init__()
        self.n_states = n_states
        self.hidden_size = hidden_size
        self.n_outputs = n_outputs
        self.fc1 = nn.Linear(self.n_states, self.hidden_size[0])
        self.fc1_activation = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size[0], self.hidden_size[1])
        self.fc2_activation = nn.ReLU()
        self.fc3 = nn.Linear(self.hidden_size[1], self.n_outputs)
    
    def forward(self, input):

        X = self.fc1(input)
        output = self.fc2(self.fc1_activation(X))
        output = self.fc3(self.fc2_activation(output))
        return output

class Agent:

    def __init__(self, n_actions, n_states, hidden_size, learning_rate, gamma = 0.99):

        self.gamma = gamma
        self.n_actions = n_actions
        self.actor= ActorNetwork(n_actions,n_states, hidden_size['actor'])
        self.critic = CriticNetwork(n_states, hidden_size['critic'])
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = learning_rate['actor'])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr = learning_rate['critic'])
        self.history = []

    def choose_action(self, observation):
        
        state = torch.tensor(observation, dtype=torch.float32)
        probs = self.actor(state)
        action_probs = Categorical(probs = probs)
        action = action_probs.sample()

        return action.detach().numpy().item()

    def store_transition(self, observation, action, reward, next_observation):
        self.history.append((observation, action, reward, next_observation))

    def learn(self, done):

        current_state, current_action,current_reward, next_state = self.history[-1]
        current_state = torch.tensor(current_state, dtype = torch.float32)
        next_state = torch.tensor(next_state, dtype = torch.float32)
        current_reward = torch.tensor(current_reward, dtype = torch.float32)
        current_action = torch.tensor(current_action, dtype = torch.float32)
        
        actor_output = self.actor(current_state)
        action_probs = Categorical(probs = actor_output)
        
        current_state_value = self.critic(current_state)
        next_state_value = self.critic(next_state)
        td_error = (current_reward + self.gamma*next_state_value*(1- int(done))) - current_state_value
        
        log_probs = action_probs.log_prob(current_action)

        actor_loss = -td_error * log_probs
        critic_loss = td_error ** 2

        loss = actor_loss+critic_loss
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        return actor_loss.item(), critic_loss.item()

if __name__ == '__main__':

    env = gym.make('CartPole-v1')
    hidden_size = dict()
    learning_rate = dict()
    
    hidden_size['actor'] = [128, 64]
    hidden_size['critic'] = [128, 64]
    
    learning_rate['actor'] = 0.003
    learning_rate['critic'] = 0.003

    agent = Agent(n_actions = env.action_space.n, n_states = 4, hidden_size = hidden_size,  learning_rate = learning_rate, gamma = 0.99)
    score_history = []
    actor_losses = []
    critic_losses = []
    n_episodes = 1500

    for i in range(n_episodes):
        
        done = False
        score = 0
        observation = env.reset()
        mean_actor_loss_in_episode = 0
        mean_critic_loss_in_episode = 0
        step_count = 1
        while not done:
            action = agent.choose_action(observation)
            next_observation, reward, done, info = env.step(action)
            agent.store_transition(observation, action, reward, next_observation)
            observation = next_observation
            score += reward
            actor_loss, critic_loss = agent.learn(done)
            mean_actor_loss_in_episode += actor_loss
            mean_critic_loss_in_episode += critic_loss
            step_count += 1

        mean_actor_loss_in_episode = mean_actor_loss_in_episode / step_count
        mean_critic_loss_in_episode = mean_critic_loss_in_episode / step_count
        
            # env.render()

        score_history.append(score)
        actor_losses.append(mean_actor_loss_in_episode)
        critic_losses.append(mean_critic_loss_in_episode)
        avg_score = np.mean(score_history[-50:])
        print('episode ', i, 'score  %.1f' % score, 'avg score %.1f' % avg_score)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex='all')
    ax1.plot(np.arange(n_episodes), score_history)
    ax2.plot(np.arange(n_episodes), actor_losses)
    ax3.plot(np.arange(n_episodes), critic_losses)
    plt.show()
