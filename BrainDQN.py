# -----------------------------
# Author: Flood Sung
# Date: 2016.3.21
# =============================
# Modified by xmfbit, 2017.4
# -----------------------------

import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

ACTIONS = 2    # total available action number for the game: UP and DO NOTHING

class BrainDQN(nn.Module):
	empty_frame = np.zeros((128, 72), dtype=np.float32)
	empty_state = np.stack((empty_frame, empty_frame, empty_frame, empty_frame), axis=0)

	def __init__(self, epsilon, mem_size, cuda):
		"""Initialization

		   epsilon: initial epsilon for exploration
                   mem_size: memory size for experience replay
                   cuda: use cuda or not
		"""
		super(BrainDQN, self).__init__()
		self.train = None
		# init replay memory
		self.replay_memory = deque()
		# init some parameters
		self.time_step = 0
		self.epsilon = epsilon
		self.actions = ACTIONS
		self.mem_size = mem_size
                self.use_cuda = cuda
		# init Q network
		self.createQNetwork()

	def createQNetwork(self):
		""" Create dqn, invoked by `__init__`

		    model structure: conv->conv->fc->fc
			change it to your new design
		"""
		self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=2)
		self.relu1 = nn.ReLU(inplace=True)
		self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
		self.relu2 = nn.ReLU(inplace=True)
		self.map_size = (64, 16, 9)
		self.fc1 = nn.Linear(self.map_size[0]*self.map_size[1]*self.map_size[2], 256)
		self.relu3 = nn.ReLU(inplace=True)
		self.fc2 = nn.Linear(256, self.actions)

	def get_q_value(self, o):
		"""Get Q value estimation w.r.t. current observation `o`

		   o -- current observation
		"""
		# get Q estimation
		out = self.conv1(o)
		out = self.relu1(out)
		out = self.conv2(out)
		out = self.relu2(out)
		out = out.view(out.size()[0], -1)
		out = self.fc1(out)
		out = self.relu3(out)
		out = self.fc2(out)
		return out

	def forward(self, o):
		"""Forward procedure to get MSE loss

		   o -- current observation
		"""
		# get Q(s,a;\theta)
		q = self.get_q_value(o)
		return q

	def set_train(self):
		"""Set phase TRAIN
		"""
		self.train = True

	def set_eval(self):
		"""Set phase EVALUATION
		"""
		self.train = False

	def set_initial_state(self, state=None):
		"""Set initial state

		   state: initial state. if None, use `BrainDQN.empty_state`
		"""
		if state is None:
			self.current_state = BrainDQN.empty_state
		else:
			self.current_state = state


	def store_transition(self, o_next, action, reward, terminal):
		"""Store transition (\fan_t, a_t, r_t, \fan_{t+1})

		   o_next: next observation, \fan_{t+1}
		   action: action, a_t
		   reward: reward, r_t
		   terminal: terminal(\fan_{t+1})
		"""
		next_state = np.append(self.current_state[1:,:,:], o_next.reshape((1,)+o_next.shape), axis=0)
		self.replay_memory.append((self.current_state, action, reward, next_state, terminal))
		if len(self.replay_memory) > self.mem_size:
			self.replay_memory.popleft()
		if not terminal:
			self.current_state = next_state
		else:
			self.set_initial_state()

	def get_action_randomly(self):
		"""Get action randomly
		"""
		action = np.zeros(self.actions, dtype=np.float32)
		#action_index = random.randrange(self.actions)
		action_index = 0 if random.random() < 0.8 else 1
		action[action_index] = 1
		return action

	def get_optim_action(self):
		"""Get optimal action based on current state
		"""
		state = self.current_state
		state_var = Variable(torch.from_numpy(state), volatile=True).unsqueeze(0)
                if self.use_cuda:
                    state_var = state_var.cuda()
		q_value = self.forward(state_var)
		_, action_index = torch.max(q_value, dim=1)
		action_index = action_index.data[0][0]
		action = np.zeros(self.actions, dtype=np.float32)
		action[action_index] = 1
		return action

	def get_action(self):
		"""Get action w.r.t current state
		"""
		if self.train and random.random() <= self.epsilon:
			return self.get_action_randomly()
		return self.get_optim_action()

	def increase_time_step(self, time_step=1):
		"""increase time step"""
		self.time_step += time_step
