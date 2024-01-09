import numpy as np
from collections import deque

class ReplayBuffer():
    def __init__(self, max_size):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_memory = deque(maxlen=self.mem_size)
        self.new_state_memory = deque(maxlen=self.mem_size)
        
        self.action_memory = deque(maxlen=self.mem_size)
        self.reward_memory = deque(maxlen=self.mem_size)
        self.terminal_memory = deque(maxlen=self.mem_size)

    def store_transition(self, state, action, reward, state_, done):
        self.state_memory.append(state)
        self.new_state_memory.append(state_)
        self.action_memory.append(action)
        self.reward_memory.append(reward)
        self.terminal_memory.append(done)

    def sample_buffer(self, batch_size):
        max_mem = len(self.state_memory)

        batch = np.random.choice(max_mem, batch_size)

        states = np.array(self.state_memory)[batch]
        actions = np.array(self.action_memory)[batch]
        rewards = np.array(self.reward_memory)[batch]
        states_ = np.array(self.new_state_memory)[batch]
        dones = np.array(self.terminal_memory)[batch]


        return states, actions, rewards, states_, dones

