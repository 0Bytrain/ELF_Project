from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class ReplayMemory:





    def __init__(self, entry_size: int, action_size: int, memory_size: int = 50000, batch_size: int = 64):

        self.entry_size = entry_size

        self.action_size = action_size

        self.memory_size = memory_size

        self.batch_size = batch_size


        self.actions = np.empty(self.memory_size, dtype=np.int64)
        self.rewards = np.empty(self.memory_size, dtype=np.float32)
        self.prestate = np.empty((self.memory_size, self.entry_size), dtype=np.float32)
        self.poststate = np.empty((self.memory_size, self.entry_size), dtype=np.float32)
        self.next_action_mask = np.empty((self.memory_size, self.action_size), dtype=np.float32)
        self.dones = np.empty(self.memory_size, dtype=np.float32)


        self.count = 0

        self.current = 0

    def add(self, prestate, poststate, reward, action, done, next_action_mask) -> None:




        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.prestate[self.current] = prestate
        self.poststate[self.current] = poststate
        self.next_action_mask[self.current] = np.asarray(next_action_mask, dtype=np.float32)
        self.dones[self.current] = float(done)


        self.count = min(self.memory_size, self.count + 1)
        self.current = (self.current + 1) % self.memory_size

    def sample(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:




        if self.count == 0:
            return None

        if self.count < self.batch_size:
            indexes = np.arange(self.count)
        else:
            indexes = np.array(random.sample(range(self.count), self.batch_size))

        return (
            self.prestate[indexes],
            self.poststate[indexes],
            self.actions[indexes],
            self.rewards[indexes],
            self.dones[indexes],
            self.next_action_mask[indexes],
        )


class DuelingDQNNet(nn.Module):










    def __init__(self, n_input: int, n_output: int, hidden_1: int = 256, hidden_2: int = 256):
        super().__init__()


        self.feature = nn.Sequential(
            nn.Linear(n_input, hidden_1),
            nn.LayerNorm(hidden_1),
            nn.ReLU(),
            nn.Linear(hidden_1, hidden_2),
            nn.LayerNorm(hidden_2),
            nn.ReLU(),
        )


        self.value = nn.Sequential(nn.Linear(hidden_2, 128), nn.ReLU(), nn.Linear(128, 1))

        self.advantage = nn.Sequential(nn.Linear(hidden_2, 128), nn.ReLU(), nn.Linear(128, n_output))

    def forward(self, x: torch.Tensor) -> torch.Tensor:




        features = self.feature(x)
        value = self.value(features)
        advantage = self.advantage(features)

        return value + advantage - advantage.mean(dim=1, keepdim=True)


@dataclass
class AgentConfig:




    state_dim: int                 
    action_dim: int                
    gamma: float = 0.97                                         
    lr: float = 1.5e-4            
    target_update_step: int = 1             
    tau: float = 0.02                              
    memory_size: int = 50000        
    batch_size: int = 64             
    grad_clip: float = 5.0           
    device: Optional[str] = None                  


class DQNAgent:




    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg
        self.state_dim = cfg.state_dim
        self.action_dim = cfg.action_dim
        self.gamma = cfg.gamma
        self.target_update_step = cfg.target_update_step
        self.tau = cfg.tau
        self.grad_clip = cfg.grad_clip
        self.update_counter = 0


        self.device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")


        self.memory = ReplayMemory(
            entry_size=self.state_dim,
            action_size=self.action_dim,
            memory_size=cfg.memory_size,
            batch_size=cfg.batch_size,
        )


        self.eval_net = DuelingDQNNet(self.state_dim, self.action_dim).to(self.device)
        self.target_net = DuelingDQNNet(self.state_dim, self.action_dim).to(self.device)


        self.target_net.load_state_dict(self.eval_net.state_dict())


        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=cfg.lr, weight_decay=1e-5)


        self.loss_fn = nn.SmoothL1Loss()

    def _finish_update(self) -> None:




        self.update_counter += 1
        if self.update_counter % self.target_update_step == 0:
            self.update_target_q_network()

    def predict_q_values(self, state: np.ndarray) -> np.ndarray:




        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.eval_net(state_tensor).detach().cpu().numpy()[0]
        return q_values

    def _masked_argmax(self, q_values: np.ndarray, action_mask: Optional[np.ndarray] = None) -> int:




        if action_mask is None:
            return int(np.argmax(q_values))

        mask = np.asarray(action_mask, dtype=bool)
        if mask.shape[0] != q_values.shape[0] or not np.any(mask):
            return int(np.argmax(q_values))


        masked_q = np.where(mask, q_values, -1e12)
        return int(np.argmax(masked_q))

    def greedy_action(self, state: np.ndarray, action_mask: Optional[np.ndarray] = None) -> int:




        return self._masked_argmax(self.predict_q_values(state), action_mask)

    def select_action(self, state: np.ndarray, epsilon: float, action_mask: Optional[np.ndarray] = None) -> int:





        valid_actions = None
        if action_mask is not None:
            mask = np.asarray(action_mask, dtype=bool)
            if np.any(mask):
                valid_actions = np.flatnonzero(mask)


        if np.random.rand() < epsilon:
            if valid_actions is not None:
                return int(np.random.choice(valid_actions))
            return int(np.random.randint(self.action_dim))


        return self.greedy_action(state, action_mask)

    @staticmethod
    def _mask_q_tensor(q_values: torch.Tensor, action_mask: Optional[torch.Tensor]) -> torch.Tensor:




        if action_mask is None:
            return q_values

        mask = action_mask.to(dtype=torch.bool)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        return q_values.masked_fill(~mask, -1e9)

    def learn(self) -> Optional[float]:









        batch = self.memory.sample()
        if batch is None:
            return None


        batch_s_t, batch_s_t_plus_1, batch_action, batch_reward, batch_done, batch_next_action_mask = batch


        batch_s_t = torch.tensor(batch_s_t, dtype=torch.float32, device=self.device)
        batch_s_t_plus_1 = torch.tensor(batch_s_t_plus_1, dtype=torch.float32, device=self.device)
        batch_action = torch.tensor(batch_action, dtype=torch.long, device=self.device)
        batch_reward = torch.tensor(batch_reward, dtype=torch.float32, device=self.device)
        batch_done = torch.tensor(batch_done, dtype=torch.float32, device=self.device)
        batch_next_action_mask = torch.tensor(batch_next_action_mask, dtype=torch.float32, device=self.device)


        q_eval = self.eval_net(batch_s_t).gather(1, batch_action.unsqueeze(1)).squeeze(1)

        with torch.no_grad():

            q_eval_next = self._mask_q_tensor(self.eval_net(batch_s_t_plus_1), batch_next_action_mask)
            next_actions = q_eval_next.argmax(dim=1, keepdim=True)


            q_target_next = self._mask_q_tensor(self.target_net(batch_s_t_plus_1), batch_next_action_mask)
            q_next = q_target_next.gather(1, next_actions).squeeze(1)


            q_target = batch_reward + self.gamma * q_next * (1.0 - batch_done)


        loss = self.loss_fn(q_eval, q_target)


        self.optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(self.eval_net.parameters(), self.grad_clip)
        self.optimizer.step()


        self._finish_update()
        return float(loss.item())

    def update_target_q_network(self) -> None:





        if self.tau >= 1.0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            return

        with torch.no_grad():
            for target_param, eval_param in zip(self.target_net.parameters(), self.eval_net.parameters()):
                target_param.data.mul_(1.0 - self.tau)
                target_param.data.add_(self.tau * eval_param.data)

    def save_model(self, path: str) -> None:




        torch.save(
            {
                "eval_net": self.eval_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "config": self.cfg.__dict__,
            },
            path,
        )

    def load_model(self, path: str) -> None:



        checkpoint = torch.load(path, map_location=self.device)
        self.eval_net.load_state_dict(checkpoint["eval_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
