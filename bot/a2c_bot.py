import torch as th
from torch import nn
from torch.optim import Adam, RMSprop
from torch.autograd import Variable
import numpy as np


def identity(x):
    return x


def entropy(p):
    return -th.sum(p * th.log(p), 1)


def index_to_one_hot(index, dim):
    if isinstance(index, np.int) or isinstance(index, np.int64):
        one_hot = np.zeros(dim)
        one_hot[index] = 1.
    else:
        one_hot = np.zeros((len(index), dim))
        one_hot[np.arange(len(index)), index] = 1.
    return one_hot


def to_tensor_var(x, dtype="float"):
    float_tensor = th.FloatTensor
    long_tensor = th.LongTensor
    byte_tensor = th.ByteTensor
    if dtype == "float":
        x = np.array(x, dtype=np.float64).tolist()
        return Variable(float_tensor(x))
    elif dtype == "long":
        x = np.array(x, dtype=np.long).tolist()
        return Variable(long_tensor(x))
    elif dtype == "byte":
        x = np.array(x, dtype=np.byte).tolist()
        return Variable(byte_tensor(x))
    else:
        x = np.array(x, dtype=np.float64).tolist()
        return Variable(float_tensor(x))


class ActorNetwork(nn.Module):
    def __init__(self, state_dim, hidden_size, output_size, output_act):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.output_act = output_act

    def __call__(self, state):
        out = nn.functional.relu(self.fc1(state))
        out = nn.functional.relu(self.fc2(out))
        out = self.output_act(self.fc3(out))
        return out


class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, output_size=1):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size + action_dim, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def __call__(self, state, action):
        out = nn.functional.relu(self.fc1(state))
        out = th.cat([out, action], 1)
        out = nn.functional.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class A2CBot:
    def __init__(self, state_dim, action_dim,
                 actor_hidden_size=128, critic_hidden_size=128,
                 actor_output_act=nn.functional.log_softmax, critic_loss="mse",
                 actor_lr=0.001, critic_lr=0.001,
                 optimizer_type="rmsprop", entropy_reg=0.01,
                 max_grad_norm=0.5):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor_hidden_size = actor_hidden_size
        self.critic_hidden_size = critic_hidden_size
        self.actor_output_act = actor_output_act
        self.critic_loss = critic_loss
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.optimizer_type = optimizer_type
        self.entropy_reg = entropy_reg
        self.max_grad_norm = max_grad_norm
        self.actor = ActorNetwork(self.state_dim, self.actor_hidden_size, self.action_dim, self.actor_output_act)
        self.critic = CriticNetwork(self.state_dim, self.action_dim, self.critic_hidden_size, 1)
        if self.optimizer_type == "adam":
            self.actor_optimizer = Adam(self.actor.parameters(), lr=self.actor_lr)
            self.critic_optimizer = Adam(self.critic.parameters(), lr=self.critic_lr)
        elif self.optimizer_type == "rmsprop":
            self.actor_optimizer = RMSprop(self.actor.parameters(), lr=self.actor_lr)
            self.critic_optimizer = RMSprop(self.critic.parameters(), lr=self.critic_lr)
        print("=" * 50)
        for x, y in locals().items():
            print(f"{x} -> {y}")
        print("=" * 50)

    def save_model(self, path_name):
        th.save(self.actor, path_name + "/actor.pkl")
        th.save(self.critic, path_name + "/critic.pkl")

    def load_model(self, path_name):
        self.actor = th.load(path_name + "/actor.pkl")
        self.critic = th.load(path_name + "/critic.pkl")

    # train on a roll out batch
    def train(self, batch):
        states_var = to_tensor_var(batch.states).view(-1, self.state_dim)
        one_hot_actions = index_to_one_hot(batch.actions, self.action_dim)
        actions_var = to_tensor_var(one_hot_actions).view(-1, self.action_dim)
        rewards_var = to_tensor_var(batch.rewards).view(-1, 1)
        loss = [None, None]

        # update actor network
        self.actor_optimizer.zero_grad()
        action_log_probs = self.actor(states_var)
        entropy_loss = th.mean(entropy(th.exp(action_log_probs)))
        action_log_probs = th.sum(action_log_probs * actions_var, 1)
        values = self.critic(states_var, actions_var)
        advantages = rewards_var - values.detach()
        pg_loss = -th.mean(action_log_probs * advantages)
        actor_loss = pg_loss - entropy_loss * self.entropy_reg
        loss[0] = float(actor_loss)
        if actor_loss != actor_loss:
            print(f"got invalid actor loss:{actor_loss} = {pg_loss} - {entropy_loss} * {self.entropy_reg}")
            actor_loss.data = th.tensor(0.0)
        actor_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        # update critic network
        self.critic_optimizer.zero_grad()
        target_values = rewards_var
        if self.critic_loss == "huber":
            critic_loss = nn.functional.smooth_l1_loss(values, target_values)
        else:
            critic_loss = nn.MSELoss()(values, target_values)
        loss[1] = float(critic_loss)
        if critic_loss != critic_loss:
            print(f"got invalid critic loss:{critic_loss}")
            critic_loss.data = th.tensor(0.0)
        critic_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()
        return loss

    # predict softmax action based on state
    def _softmax_action(self, state):
        state_var = to_tensor_var([state])
        softmax_action_var = th.exp(self.actor(state_var))
        softmax_action = softmax_action_var.data.numpy()[0]
        return softmax_action

    # choose an action based on state for execution
    def action(self, state):
        softmax_action = self._softmax_action(state)
        action = np.argmax(softmax_action)
        return action

    # evaluate value for a state-action pair
    def value(self, state, action):
        state_var = to_tensor_var([state])
        action = index_to_one_hot(action, self.action_dim)
        action_var = to_tensor_var([action])
        value_var = self.critic(state_var, action_var)
        value = value_var.data.numpy()[0]
        return value
