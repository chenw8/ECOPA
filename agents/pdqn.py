"""
是一个参数化动作空间的DQN (Deep Q-Network) 变种, 称为PDQN (Parameterized DQN);
该智能体适用于那些动作不仅包含离散部分还包含连续参数的情况;
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import Counter
from torch.autograd import Variable
import heapq
from agents.agent import Agent
from agents.memory.memory import Memory
from agents.utils import soft_update_target_network, hard_update_target_network
from agents.utils.noise import OrnsteinUhlenbeckActionNoise

import matplotlib.pyplot as plt

"""
评估不同动作的Q值
输入是状态和动作参数，输出是一个代表不同动作可能性的Q值矩阵。通过softmax函数转换为概率，并最终选择一个动作;
"""
class QActor(nn.Module):            #输入state和ParamActor的输出action_parameter_size=30。输出10维数组

    def __init__(self, state_size, action_size, action_parameter_size): #90，10，30
        super(QActor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.action_parameter_size = action_parameter_size

        # create layers
        self.layers = nn.ModuleList()
        inputSize = self.state_size + self.action_parameter_size
        self.layers.append(nn.Linear(inputSize, 256))
        self.layers.append(nn.Linear(256, 256))
        self.layers.append(nn.Linear(256, self.action_size*3))


    def forward(self, state, action_parameters):

        x = torch.cat((state, action_parameters), dim=1)
        x = F.relu(self.layers[0](x))
        x = F.relu(self.layers[1](x))
        Q = self.layers[-1](x)
        Q=Q.view(-1,3)   #变为10*3
        action_probs = F.softmax(Q, dim=-1)  # Softmax 使得概率和为 1
        action_indices=torch.multinomial(action_probs, num_samples=1)
        for i in range(len(action_indices)):
            action_indices[i]+=1
        return action_indices

"""
生成连续动作参数
根据当前的状态来生成相应的动作参数。它的输出是一组动作参数，这些参数可以影响或决定具体的动作执行方式
action_parameter_size 动作参数的总维度，即需要输出的连续参数的数量，需要映射输出的维度
"""
class ParamActor(nn.Module):    #输入state，输出维度为action_parameter_size=30

    def __init__(self, state_size, action_size, action_parameter_size):  #90，10，30
        super(ParamActor, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.action_parameter_size = action_parameter_size
        # self.squashing_function = squashing_function

        # create layers
        self.layers = nn.ModuleList()
        inputSize = self.state_size
        self.layers.append(nn.Linear(inputSize, 256))
        self.layers.append(nn.Linear(256, 256))

        self.action_parameters_output_layer = nn.Linear(256, self.action_parameter_size)
        self.action_parameters_passthrough_layer = nn.Linear(self.state_size, self.action_parameter_size)


        nn.init.zeros_(self.action_parameters_passthrough_layer.weight)
        nn.init.zeros_(self.action_parameters_passthrough_layer.bias)
        # fix passthrough layer to avoid instability, rest of network can compensate
        self.action_parameters_passthrough_layer.requires_grad = False
        self.action_parameters_passthrough_layer.weight.requires_grad = False
        self.action_parameters_passthrough_layer.bias.requires_grad = False

    def forward(self, state):
        x = state

        # print(x.shape)


        x = F.relu(self.layers[0](x))
        x = F.relu(self.layers[1](x))
        action_params = self.action_parameters_output_layer(x)
        action_params += self.action_parameters_passthrough_layer(state)
        # if self.squashing_function:
        #     assert False  # scaling not implemented yet
        #     action_params = action_params.tanh()
        #     action_params = action_params * self.action_param_lim
        return abs(action_params)   #直接用绝对值取正




class PDQNAgent(Agent): # 用于处理具有混合离散-连续动作空间的强化学习问题
    """
    DDPG actor-critic agent for parameterised action spaces
    [Hausknecht and Stone 2016]
    """

    NAME = "P-DQN Agent"

    def __init__(self,
                 observation_space,
                 action_space,
                 actor_class=QActor,
                 actor_param_class=ParamActor,
                 epsilon_initial=1.0,
                 epsilon_final=0.05,
                 epsilon_steps=10000,
                 batch_size=64,
                 gamma=0.99,
                 tau_actor=0.01,  # Polyak averaging factor for copying target weights
                 tau_actor_param=0.001,
                 replay_memory_size=1000000,
                 learning_rate_actor=0.0001,
                 learning_rate_actor_param=0.00001,
                 initial_memory_threshold=0,
                 use_ornstein_noise=False,  # if false, uses epsilon-greedy with uniform-random action-parameter exploration
                 loss_func=F.mse_loss, # F.mse_loss
                 clip_grad=10,
                 inverting_gradients=False,
                 zero_index_gradients=False,
                 indexed=False,
                 weighted=False,
                 average=False,
                 random_weighted=False,
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 seed=None):

        super(PDQNAgent, self).__init__(observation_space, action_space)
        self.device = torch.device(device)
        self.num_actions = 10    # self.num_actions =self.action_space.spaces[0].n 。原为3。表示离散动作的个数
        self.action_parameter_sizes = np.array([1 for i in range(1,31)])
        self.action_parameter_size = int(self.action_parameter_sizes.sum())  #30

        """------------------需要修改--------------"""
        self.action_max = torch.from_numpy(np.ones((self.num_actions,))).float().to(device)
        self.action_min = -self.action_max.detach()
        self.action_range = (self.action_max-self.action_min).detach()
        self.action_parameter_max_numpy =np.full((self.action_parameter_size,), 500) #原为1000，一个任务分配的资源在[0,500]
        self.action_parameter_min_numpy =np.zeros((self.action_parameter_size,))
        self.action_parameter_range_numpy = (self.action_parameter_max_numpy - self.action_parameter_min_numpy)
        self.action_parameter_max = torch.from_numpy(self.action_parameter_max_numpy).float().to(device)
        self.action_parameter_min = torch.from_numpy(self.action_parameter_min_numpy).float().to(device)
        self.action_parameter_range = torch.from_numpy(self.action_parameter_range_numpy).float().to(device)

        self.epsilon = epsilon_initial
        self.epsilon_initial = epsilon_initial
        self.epsilon_final = epsilon_final
        self.epsilon_steps = epsilon_steps
        self.indexed = indexed
        self.weighted = weighted
        self.average = average
        self.random_weighted = random_weighted
        assert (weighted ^ average ^ random_weighted) or not (weighted or average or random_weighted)

        self.action_parameter_offsets = self.action_parameter_sizes.cumsum()  #得到累计和，即[1,2,3 ……30]
        self.action_parameter_offsets = np.insert(self.action_parameter_offsets, 0, 0)  #在最前面加0，即[0,1,2,3 ……30]
        self.batch_size = batch_size
        self.gamma = gamma
        self.replay_memory_size = replay_memory_size
        self.initial_memory_threshold = initial_memory_threshold
        self.learning_rate_actor = learning_rate_actor
        self.learning_rate_actor_param = learning_rate_actor_param
        self.inverting_gradients = inverting_gradients
        self.tau_actor = tau_actor
        self.tau_actor_param = tau_actor_param
        self._step = 0
        self._episode = 0
        self.updates = 0
        self.clip_grad = clip_grad
        self.zero_index_gradients = zero_index_gradients

        self.np_random = None
        self.seed = seed
        self._seed(seed)

        self.use_ornstein_noise = use_ornstein_noise
        self.noise = OrnsteinUhlenbeckActionNoise(self.action_parameter_size, random_machine=self.np_random, mu=0., theta=0.15, sigma=0.0001) #, theta=0.01, sigma=0.01)

        print("--------",self.num_actions+self.action_parameter_size)
        # print(observation_space.shape,self.observation_space.shape[0])
        self.replay_memory = Memory(replay_memory_size, (90,), (1+self.action_parameter_size,), next_actions=False)
        self.actor = actor_class(90, self.num_actions, self.action_parameter_size).to(device)
        self.actor_target = actor_class(90, self.num_actions, self.action_parameter_size).to(device)
        hard_update_target_network(self.actor, self.actor_target)
        self.actor_target.eval()

        self.actor_param = actor_param_class(90, self.num_actions, self.action_parameter_size).to(device)
        self.actor_param_target = actor_param_class(90, self.num_actions, self.action_parameter_size).to(device)
        hard_update_target_network(self.actor_param, self.actor_param_target)
        self.actor_param_target.eval()

        self.loss_func = loss_func  # l1_smooth_loss performs better but original paper used MSE

        # Original DDPG paper [Lillicrap et al. 2016] used a weight decay of 0.01 for Q (critic)
        # but setting weight_decay=0.01 on the critic_optimiser seems to perform worse...
        # using AMSgrad ("fixed" version of Adam, amsgrad=True) doesn't seem to help either...
        self.actor_optimiser = optim.Adam(self.actor.parameters(), lr=self.learning_rate_actor) #, betas=(0.95, 0.999))
        self.actor_param_optimiser = optim.Adam(self.actor_param.parameters(), lr=self.learning_rate_actor_param) #, betas=(0.95, 0.999)) #, weight_decay=critic_l2_reg)
        self.cost_his = []

        self.macro_action=[]

        self.macro_action_0=[]
        self.macro_action_1 = []
        self.macro_action_2 = []

        self.macro_action_1_0=[]
        self.macro_action_2_0=[]
        self.macro_action_1_1=[]
        self.macro_action_2_1=[]
        self.macro_action_1_2=[]
        self.macro_action_2_2=[]
        self.macro_action_reward=[]
        self.macro_action_final=[]
        self.macro_action_flag=True
    def __str__(self):
        desc = super().__str__() + "\n"
        desc += "Actor Network {}\n".format(self.actor) + \
                "Param Network {}\n".format(self.actor_param) + \
                "Actor Alpha: {}\n".format(self.learning_rate_actor) + \
                "Actor Param Alpha: {}\n".format(self.learning_rate_actor_param) + \
                "Gamma: {}\n".format(self.gamma) + \
                "Tau (actor): {}\n".format(self.tau_actor) + \
                "Tau (actor-params): {}\n".format(self.tau_actor_param) + \
                "Inverting Gradients: {}\n".format(self.inverting_gradients) + \
                "Replay Memory: {}\n".format(self.replay_memory_size) + \
                "Batch Size: {}\n".format(self.batch_size) + \
                "Initial memory: {}\n".format(self.initial_memory_threshold) + \
                "epsilon_initial: {}\n".format(self.epsilon_initial) + \
                "epsilon_final: {}\n".format(self.epsilon_final) + \
                "epsilon_steps: {}\n".format(self.epsilon_steps) + \
                "Clip Grad: {}\n".format(self.clip_grad) + \
                "Ornstein Noise?: {}\n".format(self.use_ornstein_noise) + \
                "Zero Index Grads?: {}\n".format(self.zero_index_gradients) + \
                "Seed: {}\n".format(self.seed)
        return desc

    def set_action_parameter_passthrough_weights(self, initial_weights, initial_bias=None):
        passthrough_layer = self.actor_param.action_parameters_passthrough_layer

        assert initial_weights.shape == passthrough_layer.weight.data.size()
        passthrough_layer.weight.data = torch.Tensor(initial_weights).float().to(self.device)
        if initial_bias is not None:
            assert initial_bias.shape == passthrough_layer.bias.data.size()
            passthrough_layer.bias.data = torch.Tensor(initial_bias).float().to(self.device)
        passthrough_layer.requires_grad = False
        passthrough_layer.weight.requires_grad = False
        passthrough_layer.bias.requires_grad = False
        hard_update_target_network(self.actor_param, self.actor_param_target)

    def _seed(self, seed=None):
        """
        NOTE: this will not reset the randomly initialised weights; use the seed parameter in the constructor instead.

        :param seed:
        :return:
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self.np_random = np.random.RandomState(seed=seed)  #初始化随机数生成器。
        if seed is not None:
            torch.manual_seed(seed)
            if self.device == torch.device("cuda"):
                torch.cuda.manual_seed(seed)      #设置GPU上操作的随机种子

    def _ornstein_uhlenbeck_noise(self, all_action_parameters):
        """ Continuous action exploration using an Ornstein–Uhlenbeck process. """
        return all_action_parameters.data.numpy() + (self.noise.sample() * self.action_parameter_range_numpy)

    def start_episode(self):
        pass

    def end_episode(self):
        self._episode += 1

        ep = self._episode
        if ep < self.epsilon_steps:
            self.epsilon = self.epsilon_initial - (self.epsilon_initial - self.epsilon_final) * (
                    ep / self.epsilon_steps)
        else:
            self.epsilon = self.epsilon_final

    def act(self, state):        # 根据当前状态选择动作       #返回action是10维离散，action_parameters是30维连续
        with torch.no_grad():
            state = torch.from_numpy(state).to(self.device)
            
            
            # print(state.shape)
            

            all_action_parameters = self.actor_param.forward(state)  #30维
            for i in range(len(all_action_parameters)):
                all_action_parameters[i] *=35
            # print("all_action_parameters",all_action_parameters)

            # Hausknecht and Stone [2016] use epsilon greedy actions with uniform random action-parameter exploration
            rnd = self.np_random.uniform()   #生成[0,1]的浮点数   探索-利用平衡
            if rnd < 0.3:    #1.0:
                action=[]
                for i in range(self.num_actions):
                    action.append(self.np_random.choice(3)+1)
                # action = self.np_random.choice(self.num_actions)
                if not self.use_ornstein_noise:  #从连续参数的范围里选择30个连续参数
                    all_action_parameters=[]
                    all_action_parameters.append(torch.from_numpy(np.random.uniform(self.action_parameter_min_numpy,
                                                            self.action_parameter_max_numpy)))



                    # all_action_parameters=torch.from_numpy(np.array(all_action_parameters)).squeeze(0)

                    if isinstance(all_action_parameters, list) and len(all_action_parameters) == 1:
                        # 列表中只有一个元素，并且是tensor，直接取出来
                        all_action_parameters = all_action_parameters[0]


            else:
                # select maximum action
                # print("sssssssssssssss",state.unsqueeze(0),all_action_parameters.unsqueeze(0))
                Q_a = self.actor.forward(state.unsqueeze(0), all_action_parameters.unsqueeze(0))
                Q_a = Q_a.detach().cpu().data.numpy()
                Q_a=Q_a.reshape(1,-1)
                action=Q_a[0]

            # add noise only to parameters of chosen action
            all_action_parameters = all_action_parameters.cpu().data.numpy()
            offset = np.array([self.action_parameter_sizes[i] for i in range(len(action))], dtype=int).sum() #10

            if self.use_ornstein_noise and self.noise is not None:
                all_action_parameters[offset:offset + self.action_parameter_sizes[action]] += self.noise.sample()[offset:offset + self.action_parameter_sizes[action]]
            action_parameters = all_action_parameters            #[offset:offset+self.action_parameter_sizes[action]]
            # if self._step%100==0 :
            #     print("action, all_action_parameters",action, all_action_parameters)
        return action, action_parameters, all_action_parameters

    def _zero_index_gradients(self, grad, batch_action_indices, inplace=True): # 选择性地将梯度置零，只保留与所选动作相关的参数梯度
        assert grad.shape[0] == batch_action_indices.shape[0]
        grad = grad.cpu()

        if not inplace:
            grad = grad.clone()
        with torch.no_grad():
            ind = torch.zeros(self.action_parameter_size, dtype=torch.long)
            for a in range(self.num_actions):
                ind[self.action_parameter_offsets[a]:self.action_parameter_offsets[a+1]] = a
            # ind_tile = np.tile(ind, (self.batch_size, 1))
            ind_tile = ind.repeat(self.batch_size, 1).to(self.device)
            actual_index = ind_tile != batch_action_indices[:, np.newaxis]
            grad[actual_index] = 0.
        return grad
# P-DDPG
    def _invert_gradients(self, grad, vals, grad_type, inplace=True): # 修改梯度，使得参数更新不会超出预定的参数范围
        # 5x faster on CPU (for Soccer, slightly slower for Goal, Platform?)
        if grad_type == "actions":
            max_p = self.action_max
            min_p = self.action_min
            rnge = self.action_range
        elif grad_type == "action_parameters":
            max_p = self.action_parameter_max
            min_p = self.action_parameter_min
            rnge = self.action_parameter_range
        else:
            raise ValueError("Unhandled grad_type: '"+str(grad_type) + "'")

        max_p = max_p.cpu()
        min_p = min_p.cpu()
        rnge = rnge.cpu()
        grad = grad.cpu()
        vals = vals.cpu()

        assert grad.shape == vals.shape

        if not inplace:
            grad = grad.clone()
        with torch.no_grad():
            # index = grad < 0  # actually > but Adam minimises, so reversed (could also double negate the grad)
            index = grad > 0
            grad[index] *= (index.float() * (max_p - vals) / rnge)[index]
            grad[~index] *= ((~index).float() * (vals - min_p) / rnge)[~index]

        return grad

    def step(self, state, action, reward, next_state, next_action, terminal, time_steps=1): # 处理智能体与环境交互的单个步骤，存储经验并触发学习过程
        # print(action)
        act, all_action_parameters = action
        self._step += 1
        a=np.concatenate(([act],all_action_parameters)).ravel()

        self._add_sample(state, np.concatenate(([act],all_action_parameters)).ravel(), reward, next_state, np.concatenate(([next_action[0]],next_action[1])).ravel(), terminal=terminal)

        if self._step >= self.batch_size and self._step >= self.initial_memory_threshold:
            self._optimize_td_loss()
            self.updates += 1

    #TODO:取reward最高的macro_action
    # def get_macro_action(self,max_macro_action_number=10):
    #     # re1 = heapq.nlargest(max_macro_action_number, self.macro_action_2)  # 求最大的三个元素，并排序
    #     # print("re1",re1)
    #
    #     re_0 = map(self.macro_action_2_0.index, heapq.nlargest(max_macro_action_number, self.macro_action_2_0))  # 求最大的三个索引    nsmallest与nlargest相反，求最小
    #     re_1 = map(self.macro_action_2_1.index, heapq.nlargest(max_macro_action_number, self.macro_action_2_1))  # 求最大的三个索引    nsmallest与nlargest相反，求最小
    #     re_2 = map(self.macro_action_2_2.index, heapq.nlargest(max_macro_action_number, self.macro_action_2_2))  # 求最大的三个索引    nsmallest与nlargest相反，求最小
    #     re_0=list(re_0)
    #     re_1=list(re_1)
    #     re_2=list(re_2)
    #
    #     for i in re_0:
    #         self.macro_action_0.append(self.macro_action_1_0[i])
    #     for i in re_1:
    #         self.macro_action_1.append(self.macro_action_1_1[i])
    #     for i in re_2:
    #         self.macro_action_2.append(self.macro_action_1_2[i])
    #     self.macro_action.extend(self.macro_action_0)
    #     self.macro_action.extend(self.macro_action_1)
    #     self.macro_action.extend(self.macro_action_2)
    #     print("self.macro_action",self.macro_action)
    #     self.macro_action_flag=False

    def _add_sample(self, state, action, reward, next_state, next_action, terminal):
        assert len(action) == 1 + self.action_parameter_size
        self.replay_memory.append(state, action, reward, next_state, terminal=terminal)

    def _optimize_td_loss(self): # 从经验回放中采样一批数据，计算TD误差并更新actor和param_actor网络的权重；
        # TD误差（Temporal Difference Error），即时间差分误差，是强化学习中一个核心概念，主要用于评估和改进智能体的预测能力。
        # 它通过比较当前时刻对未来的预测与下一时刻的实际观察来更新价值函数或策略
        if self._step < self.batch_size or self._step < self.initial_memory_threshold:
            return
        # Sample a batch from replay memory
        states, actions, rewards, next_states, terminals = self.replay_memory.sample(self.batch_size, random_machine=self.np_random)

        states = torch.from_numpy(states).to(self.device)

        actions_combined = torch.from_numpy(actions).to(self.device)  # make sure to separate actions and parameters
        actions = actions_combined[:, 0].long()
        action_parameters = actions_combined[:, 1:]
        rewards = torch.from_numpy(rewards).to(self.device).squeeze()
        next_states = torch.from_numpy(next_states).to(self.device)
        terminals = torch.from_numpy(terminals).to(self.device).squeeze()

        # print("dfgdfgfgdn", terminals)
        # ---------------------- optimize Q-network ----------------------
        with torch.no_grad():
            pred_next_action_parameters = self.actor_param_target.forward(next_states)
            pred_Q_a = self.actor_target(next_states, pred_next_action_parameters)
            Qprime = torch.max(pred_Q_a, 1, keepdim=True)[0].squeeze()  # 下一个状态s′进行预测以得到最大Q值（Qprime）
            # Compute the TD error
            target = rewards + (1 - terminals) * self.gamma * Qprime # Qtarget​=r+γmaxa′​Q(s′,a′)
        # Compute current Q-values using policy network
        q_values = self.actor(states, action_parameters)

        y_predicted = q_values.gather(1, actions.view(-1, 1)).squeeze()
        y_expected = target
        loss_Q = self.loss_func(y_predicted, y_expected)

        self.actor_optimiser.zero_grad()   #清空上一步残留参数值
        loss_Q.backward()
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_grad)
        self.actor_optimiser.step()   # 将参数更新值施加到 net 的 parameters 上

        # ---------------------- optimize actor ----------------------
        with torch.no_grad():
            action_params = self.actor_param(states)

        action_params.requires_grad = True
        assert (self.weighted ^ self.average ^ self.random_weighted) or \
               not (self.weighted or self.average or self.random_weighted)
        Q = self.actor(states, action_params)
        Q_val = Q
        if self.weighted:
            # approximate categorical probability density (i.e. counting)
            counts = Counter(actions.cpu().numpy())
            weights = torch.from_numpy(
                np.array([counts[a] / actions.shape[0] for a in range(self.num_actions)])).float().to(self.device)
            Q_val = weights * Q
        elif self.average:
            Q_val = Q / self.num_actions
        elif self.random_weighted:
            weights = np.random.uniform(0, 1., self.num_actions)
            weights /= np.linalg.norm(weights)
            weights = torch.from_numpy(weights).float().to(self.device)
            Q_val = weights * Q
        if self.indexed:
            Q_indexed = Q_val.gather(1, actions.unsqueeze(1))
            Q_loss = torch.mean(Q_indexed)
        else:
            Q_loss = torch.mean(torch.sum(Q_val, 1))   #求和再取均值
        # print('Q')
        # print(Q_loss)
        x=Q_loss
        x_np = x.data.cpu().numpy()
        self.cost_his.append(x_np)

        self.actor.zero_grad()
        Q_loss.backward()

        from copy import deepcopy
        # print('-------------')
        # print(action_params.grad.data)
        delta_a = deepcopy(action_params.grad.data)
        # step 2
        action_params = self.actor_param(Variable(states))

        delta_a[:] = self._invert_gradients(delta_a, action_params, grad_type="action_parameters", inplace=True)

        if self.zero_index_gradients:
            delta_a[:] = self._zero_index_gradients(delta_a, batch_action_indices=actions, inplace=True)

        out = -torch.mul(delta_a, action_params)
        # print(out)
        self.actor_param.zero_grad()
        out.backward(torch.ones(out.shape).to(self.device))

        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.actor_param.parameters(), self.clip_grad)

        self.actor_param_optimiser.step()

        soft_update_target_network(self.actor, self.actor_target, self.tau_actor)
        soft_update_target_network(self.actor_param, self.actor_param_target, self.tau_actor_param)

    def save_models(self, prefix):
        """
        saves the target actor and critic models
        :param prefix: the count of episodes iterated
        :return:
        """
        torch.save(self.actor.state_dict(), prefix + '_actor.pt')
        torch.save(self.actor_param.state_dict(), prefix + '_actor_param.pt')
        print('Models saved successfully')

    def load_models(self, prefix):
        """
        loads the target actor and critic models, and copies them onto actor and critic models
        :param prefix: the count of episodes iterated (used to find the file name)
        :param target: whether to load the target newtwork too (not necessary for evaluation)
        :return:
        """
        # also try load on CPU if no GPU available?
        self.actor.load_state_dict(torch.load(prefix + '_actor.pt', map_location='cpu'))
        self.actor_param.load_state_dict(torch.load(prefix + '_actor_param.pt', map_location='cpu'))
        print('Models loaded successfully')

    def plot_cost(self):
        plt.plot(np.arange(len(self.cost_his)), self.cost_his, c='y', label='pdqn_actor_loss')
        plt.legend(loc='best')
        plt.ylabel('loss')
        plt.xlabel('Training Steps')
        plt.show()

