import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, state_dim, discrete_action_dim, parameter_action_dim,all_parameter_action_dim ,discrete_emb_dim, parameter_emb_dim,
                 max_size=int(1e6)): #action_dim是15维度(M)
        self.mem_size = max_size
        self.cnt = 0    #当前指针
        self.size = 0   #当前容量
        self.storage = []

        self.state = np.zeros((self.mem_size, state_dim))

        self.discrete_action = np.zeros((max_size, discrete_action_dim))
        self.parameter_action = np.zeros((max_size, parameter_action_dim))
        self.all_parameter_action = np.zeros((max_size, all_parameter_action_dim))

        self.discrete_action_dim=discrete_action_dim
        self.parameter_action_dim=parameter_action_dim
        self.all_parameter_action_dim=all_parameter_action_dim

        self.discrete_emb = np.zeros((max_size, discrete_emb_dim*10))
        self.parameter_emb = np.zeros((max_size, parameter_emb_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.state_next_state = np.zeros((max_size, state_dim))

        self.reward = np.zeros((self.mem_size,1 ))#每个奖励
        self.done = np.zeros((self.mem_size,1 )) #是否不是中止状态

        self.t_com = np.zeros((self.mem_size,10 )) #中止时间
        self.pre_loc = np.zeros((self.mem_size,10 )) #预测的结束位置

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 将新的信息存入缓冲区
    def add(self, state, discrete_action, parameter_action, all_parameter_action,discrete_emb, parameter_emb, next_state,state_next_state, reward, done,t_com,pre_loc,actor_hx,actor_cx,miu):
        #加入actor的hx和cx
        if discrete_emb is None:
            discrete_emb = torch.full((self.discrete_action_dim*10,),float('nan')).to(self.device)
        if parameter_emb is None:
            parameter_emb = torch.full((self.parameter_action_dim,),float('nan')).to(self.device)
        if all_parameter_action is None:
            all_parameter_action = torch.full((self.all_parameter_action_dim,),float('nan')).to(self.device)
        if actor_hx is None:
            actor_hx = torch.full((66,),float('nan')).to(self.device)
        if actor_cx is None:
            actor_cx = torch.full((66,),float('nan')).to(self.device)

        transition=(state, discrete_action, parameter_action, all_parameter_action,discrete_emb, parameter_emb, next_state,state_next_state, reward, done,t_com,pre_loc,actor_hx,actor_cx,miu)
        if len(self.storage) == self.mem_size:
            self.storage[int(self.cnt)] = transition
            self.cnt = (self.cnt + 1) % self.mem_size
        else:
            self.storage.append(transition)
        # self.state[self.ptr] = state
        # self.discrete_action[self.ptr] = discrete_action
        # self.parameter_action[self.ptr] = parameter_action
        # self.all_parameter_action[self.ptr] = all_parameter_action
        # self.discrete_emb[self.ptr] = discrete_emb
        # self.parameter_emb[self.ptr] = parameter_emb
        # self.next_state[self.ptr] = next_state
        # self.state_next_state[self.ptr] = state_next_state
        #
        # self.reward[self.ptr] = reward
        # self.not_done[self.ptr] = 1. - done

        # mem_idx = self.mem_cnt % self.mem_size #0，1，2，3……
        self.size = min(self.size + 1, self.mem_size)

    def sample_base_indexes(self, no_of_samples_from_episode, batch_size, step_size=2):
        # no_of_samples_from_episode=32
        # ind = np.random.randint(0, len(self.storage), size=batch_size)
        no_of_records = self.size
        no_of_episode_records = int(no_of_records / 100)  # 选择100个连续的时间
        episode_no_list = np.random.randint(0, no_of_episode_records,
                                            size=batch_size)  # 生成batch_size个0到no_of_episode_records-1的随机数，可重复
        offset_list = np.random.randint(0, 89, size=batch_size)  #最大选择89，一次取10个连续的
        index_list = episode_no_list * 100 + offset_list
        return index_list


    # 从缓冲区随机采样，返回一批次的转移信息
    def sample_buffer(self,batch_size,index_list):#随机抽取存储样例
        # mem_len = min(self.mem_size, self.mem_cnt)
        # batch = np.random.choice(mem_len, self.batch_size, replace=True)  # 采样指定数量的索引。返回数组

        # 从0-mem_len-1中随机采样batch_size个索引。replace=True表示可以重复采样。
        # ind = np.random.randint(0, self.size, size=batch_size)  # 生成指定范围内的size个随机数

        state,discrete_action,parameter_action,all_parameter_action,discrete_emb,parameter_emb,next_state,state_next_state,reward,done,t_com,pre_loc,actor_hx,actor_cx,miu = [],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
        window_size=10
        for i in index_list:
            state_, discrete_action_, parameter_action_, all_parameter_action_, discrete_emb_, parameter_emb_, next_state_, state_next_state_, reward_, done_,t_com_,pre_loc_,actor_hx_,actor_cx_,miu_ =[],[], [], [], [], [], [], [], [], [], [],[],[],[],[]
            for j in range(window_size):
                state_.append(np.array(self.storage[i+j][0],copy=False))
                discrete_action_.append(np.array(self.storage[i+j][1],copy=False))
                parameter_action_.append(np.array(self.storage[i+j][2],copy=False))
                all_parameter_action_.append(np.array(self.storage[i+j][3].cpu(),copy=False))
                discrete_emb_.append(np.array(self.storage[i+j][4],copy=False))
                parameter_emb_.append(np.array(self.storage[i+j][5],copy=False))
                next_state_.append(np.array(self.storage[i+j][6],copy=False))
                state_next_state_.append(np.array(self.storage[i+j][7],copy=False))
                reward_.append(self.storage[i+j][8])
                done_.append(self.storage[i+j][9])
                t_com_.append(np.array(self.storage[i+j][10],copy=False))
                pre_loc_.append(self.storage[i+j][11])
                actor_hx_.append(self.storage[i+j][12])
                actor_cx_.append(self.storage[i+j][13])
                miu_.append(self.storage[i+j][14])
            state.append(state_)
            discrete_action.append(discrete_action_)
            parameter_action.append(parameter_action_)
            all_parameter_action.append(all_parameter_action_)
            discrete_emb.append(discrete_emb_)
            parameter_emb.append(parameter_emb_)
            next_state.append(next_state_)
            state_next_state.append(state_next_state_)
            reward.append(reward_)
            done.append(done_)
            t_com.append(t_com_)
            pre_loc.append(pre_loc_)
            actor_hx.append(actor_hx_)
            actor_cx.append(actor_cx_)
            miu.append(miu_)
        # reward: [[tensor(-149.0731), tensor(-149.4151), tensor(-99.6072)]]
        return (
            torch.FloatTensor(state).to(self.device),
            torch.FloatTensor(discrete_action).to(self.device),  # batch_size行、1列
            torch.FloatTensor(parameter_action).to(self.device),  # batch_size行、2列
            torch.FloatTensor(all_parameter_action).to(self.device),  # batch_size行、12列
            torch.FloatTensor(discrete_emb).to(self.device),  # batch_size行、6列
            torch.FloatTensor(parameter_emb).to(self.device),  # batch_size行、6列
            torch.FloatTensor(next_state).to(self.device),
            torch.FloatTensor(state_next_state).to(self.device),
            torch.FloatTensor(reward).to(self.device),
            torch.FloatTensor(done).to(self.device),
            torch.FloatTensor(t_com).to(self.device),
            torch.FloatTensor([[[tensor.detach().cpu().numpy() for tensor in row] for row in batch] for batch in pre_loc]).to(self.device),
            torch.FloatTensor([[[tensor.detach().cpu().numpy() for tensor in row] for row in batch] for batch in actor_hx]).to(self.device),
            torch.FloatTensor([[[tensor.detach().cpu().numpy() for tensor in row] for row in batch] for batch in actor_cx]).to(self.device),
            torch.FloatTensor(reward).to(self.device)
        )
    def sample(self,batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)  # 生成指定范围内的size个随机数
        batch = [self.storage[i] for i in ind]
        return (
            torch.FloatTensor([t[0] for t in batch]).to(self.device),
            torch.FloatTensor([t[1] for t in batch]).to(self.device),
            torch.FloatTensor([t[2] for t in batch]).to(self.device),
            torch.FloatTensor([t[3] for t in batch]).to(self.device),
            [t[4] for t in batch],
            [t[5] for t in batch],
            torch.FloatTensor([t[6] for t in batch]).to(self.device),
            torch.FloatTensor([t[7] for t in batch]).to(self.device),
            torch.FloatTensor([t[8] for t in batch]).to(self.device),
            torch.FloatTensor([t[9] for t in batch]).to(self.device)
        )

