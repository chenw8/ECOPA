import copy
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 负责生成给定状态下的动作(连续和离散的动作)
class Actor(nn.Module):
    def __init__(self, state_dim, discrete_action_dim, parameter_action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3_1 = nn.Linear(256, discrete_action_dim*10)  #6*10(10个用户) - Ml1
        self.l3_2 = nn.Linear(256, parameter_action_dim) #6

        self.lstm = nn.LSTMCell(state_dim+66, 66)  #Ml1+l2的维度为66

        self.max_action = max_action

    def forward(self, state):
        (state), (hx, cx) = state
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        discrete_action = self.max_action * torch.tanh(self.l3_1(a))
        parameter_action = 3 * torch.tanh(self.l3_2(a))
        """parameter_action = self.max_action * torch.tanh(self.l3_2(a))"""
        x=torch.cat([discrete_action,parameter_action,state],dim=1)
        hx, cx = self.lstm(x, (hx, cx))
        discrete_action=hx[:,:60]     #取所有行前60列
        parameter_action=hx[:,-6:]
        return discrete_action, parameter_action,hx,cx


class Critic(nn.Module):     #critic框架中有6层，实现Q1和Q2，而不用两个网络。只用一个critic和一个目标critic就行
    def __init__(self, state_dim, discrete_action_dim, parameter_action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + discrete_action_dim*10 + parameter_action_dim, 256) #输入的离散维度是Ml1(60)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + discrete_action_dim*10 + parameter_action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, discrete_action, parameter_action):
        sa = torch.cat([state, discrete_action, parameter_action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, discrete_action, parameter_action):   #只用前三层网络计算一个Q
        sa = torch.cat([state, discrete_action, parameter_action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

class Pre(nn.Module):
    def __init__(self,input_dim,output_dim):
        #输入现在的位置以及要预测的某时刻t，输出时刻t的位置
        super(Pre, self).__init__()

        self.l1=nn.Linear(input_dim,128)
        self.l2=nn.Linear(128,128)
        self.l3=nn.Linear(128,output_dim)

    def forward(self,input):
        x=F.relu(self.l1(input))
        x=F.relu(self.l2(x))
        output=self.l3(x)
        output=output.clamp(-5000,17000)
        return output


class TD3(object):
    def __init__(
            self,
            state_dim,
            discrete_action_dim,   #6 ei
            parameter_action_dim,  #6 z
            max_action,            #1.0
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2
    ):
        self.discrete_action_dim = discrete_action_dim #6
        self.parameter_action_dim = parameter_action_dim #6

        self.user_num=10
        self.ES_num=3

        '''
        -------------------------需要修改----------------------离散动作与连续动作的范围。看看是潜空间的范围还是真实空间的范围。这好像是梯度范围？
        '''
        self.action_max = torch.from_numpy(np.ones((self.discrete_action_dim*10,))).float().to(device) #维度为60的全1数组
        self.action_min = -self.action_max.detach()
        self.action_range = (self.action_max - self.action_min).detach()   #维度为60的全2向量

        self.action_parameter_max = torch.from_numpy(np.ones((self.parameter_action_dim,))*3).float().to(device)
        """没有*3"""
        self.action_parameter_min = -self.action_parameter_max.detach()
        # print(" self.action_parameter_max_numpy", self.action_parameter_max)
        self.action_parameter_range = (self.action_parameter_max - self.action_parameter_min)

        #actor与目标actor  输出动作为ei和z，都是6维
        self.actor = Actor(state_dim, discrete_action_dim, parameter_action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)  # 默认3e-4

        #critic与目标critic
        self.critic = Critic(state_dim, discrete_action_dim, parameter_action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        #预测网络
        input_dim=4
        output_dim=2
        self.pre1=Pre(input_dim,output_dim)
        self.pre1_optimizer = torch.optim.Adam(self.pre1.parameters(), lr=3e-4)
        self.pre2 = Pre(input_dim, output_dim)
        self.pre2_optimizer = torch.optim.Adam(self.pre2.parameters(), lr=3e-4)
        self.pre3 = Pre(input_dim, output_dim)
        self.pre3_optimizer = torch.optim.Adam(self.pre3.parameters(), lr=3e-4)
        self.pre4 = Pre(input_dim, output_dim)
        self.pre4_optimizer = torch.optim.Adam(self.pre4.parameters(), lr=3e-4)
        self.pre5 = Pre(input_dim, output_dim)
        self.pre5_optimizer = torch.optim.Adam(self.pre5.parameters(), lr=3e-4)
        self.pre6 = Pre(input_dim, output_dim)
        self.pre6_optimizer = torch.optim.Adam(self.pre6.parameters(), lr=3e-4)
        self.pre7 = Pre(input_dim, output_dim)
        self.pre7_optimizer = torch.optim.Adam(self.pre7.parameters(), lr=3e-4)
        self.pre8 = Pre(input_dim, output_dim)
        self.pre8_optimizer = torch.optim.Adam(self.pre8.parameters(), lr=3e-4)
        self.pre9 = Pre(input_dim, output_dim)
        self.pre9_optimizer = torch.optim.Adam(self.pre9.parameters(), lr=3e-4)
        self.pre10 = Pre(input_dim, output_dim)
        self.pre10_optimizer = torch.optim.Adam(self.pre10.parameters(), lr=3e-4)
        self.pre_list=[self.pre1,self.pre2,self.pre3,self.pre4,self.pre5,self.pre6,self.pre7,self.pre8,self.pre9,self.pre10]
        self.pre_optimizer_list=[self.pre1_optimizer,self.pre2_optimizer,self.pre3_optimizer,self.pre4_optimizer,self.pre5_optimizer,self.pre6_optimizer,self.pre7_optimizer,self.pre8_optimizer,self.pre9_optimizer,self.pre10_optimizer]

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0  #？？？

    def select_action(self, state,actor_hx,actor_cx):  # 输出动作为Ml1(ei)和l2(z)，分别为60维和6维
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        all_discrete_action, all_parameter_action,hx,cx = self.actor((state, (actor_hx, actor_cx)))
        return all_discrete_action.cpu().data.numpy().flatten(), all_parameter_action.cpu().data.numpy().flatten(),hx,cx

    def train(self, replay_buffer, action_rep, c_rate, recon_s_rate, batch_size=250):

        no_of_samples_from_episode = 32
        step_size = 2
        index_numpy_base_list = replay_buffer.sample_base_indexes(no_of_samples_from_episode, int(batch_size/10),
                                                                  step_size=step_size)
        ones_batch = np.ones(batch_size)

        #传入缓冲区、VAE表示空间、每个潜空间维度的上下界、δ预测损失与batch_size
        recon_s_rate = recon_s_rate * 2.0   #原为5.0
        self.total_it += 1
        # Sample replay buffer- 采样256组数据
        index_list=(index_numpy_base_list).tolist()  #
        """---------------在经验池中加入结束时间，用于预测网络输入------------"""
        """---------------在经验池中加入预测结束时间的位置，用于预测网络输入------------"""
        state, discrete_action, parameter_action, all_parameter_action, discrete_emb, parameter_emb, next_state, state_next_state, reward, done,t_com,pre_loc,actor_hx,actor_cx,miu = replay_buffer.sample_buffer(
            batch_size,index_list)
        # print("parameter_emb",parameter_emb)  #都在-1到1之间
        #得到的discrete_action是1维、parameter_action是2维、all_parameter_action是12维、discrete_emb和parameter_emb是6维、
        # print("discrete_emb----------",discrete_emb)
        # print("discrete_action.shape",discrete_action.shape)  #torch.Size([25, 10, 10])
        # print("parameter_action",parameter_action)
        # print("reward",reward)
        # print("actor_hx",actor_hx)
        # print("actor_cx",actor_cx)
        with torch.no_grad():
            #输入从D中取到的256个离散动作k(1维),展成1行256个数组，再用squeeze转为256行的一维张量(一个列向量，256个数字，一个数字表示一个离散动作)1
            #输入到嵌入表中，得到对应的256个连续向量ei(6维)
            #根据采样池中的k得到的ei
            discrete_emb = discrete_emb.view(1, -1).reshape(250, 60)

            """自己修改部分     ---------   把用k得到的ei拼接起来成60维的ei，要输入vae中"""
            discrete_action=discrete_action.view(1,-1).reshape(250,10)  #转为1行再变为250*10
            x1=action_rep.get_embedding(int(discrete_action[0][0])-1).to(device)
            for i in range(1, self.user_num):
                y = action_rep.get_embedding(int(discrete_action[0][i]) - 1).to(device)
                x1 = torch.cat((x1, y), dim=0)
            for j in range(1,len(discrete_action)):
                x = action_rep.get_embedding(int(discrete_action[j][0])-1).to(device)
                for i in range(1,self.user_num):
                    y = action_rep.get_embedding(int(discrete_action[j][i]) - 1).to(device)
                    x = torch.cat((x, y), dim=0)
                x1=torch.cat((x1, x), dim=0)
            discrete_emb_ = x1.reshape(250, -1).to(device)
            #torch.Size([250, 60])
            """自己修改部分"""
            # discrete_emb_ = action_rep.get_embedding(discrete_action.reshape(1,
            #                                                                  -1).squeeze().long()).to(device)

            # discrete relable need noise
            noise_discrete = (          #生成一个和discrete_emb_维度相同的噪声向量，噪声从N(0,1)随机采样，再×0.1，最后再用clampx轴范围[-0.5,0.5]
                    torch.randn_like(discrete_emb_) * 0.1
            ).clamp(-self.noise_clip, self.noise_clip)
            discrete_emb_table = discrete_emb_.clamp(-self.max_action, self.max_action).to(device)    #约束得到的ei
            discrete_emb_table_noise = (discrete_emb_ + noise_discrete).clamp(-self.max_action, self.max_action)#加上噪音后的ei

            discrete_action_old = action_rep.select_discrete_action(discrete_emb)#输入采样池中的ei，选择动作k。输入的torch.size([250,60])
            d_new = discrete_action.reshape(-1,10).cpu().numpy()   #采样池现在存储的动作k(1维)
            # print("d_new", d_new)  #250,10
            d_old = discrete_action_old             #根据采样池中的ei得到的动作k(1维)
            d_bing = (d_new == d_old) * 1           #250*10

            # discrete_relable_rate
            #discrete_relable_rate是在一批中(256组动作中)，存储的k和使用ei得到的k相同的比例
            discrete_relable_rate = 1 #sum(d_bing.reshape(1, -1)[0]) / batch_size   #d_bing.reshape(1, -1)[0]是把二维变为一维(即一行)
            d_bing = torch.FloatTensor(d_bing).float().to(device)
            #优化根据采样池中的k得到的ei，如果k与通过ei得到的k相同的位置，就用采样池中的ei，如果不一样，就用通过采样池中的k得到的ei加上噪音

            discrete_emb=discrete_emb.reshape(250,10,6)
            discrete_emb_table_noise=discrete_emb_table_noise.reshape(250,10,6)
            discrete_emb_ = d_bing.unsqueeze(-1) * discrete_emb + (1.0 - d_bing.unsqueeze(-1)) * discrete_emb_table_noise
            # print("discrete_emb_final",discrete_emb_)

            state=state.reshape(250,90).to(device)
            next_state=next_state.reshape(250,90).to(device)
            parameter_emb=parameter_emb.reshape(250,6).to(device)
            #分别为torch.Size([250, 90])、torch.Size([250, 6])、torch.Size([250, 60])
            predict_delta_state = action_rep.select_delta_state(state, parameter_emb, discrete_emb_table)#传入s，z和ei，得到预测的δ。

            # print("predict_delta_state",predict_delta_state)
            # print("predict_delta_state.shape",predict_delta_state.shape)
            # print("state_next_state",state_next_state.cpu().numpy())
            # print("state_next_state.shape",state_next_state.shape)
            state_next_state = state_next_state.reshape(250,90)
            #得到每个δ-δ预测的MSE，最后转为列向量，表示一列数字，每个数字表示每组样本的LDYN？
            delta_state = (np.square(predict_delta_state - state_next_state.cpu().numpy())).mean(axis=1).reshape(-1, 1)
            # delta_state=predict_delta_state-state_next_state.cpu().numpy()
            # delta_state=np.mean(delta_state, axis=1).reshape(-1, 1)
            s_bing = (abs(delta_state) < recon_s_rate) * 1  #现在计算的LDYN与传入的训练VAE时的LDYN相比较，小于则是1，其他为0
            # print("abs(delta_state)",abs(delta_state))   #几百到几千
            # print("recon_s_rate",recon_s_rate)   #10000+
            parameter_relable_rate = 1 #sum(s_bing.reshape(1, -1)[0]) / batch_size  #计算这批样本中小于的比例
            s_bing = torch.FloatTensor(s_bing).float().to(device)
            # print("s_bing",s_bing)

            #传入s，约束后的ei，样本中的连续参数
            parameter_action=parameter_action.reshape(250,-1)
            recon_c, recon_s, mean, std = action_rep.vae(state, discrete_emb_table, parameter_action)
            parameter_emb_ = mean + std * torch.randn_like(std)
            #parameter_emb_表示连续动作，256行，6列。每行表示每个样本的连续动作z
            for i in range(len(parameter_emb_[0])):   #parameter_emb_[:, i:i + 1]是选择所有行，第i列处理。约束z的范围
                parameter_emb_[:, i:i + 1] = self.true_parameter_emb(parameter_emb_[:, i:i + 1], c_rate, i)
            # print("parameter_emb",parameter_emb)
            # print("parameter_emb_",parameter_emb_)

            parameter_emb_ = s_bing * parameter_emb + (1 - s_bing) * parameter_emb_
            # print("parameter_emb_final",parameter_emb_)

            discrete_emb_ = discrete_emb_.clamp(-self.max_action, self.max_action)
            parameter_emb_ = parameter_emb_.clamp(-self.max_action, self.max_action)

            discrete_emb = discrete_emb_     #最后得到的ei 。也是60
            parameter_emb = parameter_emb_.to(device)   #最后得到的z


        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise_discrete = (       #ei的噪音
                    torch.randn_like(discrete_emb) * self.policy_noise  #policy_noise为默认0.2
            ).clamp(-self.noise_clip, self.noise_clip)  #噪音的约束noise_clip为0.5
            noise_parameter = (      #z的噪音
                    torch.randn_like(parameter_emb) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            """-------ei要取平均"""
            actor_hx=actor_hx.reshape(250,-1).to(device)
            actor_cx=actor_cx.reshape(250,-1).to(device)
            noise_discrete=noise_discrete.reshape(250,60)
            noise_parameter=noise_parameter.reshape(250,6)

            next_discrete_action, next_parameter_action,hx,cx = self.actor_target((next_state,(actor_hx, actor_cx))) #使用目标actor得到下一个状态的动作ei和z(ei是60维，z是6维)
            next_discrete_action = (next_discrete_action + noise_discrete).clamp(-self.max_action, self.max_action).to(device) #下一个ei加噪音,next_discrete_action.shape torch.Size([250, 60])
            next_parameter_action = (next_parameter_action + noise_parameter).clamp(-3, 3).to(device) #下一个z加噪音
            """-----------clamp(-self.max_action, self.max_action)-------------"""
            # Compute the target Q value
            #传入s'，下一个ei，下一个z得到
            target_Q1, target_Q2 = self.critic_target(next_state, next_discrete_action, next_parameter_action) #一个critic得到Q1Q2
            target_Q = torch.min(target_Q1, target_Q2)  #选最小
            reward=reward.reshape(250,1)
            done=done.reshape(250,1)
            target_Q = reward + (1-done) * self.discount * target_Q   #计算TD误差

        # Get current Q estimates
        #计算现在的Q
        discrete_emb=discrete_emb.reshape(250,60).to(device)
        current_Q1, current_Q2 = self.critic(state, discrete_emb, parameter_emb)

        # Compute critic loss
        #计算critic的损失函数
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic  - 更新critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 位置预测网络更新 - 经验池中数据需要加时刻t,t从1开始
        state_=state[:,0:70].reshape(250,10,7)
        user_id=state_[:,:,0]   #取所有id 浮点数 torch.Size([250, 10])
        pre_loc=pre_loc.reshape(250,10,2)
        t_com=t_com.reshape(250,10)
        # print("pre_loc",pre_loc)
        # print("pre_loc.shape",pre_loc.shape)  #torch.Size([25, 10, 10, 2])
        # print("t_com",t_com)
        # print("t_com.shape",t_com.shape)  #torch.Size([25, 10, 10])
        self.train_pre_net(pre_loc,user_id,t_com)
        # print("out_pre_train")


        # Delayed policy updates  - policy延迟更新
        if self.total_it % self.policy_freq == 0:
            inverting_gradients = True
            # inverting_gradients = False
            # Compute actor loss
            
            
            # 梯度反转的基本思想是：
            #    当动作值接近动作空间边界时，传统的梯度更新可能会将动作推出有效范围
            #    梯度反转会改变梯度方向，使其始终指向动作空间的有效区域内部
            # 为什么可以让两部分更稳定：
            # 在混合动作空间中，离散动作和连续动作的梯度规模和性质可能差异很大；传统梯度更新可能导致某一部分（离散或连续）的更新幅度过大
            # 梯度反转确保所有动作分量的更新都保持在有效范围内；这种方法避免了边界处的梯度爆炸或消失问题
            if inverting_gradients:  #当动作空间是离散+混合时，梯度更新可能不平衡，用梯度反转强制让两部分梯度对其，更稳定？？？
                with torch.no_grad():
                    next_discrete_action, next_parameter_action,hx,cx = self.actor((state,(actor_hx, actor_cx)))
                    action_params = torch.cat((next_discrete_action, next_parameter_action), dim=1) #dim=1表示每行的ei和z拼接成一行
                action_params.requires_grad = True   #使用反向传播backward时计算梯度
                actor_loss = self.critic.Q1(state, action_params[:, :self.discrete_action_dim*10],
                                            action_params[:, self.discrete_action_dim*10:]).mean()
            else:
                next_discrete_action, next_parameter_action,hx,cx = self.actor((state,(actor_hx, actor_cx)))
                #负的Q值，是TD3标准loss函数
                actor_loss = -self.critic.Q1(state, next_discrete_action, next_parameter_action).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()

            if inverting_gradients:    #如果梯度反转。对梯度进行裁剪
                from copy import deepcopy
                #上一步actor_loss.backward()已经计算更新的梯度
                delta_a = deepcopy(action_params.grad.data)  #把ei和z拼接后的向量的梯度复制给delta_a，data不涉及自动求导
                # 2 - apply inverting gradients and combine with gradients from actor
                actions, action_params,hx_,cx_ = self.actor((state,(actor_hx, actor_cx)))   #actions是ei，action_params是z
                action_params = torch.cat((actions, action_params), dim=1)  #再把ei和z拼接赋值给action_params
                delta_a[:, self.discrete_action_dim*10:] = self._invert_gradients(
                    delta_a[:, self.discrete_action_dim*10:].cpu(),    #delta_a的z的梯度
                    action_params[:, self.discrete_action_dim*10:].cpu(),   #刚得到的action_params的z的值
                    grad_type="action_parameters", inplace=True)
                delta_a[:, :self.discrete_action_dim*10] = self._invert_gradients(   #0-60维
                    delta_a[:, :self.discrete_action_dim*10].cpu(),        #ei的梯度
                    action_params[:, :self.discrete_action_dim*10].cpu(),   #ei的值
                    grad_type="actions", inplace=True)
                out = -torch.mul(delta_a, action_params) #mul是逐元素相乘。表示每个参数更新量，取负是在负梯度方向更新
                self.actor.zero_grad()
                #out是通过actor计算得到的，所以对out反向传播会更新actor的参数
                #backword是计算损失函数对网络中每个参数的偏导数来计算梯度
                #如果out是一个向量，backward括号中要加一个与out形状相同的向量，表示out每个量在反向传播时的权重(重要性)，越大说明越重要，更新时变化量越大
                #在这里用了全1向量，表示所有参数的权重相同
                out.backward(torch.ones(out.shape).to(device))
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.) #对梯度裁剪，防止梯度过大。10是L2范数，是阈值

            # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.)
            #actor的loss和梯度反转的损失造成的梯度计算都计入grad中，然后更新actor的参数
            self.actor_optimizer.step()
            # print("out_train")

            # Update the frozen target models
            #软更新目标网络
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # actor_hx.detach()
        # actor_cx.detach()

        return discrete_relable_rate, parameter_relable_rate,x


    # def save(self, filename, directory):
    #     torch.save(self.vae.state_dict(), '%s/%s_vae.pth' % (directory, filename))
    #     # torch.save(self.vae.embeddings, '%s/%s_embeddings.pth' % (directory, filename))
    #
    # def load(self, filename, directory):
    #     self.vae.load_state_dict(torch.load('%s/%s_vae.pth' % (directory, filename), map_location=self.device))
    #     # self.vae.embeddings = torch.load('%s/%s_embeddings.pth' % (directory, filename), map_location=self.device)


    def save(self, filename,directory):
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
        torch.save(self.critic_optimizer.state_dict(), '%s/%s_critic_optimizer.pth' % (directory, filename))

        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.actor_optimizer.state_dict(), '%s/%s_actor_optimizer.pth' % (directory, filename))

    def load(self, filename,directory):
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
        self.critic_optimizer.load_state_dict(torch.load('%s/%s_critic_optimizer.pth' % (directory, filename)))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.actor_optimizer.load_state_dict(torch.load('%s/%s_actor_optimizer.pth' % (directory, filename)))
        self.actor_target = copy.deepcopy(self.actor)

    def _invert_gradients(self, grad, vals, grad_type, inplace=True):#对正梯度与负梯度缩放，确保不会超出范围
        #grad是delta_a的z或ei的梯度   vals是action_params的z或ei的动作的值   grad_type是"actions"或"action_parameters"

        if grad_type == "actions":
            max_p = self.action_max.cpu()
            min_p = self.action_min.cpu()
            rnge = self.action_range.cpu()
        elif grad_type == "action_parameters":
            max_p = self.action_parameter_max.cpu()
            min_p = self.action_parameter_min.cpu()
            rnge = self.action_parameter_range.cpu()
        else:
            raise ValueError("Unhandled grad_type: '" + str(grad_type) + "'")

        assert grad.shape == vals.shape  #确保形状一致

        if not inplace:  #inplace为false，克隆梯度，不会对原来梯度修改。为True，会修改原来梯度
            grad = grad.clone()
        with torch.no_grad():
            for n in range(grad.shape[0]):  #每一行 - 250
                # index = grad < 0  # actually > but Adam minimises, so reversed (could also double negate the grad)
                index = grad[n] > 0   #index是一个与grad形状相同的数组，为正的位置是True，反之为False
                grad[n][index] *= (index.float() * (max_p - vals[n]) / rnge)[index] #所有index为true的项进行这一步操作。根据最大值缩放梯度范围
                grad[n][~index] *= ((~index).float() * (vals[n] - min_p) / rnge)[~index]  #所有index为false的项进行这一步操作

        return grad

    def count_boundary(self, c_rate):   #c_rate中有每个维度的上下界。[[x,y],[a,b]]，hardgoal中一共有6维
        median = (c_rate[0] - c_rate[1]) / 2
        offset = c_rate[0] - 1 * median
        return median, offset      #返回每个维度上下界的中位数与偏移量

    def true_parameter_emb(self, parameter_action, c_rate, i):  #归一化每一维，使得其在约束范围内(c_rate)
        # parameter_action_ = parameter_action.clone()
        median, offset = self.count_boundary(c_rate[i])
        # parameter_action_[i] = parameter_action_[i] * median + offset
        parameter_action = (parameter_action - offset) / median
        return parameter_action

    def loc_pre(self,t,t_com,loc,tau):           #output为10维向量，表示预测的位置
        output=[]
        for i in range(len(tau)):
            tt=t[i]
            t_c=t_com[i]
            location=loc[i]
            input=np.concatenate([[tt],[t_c],location])  #拼接后为3维
            id=int(tau[i][0])
            pre=self.pre_list[id-1]
            input=torch.tensor(input).float()
            x=pre(input)            #x是2维
            output.append(x)
        return output

    def read_csv_to_array(self,filename):  # 读取csv文件
        with open(filename, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            data = list(reader)  # 将CSV内容转换为列表
        return data

    def get_t_loc(self,t):   #传入时刻t，返回时刻t的位置信息。loc是250*10，每行表示一个时刻的所有用户的坐标，有10项
        loc=[]
        for i in range(len(t)):  #250
            loc_temp=[]
            for j in range(10):  #10个用户
                x=[]
                if j<9:
                    location=self.read_csv_to_array('user_trace/NewYork_30sec_00{}.csv'.format(j+1))
                if j==9:
                    location=self.read_csv_to_array('user_trace/NewYork_30sec_010.csv')
                if t[i][j] >= 100:
                    x.append(float(location[-1][1]))
                    x.append(float(location[-1][2]))
                else:
                    x.append(float(location[int(t[i][j])][1]))
                    x.append(float(location[int(t[i][j])][2]))
                loc_temp.append(x)
            loc.append(loc_temp)
        loc=torch.tensor(loc).float().to(device)
        return loc


    # 预测用户位置
    def train_pre_net(self,pre_loc,user_id,t_com):
        real=self.get_t_loc(t_com).clone().detach().requires_grad_(True).to(device)   #real是250*10*2的tensor。每行表示t_com数组中每个时刻10个用户的真实位置
        pre =pre_loc.requires_grad_(True)
        pre_loss = F.mse_loss(real, pre, reduction='none')  #pre_loss为250*10*2
        pre_loss_sample=pre_loss.sum(dim=-1)  # 对最后一维求和。pre_loss_sample为250*10
        for i in range(len(user_id)): #250。     user_id是250*10
            for j in range(len(user_id[0])): #10
                id=int(user_id[i][j])
                loss=pre_loss_sample[i][j]
                self.pre_optimizer_list[id-1].zero_grad()
                loss.backward(retain_graph=True)
                self.pre_optimizer_list[id-1].step()
