# TODO: s discrete continue ->s"
import numpy as np
import torch
from torch import float32
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from embedding.Utils.utils import NeuralNet, pairwise_distances, pairwise_hyp_distances, squash, atanh
from embedding.Utils import Basis
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.nn.functional as functional


# Vanilla Variational Auto-Encoder
class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, action_embedding_dim, parameter_action_dim, latent_dim, max_action,
                 hidden_size=256):
        super(VAE, self).__init__()

        self.user_num = 10  #用户数目
        self.ES_num=3

        # embedding table
        init_tensor = torch.rand(self.ES_num,    #一个3行(只有三个离散动作，表示服务器123)、action_embedding_dim(6)列的[0,1)的矩阵，最后在[-1,1]
                                 action_embedding_dim) * 2 - 1  # Don't initialize near the extremes.
        #定义一个可学习的嵌入参数，Parameter表示能自动求导，并将其放入模型的参数列表中，requires_grad=True表示可学习
        #embeddings是一个二维张量，action_dim行、action_embedding_dim列。embeddings[action]表示第action行内容；嵌入表
        self.embeddings = torch.nn.Parameter(init_tensor.type(float32), requires_grad=True)
        print("self.embeddings", self.embeddings)
        #encoder网络结构
        self.e0_0 = nn.Linear(state_dim + action_embedding_dim*self.user_num, hidden_size)  #s维度+d1维度(6)*M(10用户)
        self.e0_1 = nn.Linear(parameter_action_dim, hidden_size)              #输入30维度，选连续参数，表示Xp。原是2，是(x,y)

        self.e1 = nn.Linear(hidden_size, hidden_size)
        self.e2 = nn.Linear(hidden_size, hidden_size)
        self.mean = nn.Linear(hidden_size, latent_dim)                  #latent_dim是d2维度
        self.log_std = nn.Linear(hidden_size, latent_dim)

        #decoder网络结构
        self.d0_0 = nn.Linear(state_dim + action_embedding_dim*self.user_num, hidden_size)  #s维度+d1维度(条件)*M(10用户)
        self.d0_1 = nn.Linear(latent_dim, hidden_size)                        #d2维度  6
        self.d1 = nn.Linear(hidden_size, hidden_size)
        self.d2 = nn.Linear(hidden_size, hidden_size)

        self.parameter_action_output = nn.Linear(hidden_size, parameter_action_dim)  #输出维度为30。原是2

        self.d3 = nn.Linear(hidden_size, hidden_size)

        self.delta_state_output = nn.Linear(hidden_size, state_dim)                   #s维度(预测δ)

        self.max_action = max_action
        self.latent_dim = latent_dim

    # forward 方法接收输入的状态 (state)、离散动作嵌入 (action) 和连续参数 (action_parameter)，
    # 然后通过编码器部分计算潜在空间的均值和标准差，并生成潜在变量 z。接着，使用这个潜在变量 z 通过解码器部分来预测连续参数和状态的变化
    def forward(self, state, action, action_parameter):    #s，ei，连续参数 (传入的ei是10*6维(Ml1))； 
        # 接收输入：状态(state)、离散动作嵌入(action)和连续参数(action_parameter)

        z_0 = F.relu(self.e0_0(torch.cat([state, action], 1)))
        z_1 = F.relu(self.e0_1(action_parameter))
        z = z_0 * z_1

        z = F.relu(self.e1(z))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)  #clamp是把每个元素限制在[-4,15]之间

        std = torch.exp(log_std)   #直接对标准差进行学习可能会导致数值不稳定，而使用对数标准差可以避免负值。
        z = mean + std * torch.randn_like(std)  #torch.randn_like(std)是形状与std相同的随机数(噪声)，维度为mean相同
        u, s = self.decode(state, z, action)      #返回30维的连续参数与预测的s

        # 输出：潜在空间的均值和标准差
        return u, s, mean, std

    def decode(self, state, z=None, action=None, clip=None, raw=False):      #输出30维的连续参数以及预测δ。这个action需要是10*6维(Ml1)
        # 输入：状态、潜在变量z、离散动作嵌入
        
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            z = torch.randn((state.shape[0], self.latent_dim)).to(device)  #batch size行、d2列的数组，存标准正态分布的随机值
            if clip is not None:
                z = z.clamp(-clip, clip)    #把z中元素限制在self.clip的范围内
        v_0 = F.relu(self.d0_0(torch.cat([state, action], 1)))
        v_1 = F.relu(self.d0_1(z))
        v = v_0 * v_1
        v = F.relu(self.d1(v))
        v = F.relu(self.d2(v))

        parameter_action = self.parameter_action_output(v)  #输出的是30维，表示连续参数，原为2

        v = F.relu(self.d3(v))
        s = self.delta_state_output(v)

        if raw: return parameter_action, s
        #使用relu变为正再×max_action(用relu或者取绝对值呢？)
        
        # 输出：重建的连续参数和预测的状态变化
        return self.max_action * torch.relu(parameter_action), torch.tanh(s)


# Action_representation 是一个更高层次的动作表示模块，它使用 VAE 来实现对动作的降维和重建
class Action_representation(NeuralNet):
    def __init__(self,
                 state_dim,
                 action_dim,        #11 k
                 parameter_action_dim,  #30 原为选定k之后的连续参数xk,是2。现在是30
                 reduced_action_dim=2,  #6 ei  -默认为2
                 reduce_parameter_action_dim=2,    #6 z
                 embed_lr=1e-4,
                 ):
        super(Action_representation, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.parameter_action_dim = parameter_action_dim
        self.reduced_action_dim = reduced_action_dim
        self.reduce_parameter_action_dim = reduce_parameter_action_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        # Action embeddings to project the predicted action into original dimensions
        # latent_dim=action_dim*2+parameter_action_dim*2
        self.latent_dim = self.reduce_parameter_action_dim      #6 z的维度
        self.embed_lr = embed_lr

        self.user_num = 10  #用户数目
        self.ES_num = 3

        self.vae = VAE(state_dim=self.state_dim, action_dim=self.action_dim,
                       action_embedding_dim=self.reduced_action_dim, parameter_action_dim=self.parameter_action_dim,
                       latent_dim=self.latent_dim, max_action=1.0,
                       hidden_size=256).to(self.device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=1e-4)

    def discrete_embedding(self, ):
        emb = self.vae.embeddings

        return emb

    def unsupervised_loss(self, s1, a1, a2, s2, sup_batch_size, embed_lr): #这里s1是state,a1是离散动作k，a2是连续参数，s2是s-s'
        #这个函数输出损失值，并且已经利用损失值训练过vae（train step）
        """自己修改部分   -------------  -把ei通过k变为(64,60)形式  """
        x = (self.get_embedding(int(a1[0][0])-1)).to(device)
        for i in range(1, self.user_num):
            y = self.get_embedding(int(a1[0][i]) - 1).to(device)
            x = torch.cat((x, y), dim=0)
        a_ = x.reshape(1, -1).squeeze().long().to(device)   #a_是60维
        for j in range(1,a1.shape[0]):   #遍历64(batch_size)
            x = (self.get_embedding(int(a1[j][0])-1)).to(device)
            for i in range(1, self.user_num):
                y = self.get_embedding(int(a1[j][i]) - 1).to(device)
                x = torch.cat((x, y), dim=0)
            x=x.reshape(1, -1).squeeze().long().to(device)
            a_=torch.cat([a_,x])
        a_=a_.reshape(64, -1)
        """自己修改部分"""
        #a1是torch.Size([60])，s1是torch.Size([64, 90])
        # a1 = self.get_embedding(a1).to(self.device)   #k经过嵌入表得到ei

        s1 = s1.to(self.device)
        s2 = s2.to(self.device)
        a2 = a2.to(self.device)

        vae_loss, recon_loss_d, recon_loss_c, KL_loss = self.train_step(s1, a_, a2, s2, sup_batch_size, embed_lr)
        return vae_loss, recon_loss_d, recon_loss_c, KL_loss

    def loss(self, state, action_d, action_c, next_state, sup_batch_size):  #用在train step函数里
        # action_d是离散动作k对应的ei，action_c是连续参数，next_state是s-s'
        recon_c, recon_s, mean, std = self.vae(state, action_d, action_c)    #s，ei，连续参数p。传入的action_d应该是60维
        #recon_c、recon_s是解码后的连续动作和预测δ

        recon_loss_s = F.mse_loss(recon_s, next_state, size_average=True)     #预测损失，这里的next_state是s-s'(LD)
        recon_loss_c = F.mse_loss(recon_c, action_c, size_average=True)       #LVAE第一项(LV第一项)

        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()

        # vae_loss = 0.25 * recon_loss_s + recon_loss_c + 0.5 * KL_loss
        # vae_loss = 0.25 * recon_loss_s + 2.0 * recon_loss_c + 0.5 * KL_loss  #best
        #vae_loss=LDYN+2*LAVE第一项+0.5*KL
        vae_loss = recon_loss_s + 2.0 * recon_loss_c + 0.5 * KL_loss
        # print("vae loss",vae_loss)
        # return vae_loss, 0.25 * recon_loss_s, recon_loss_c, 0.5 * KL_loss
        # return vae_loss, 0.25 * recon_loss_s, 2.0 * recon_loss_c, 0.5 * KL_loss #best
        #返回LHYAR、LDYN、2*LAVE第一项、0.5*KL
        return vae_loss, recon_loss_s, 2.0 * recon_loss_c, 0.5 * KL_loss

    def train_step(self, s1, a1, a2, s2, sup_batch_size, embed_lr=1e-4):   #s1是state,a1是离散动作k对应的ei，a2是连续参数，s2是s-s'
        state = s1
        action_d = a1
        action_c = a2
        next_state = s2
        vae_loss, recon_loss_s, recon_loss_c, KL_loss = self.loss(state, action_d, action_c, next_state,
                                                                  sup_batch_size)

        # 更新VAE - 嵌入表的参数被放入vae的参数列表中一并更新
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=embed_lr)
        self.vae_optimizer.zero_grad()
        vae_loss.backward()
        self.vae_optimizer.step()

        return vae_loss.cpu().data.numpy(), recon_loss_s.cpu().data.numpy(), recon_loss_c.cpu().data.numpy(), KL_loss.cpu().data.numpy()

    def select_parameter_action(self, state, z, action): #输入s，z，ei给解码器，返回预测的xk'
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            z = torch.FloatTensor(z.reshape(1, -1)).to(self.device)
            action = torch.FloatTensor(action.reshape(1, -1)).to(self.device)
            action_c, state = self.vae.decode(state, z, action)   #得到预测的xk'和δ
        return action_c.cpu().data.numpy().flatten()

    # def select_delta_state(self, state, z, action):
    #     with torch.no_grad():
    #         state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
    #         z = torch.FloatTensor(z.reshape(1, -1)).to(self.device)
    #         action = torch.FloatTensor(action.reshape(1, -1)).to(self.device)
    #         action_c, state = self.vae.decode(state, z, action)
    #     return state.cpu().data.numpy().flatten()
    def select_delta_state(self, state, z, action):    #用法同上，输入s，z，ei给解码器，返回预测的δ
        with torch.no_grad():
            action_c, state = self.vae.decode(state, z, action)
        return state.cpu().data.numpy()

    def get_embedding(self, action):  #输入动作k，得到对应的ei
        #得到embedding中相应的动作
        # Get the corresponding target embedding
        action_emb = self.vae.embeddings[action]
        action_emb = torch.tanh(action_emb)  #tanh把ei的每个维度放到-1到1
        return action_emb

    def get_match_scores(self, action):    #输入是actor输出的离散动作，是ei，6维
        # compute similarity probability based on L2 norm
        embeddings = self.vae.embeddings
        embeddings = torch.tanh(embeddings)
        action = action.to(self.device)
        # compute similarity probability based on L2 norm
        #action是一个1行×discrete_emb_dim（6)的张量，embeddings是一个(3,action_embedding_dim(6))的张量
        #计算欧氏距离，返回similarity表示欧氏距离，返回的是负数，最后取similarity中最大值，表示绝对值最小，也是最近的动作
        similarity = - pairwise_distances(action, embeddings)  # Negate euclidean to convert diff into similarity score
        # 返回一个k维的行向量(discrete_action_dim)
        return similarity

        # 获得最优动作，输出于embedding最相近的action 作为最优动作.

    def select_discrete_action(self, action):  #输入actor输出的离散动作，是60维，表示M个6维ei。返回最好动作的索引(一个或一组)
        best_action = []
        if len(action)==1:
            action=action.reshape(self.user_num,self.reduced_action_dim)  #化为10*6维
            for i in range(self.user_num):
                act = action[i]  # act是torch.size([6])
                similarity = self.get_match_scores(act)  # 返回1*3的数组
                val, pos = torch.max(similarity, dim=1)
                if len(pos) == 1:
                    best_action.append(pos.cpu().item() + 1)  # 得到的是索引012？应该+1变成123
                else:
                    best_action.append(pos[0].cpu().item() + 1)  # 直接选择第一个  #best_action.append(pos.cpu().numpy())
        if len(action)==250:   #用在TD3里处理250组动作
            action = action.reshape(-1,self.user_num, self.reduced_action_dim)  # 化为250*10*6维
            for j in range(len(action)):
                bet_act=[]
                for i in range(self.user_num):
                    act=action[j][i]
                    similarity=self.get_match_scores(act)
                    val, pos = torch.max(similarity, dim=1)
                    if len(pos) == 1:
                        bet_act.append(pos.cpu().item() + 1)
                    else:
                        bet_act.append(pos[1].cpu().item() + 1)
                best_action.append(bet_act)
        return best_action
        # similarity = self.get_match_scores(action)   #返回256(batch）个k维的行向量(discrete_action_dim)
        # val, pos = torch.max(similarity, dim=1)  #val为值，pos为索引.dim=1表示找到每一行中最大的索引
        # # print("pos",pos,len(pos))
        # if len(pos) == 1:
        #     return pos.cpu().item()  # data.numpy()[0]
        # else:
        #     # print("pos.cpu().item()", pos.cpu().numpy())
        #     return pos.cpu().numpy()

    def save(self, filename, directory):
        torch.save(self.vae.state_dict(), '%s/%s_vae.pth' % (directory, filename))
        # torch.save(self.vae.embeddings, '%s/%s_embeddings.pth' % (directory, filename))

    def load(self, filename, directory):
        self.vae.load_state_dict(torch.load('%s/%s_vae.pth' % (directory, filename), map_location=self.device))
        # self.vae.embeddings = torch.load('%s/%s_embeddings.pth' % (directory, filename), map_location=self.device)

    def get_c_rate(self, s1, a1, a2, s2, batch_size=100, range_rate=5):#返回每个潜空间维度的上下界数组(c_rate=[x,y],[a,b]……共六个[]，每一个表示一个维度的上下界)与经验池中的δ和用经验池中的动作计算的δ的LDYN
        # 得到c。输入replaybuffer中的s、1维的动作size个，弄成一行、2维度动作size个，弄成一行、buffer中存储的δ

        """自己修改部分"""
        x = (self.get_embedding(int(a1[0][0]) - 1)).to(device)
        for i in range(1, self.user_num):
            y = self.get_embedding(int(a1[0][i]) - 1).to(device)
            x = torch.cat((x, y), dim=0)
        a_ = x.reshape(1, -1).squeeze().long().to(device)  # a_是60维
        for j in range(1, a1.shape[0]):  # 遍历64(batch_size)
            x = (self.get_embedding(int(a1[j][0]) - 1)).to(device)
            for i in range(1, self.user_num):
                y = self.get_embedding(int(a1[j][i]) - 1).to(device)
                x = torch.cat((x, y), dim=0)
            x = x.reshape(1, -1).squeeze().long().to(device)
            a_ = torch.cat([a_, x])
        a_ = a_.reshape(5000, -1)
        """自己修改部分"""

        # a1 = self.get_embedding(a1).to(self.device)  #这里的a1是1×size的，每一个表示一个数，难道是选定的离散动作k？
        s1 = s1.to(self.device)
        s2 = s2.to(self.device)
        a2 = a2.to(self.device)
        recon_c, recon_s, mean, std = self.vae(s1, a_, a2)  #reconc是2维的，k与xk。recons是预测的δ
        # print("recon_s",recon_s.shape)
        z = mean + std * torch.randn_like(std)  #mean是6维，即d2维。torch.randn_like(std) 是一个与std维度相同的随机数，从N(0,1)取样
        #mean的个数是从采样池中抽的，则z是batch_size行、6列
        z = z.cpu().data.numpy()
        c_rate = self.z_range(z, batch_size, range_rate)
        # print("s2",s2.shape)

        recon_s_loss = F.mse_loss(recon_s, s2, size_average=True)
        #计算预测的δ与真实的δ的MSE损失LDYN，返回所有样本的平均值。设置size_average=False则是返回所有样本的损失值总和

        # recon_s = abs(np.mean(recon_s.cpu().data.numpy()))
        return c_rate, recon_s_loss.detach().cpu().numpy()

    def z_range(self, z, batch_size=100, range_rate=5):     #选定bupper与blower
        #在hard goal里，z是batch_size行、6列。假设为256行，六列。则应该去len(z[0]) == 6:
        #z1到z6保存z的每一列的数，然后对他们排序，选择每一列的上下界，最后把每一列的上下界返回

        self.z1, self.z2, self.z3, self.z4, self.z5, self.z6, self.z7, self.z8, self.z9,\
        self.z10,self.z11,self.z12,self.z13,self.z14,self.z15,self.z16 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        border = int(range_rate * (batch_size / 100))

        # print("border",border)
        if len(z[0]) == 16:
            for i in range(len(z)):
                self.z1.append(z[i][0])
                self.z2.append(z[i][1])
                self.z3.append(z[i][2])
                self.z4.append(z[i][3])
                self.z5.append(z[i][4])
                self.z6.append(z[i][5])
                self.z7.append(z[i][6])
                self.z8.append(z[i][7])
                self.z9.append(z[i][8])
                self.z10.append(z[i][9])
                self.z11.append(z[i][10])
                self.z12.append(z[i][11])
                self.z13.append(z[i][12])
                self.z14.append(z[i][13])
                self.z15.append(z[i][14])
                self.z16.append(z[i][15])

        if len(z[0]) == 16:
            self.z1.sort(), self.z2.sort(), self.z3.sort(), self.z4.sort(), self.z5.sort(), self.z6.sort(), self.z7.sort(), self.z8.sort(), \
            self.z9.sort(), self.z10.sort(), self.z11.sort(), self.z12.sort(),self.z13.sort(), self.z14.sort(), self.z15.sort(), self.z16.sort()
            c_rate_1_up = self.z1[-border - 1]
            c_rate_1_down = self.z1[border]
            c_rate_2_up = self.z2[-border - 1]
            c_rate_2_down = self.z2[border]
            c_rate_3_up = self.z3[-border - 1]
            c_rate_3_down = self.z3[border]
            c_rate_4_up = self.z4[-border - 1]
            c_rate_4_down = self.z4[border]
            c_rate_5_up = self.z5[-border - 1]
            c_rate_5_down = self.z5[border]
            c_rate_6_up = self.z6[-border - 1]
            c_rate_6_down = self.z6[border]
            c_rate_7_up = self.z7[-border - 1]
            c_rate_7_down = self.z7[border]
            c_rate_8_up = self.z8[-border - 1]
            c_rate_8_down = self.z8[border]
            c_rate_9_up = self.z9[-border - 1]
            c_rate_9_down = self.z9[border]
            c_rate_10_up = self.z10[-border - 1]
            c_rate_10_down = self.z10[border]
            c_rate_11_up = self.z11[-border - 1]
            c_rate_11_down = self.z11[border]
            c_rate_12_up = self.z12[-border - 1]
            c_rate_12_down = self.z12[border]
            c_rate_13_up = self.z13[-border - 1]
            c_rate_13_down = self.z13[border]
            c_rate_14_up = self.z14[-border - 1]
            c_rate_14_down = self.z14[border]
            c_rate_15_up = self.z15[-border - 1]
            c_rate_15_down = self.z15[border]
            c_rate_16_up = self.z16[-border - 1]
            c_rate_16_down = self.z16[border]

            c_rate_1, c_rate_2, c_rate_3, c_rate_4, c_rate_5, c_rate_6, c_rate_7, c_rate_8, \
            c_rate_9, c_rate_10, c_rate_11, c_rate_12, c_rate_13, c_rate_14, c_rate_15, c_rate_16 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
            c_rate_1.append(c_rate_1_up), c_rate_1.append(c_rate_1_down)
            c_rate_2.append(c_rate_2_up), c_rate_2.append(c_rate_2_down)
            c_rate_3.append(c_rate_3_up), c_rate_3.append(c_rate_3_down)
            c_rate_4.append(c_rate_4_up), c_rate_4.append(c_rate_4_down)
            c_rate_5.append(c_rate_5_up), c_rate_5.append(c_rate_5_down)
            c_rate_6.append(c_rate_6_up), c_rate_6.append(c_rate_6_down)
            c_rate_7.append(c_rate_7_up), c_rate_7.append(c_rate_7_down)
            c_rate_8.append(c_rate_8_up), c_rate_8.append(c_rate_8_down)
            c_rate_9.append(c_rate_9_up), c_rate_9.append(c_rate_9_down)
            c_rate_10.append(c_rate_10_up), c_rate_10.append(c_rate_10_down)
            c_rate_11.append(c_rate_11_up), c_rate_11.append(c_rate_11_down)
            c_rate_12.append(c_rate_12_up), c_rate_12.append(c_rate_12_down)
            c_rate_13.append(c_rate_13_up), c_rate_13.append(c_rate_13_down)
            c_rate_14.append(c_rate_14_up), c_rate_14.append(c_rate_14_down)
            c_rate_15.append(c_rate_15_up), c_rate_15.append(c_rate_15_down)
            c_rate_16.append(c_rate_16_up), c_rate_16.append(c_rate_16_down)

            return c_rate_1, c_rate_2, c_rate_3, c_rate_4, c_rate_5, c_rate_6, c_rate_7, c_rate_8,\
                   c_rate_9, c_rate_10, c_rate_11, c_rate_12,c_rate_13, c_rate_14, c_rate_15, c_rate_16

        if len(z[0]) == 12:
            for i in range(len(z)):
                self.z1.append(z[i][0])
                self.z2.append(z[i][1])
                self.z3.append(z[i][2])
                self.z4.append(z[i][3])
                self.z5.append(z[i][4])
                self.z6.append(z[i][5])
                self.z7.append(z[i][6])
                self.z8.append(z[i][7])
                self.z9.append(z[i][8])
                self.z10.append(z[i][9])
                self.z11.append(z[i][10])
                self.z12.append(z[i][11])

        if len(z[0]) == 12:
            self.z1.sort(), self.z2.sort(), self.z3.sort(), self.z4.sort(), self.z5.sort(), self.z6.sort(), self.z7.sort(), self.z8.sort(), \
            self.z9.sort(), self.z10.sort(), self.z11.sort(), self.z12.sort()
            c_rate_1_up = self.z1[-border - 1]
            c_rate_1_down = self.z1[border]
            c_rate_2_up = self.z2[-border - 1]
            c_rate_2_down = self.z2[border]
            c_rate_3_up = self.z3[-border - 1]
            c_rate_3_down = self.z3[border]
            c_rate_4_up = self.z4[-border - 1]
            c_rate_4_down = self.z4[border]
            c_rate_5_up = self.z5[-border - 1]
            c_rate_5_down = self.z5[border]
            c_rate_6_up = self.z6[-border - 1]
            c_rate_6_down = self.z6[border]
            c_rate_7_up = self.z7[-border - 1]
            c_rate_7_down = self.z7[border]
            c_rate_8_up = self.z8[-border - 1]
            c_rate_8_down = self.z8[border]
            c_rate_9_up = self.z9[-border - 1]
            c_rate_9_down = self.z9[border]
            c_rate_10_up = self.z10[-border - 1]
            c_rate_10_down = self.z10[border]
            c_rate_11_up = self.z11[-border - 1]
            c_rate_11_down = self.z11[border]
            c_rate_12_up = self.z12[-border - 1]
            c_rate_12_down = self.z12[border]
            c_rate_1, c_rate_2, c_rate_3, c_rate_4, c_rate_5, c_rate_6, c_rate_7, c_rate_8, c_rate_9, c_rate_10, c_rate_11, c_rate_12 = [], [], [], [], [], [], [], [], [], [], [], []
            c_rate_1.append(c_rate_1_up), c_rate_1.append(c_rate_1_down)
            c_rate_2.append(c_rate_2_up), c_rate_2.append(c_rate_2_down)
            c_rate_3.append(c_rate_3_up), c_rate_3.append(c_rate_3_down)
            c_rate_4.append(c_rate_4_up), c_rate_4.append(c_rate_4_down)
            c_rate_5.append(c_rate_5_up), c_rate_5.append(c_rate_5_down)
            c_rate_6.append(c_rate_6_up), c_rate_6.append(c_rate_6_down)
            c_rate_7.append(c_rate_7_up), c_rate_7.append(c_rate_7_down)
            c_rate_8.append(c_rate_8_up), c_rate_8.append(c_rate_8_down)
            c_rate_9.append(c_rate_9_up), c_rate_9.append(c_rate_9_down)
            c_rate_10.append(c_rate_10_up), c_rate_10.append(c_rate_10_down)
            c_rate_11.append(c_rate_11_up), c_rate_11.append(c_rate_11_down)
            c_rate_12.append(c_rate_12_up), c_rate_12.append(c_rate_12_down)
            return c_rate_1, c_rate_2, c_rate_3, c_rate_4, c_rate_5, c_rate_6, c_rate_7, c_rate_8, c_rate_9, c_rate_10, c_rate_11, c_rate_12

        if len(z[0]) == 10:
            for i in range(len(z)):
                self.z1.append(z[i][0])
                self.z2.append(z[i][1])
                self.z3.append(z[i][2])
                self.z4.append(z[i][3])
                self.z5.append(z[i][4])
                self.z6.append(z[i][5])
                self.z7.append(z[i][6])
                self.z8.append(z[i][7])
                self.z9.append(z[i][8])
                self.z10.append(z[i][9])

        if len(z[0]) == 10:
            self.z1.sort(), self.z2.sort(), self.z3.sort(), self.z4.sort(), self.z5.sort(), self.z6.sort(), self.z7.sort(), self.z8.sort(), self.z9.sort(), self.z10.sort()
            c_rate_1_up = self.z1[-border - 1]
            c_rate_1_down = self.z1[border]
            c_rate_2_up = self.z2[-border - 1]
            c_rate_2_down = self.z2[border]
            c_rate_3_up = self.z3[-border - 1]
            c_rate_3_down = self.z3[border]
            c_rate_4_up = self.z4[-border - 1]
            c_rate_4_down = self.z4[border]
            c_rate_5_up = self.z5[-border - 1]
            c_rate_5_down = self.z5[border]
            c_rate_6_up = self.z6[-border - 1]
            c_rate_6_down = self.z6[border]
            c_rate_7_up = self.z7[-border - 1]
            c_rate_7_down = self.z7[border]
            c_rate_8_up = self.z8[-border - 1]
            c_rate_8_down = self.z8[border]
            c_rate_9_up = self.z9[-border - 1]
            c_rate_9_down = self.z9[border]
            c_rate_10_up = self.z10[-border - 1]
            c_rate_10_down = self.z10[border]
            c_rate_1, c_rate_2, c_rate_3, c_rate_4, c_rate_5, c_rate_6, c_rate_7, c_rate_8, c_rate_9, c_rate_10 = [], [], [], [], [], [], [], [], [], []
            c_rate_1.append(c_rate_1_up), c_rate_1.append(c_rate_1_down)
            c_rate_2.append(c_rate_2_up), c_rate_2.append(c_rate_2_down)
            c_rate_3.append(c_rate_3_up), c_rate_3.append(c_rate_3_down)
            c_rate_4.append(c_rate_4_up), c_rate_4.append(c_rate_4_down)
            c_rate_5.append(c_rate_5_up), c_rate_5.append(c_rate_5_down)
            c_rate_6.append(c_rate_6_up), c_rate_6.append(c_rate_6_down)
            c_rate_7.append(c_rate_7_up), c_rate_7.append(c_rate_7_down)
            c_rate_8.append(c_rate_8_up), c_rate_8.append(c_rate_8_down)
            c_rate_9.append(c_rate_9_up), c_rate_9.append(c_rate_9_down)
            c_rate_10.append(c_rate_10_up), c_rate_10.append(c_rate_10_down)
            return c_rate_1, c_rate_2, c_rate_3, c_rate_4, c_rate_5, c_rate_6, c_rate_7, c_rate_8, c_rate_9, c_rate_10

        if len(z[0]) == 8:
            for i in range(len(z)):
                self.z1.append(z[i][0])
                self.z2.append(z[i][1])
                self.z3.append(z[i][2])
                self.z4.append(z[i][3])
                self.z5.append(z[i][4])
                self.z6.append(z[i][5])
                self.z7.append(z[i][6])
                self.z8.append(z[i][7])

        if len(z[0]) == 8:
            self.z1.sort(), self.z2.sort(), self.z3.sort(), self.z4.sort(), self.z5.sort(), self.z6.sort(), self.z7.sort(), self.z8.sort()
            c_rate_1_up = self.z1[-border - 1]
            c_rate_1_down = self.z1[border]
            c_rate_2_up = self.z2[-border - 1]
            c_rate_2_down = self.z2[border]
            c_rate_3_up = self.z3[-border - 1]
            c_rate_3_down = self.z3[border]
            c_rate_4_up = self.z4[-border - 1]
            c_rate_4_down = self.z4[border]
            c_rate_5_up = self.z5[-border - 1]
            c_rate_5_down = self.z5[border]
            c_rate_6_up = self.z6[-border - 1]
            c_rate_6_down = self.z6[border]
            c_rate_7_up = self.z7[-border - 1]
            c_rate_7_down = self.z7[border]
            c_rate_8_up = self.z8[-border - 1]
            c_rate_8_down = self.z8[border]
            c_rate_1, c_rate_2, c_rate_3, c_rate_4, c_rate_5, c_rate_6, c_rate_7, c_rate_8 = [], [], [], [], [], [], [], []
            c_rate_1.append(c_rate_1_up), c_rate_1.append(c_rate_1_down)
            c_rate_2.append(c_rate_2_up), c_rate_2.append(c_rate_2_down)
            c_rate_3.append(c_rate_3_up), c_rate_3.append(c_rate_3_down)
            c_rate_4.append(c_rate_4_up), c_rate_4.append(c_rate_4_down)
            c_rate_5.append(c_rate_5_up), c_rate_5.append(c_rate_5_down)
            c_rate_6.append(c_rate_6_up), c_rate_6.append(c_rate_6_down)
            c_rate_7.append(c_rate_7_up), c_rate_7.append(c_rate_7_down)
            c_rate_8.append(c_rate_8_up), c_rate_8.append(c_rate_8_down)
            return c_rate_1, c_rate_2, c_rate_3, c_rate_4, c_rate_5, c_rate_6, c_rate_7, c_rate_8

        if len(z[0]) == 6:
            for i in range(len(z)):
                self.z1.append(z[i][0])
                self.z2.append(z[i][1])
                self.z3.append(z[i][2])
                self.z4.append(z[i][3])
                self.z5.append(z[i][4])
                self.z6.append(z[i][5])

        if len(z[0]) == 6:
            self.z1.sort(), self.z2.sort(), self.z3.sort(), self.z4.sort(), self.z5.sort(), self.z6.sort()
            c_rate_1_up = self.z1[-border - 1]
            c_rate_1_down = self.z1[border]
            c_rate_2_up = self.z2[-border - 1]
            c_rate_2_down = self.z2[border]
            c_rate_3_up = self.z3[-border - 1]
            c_rate_3_down = self.z3[border]
            c_rate_4_up = self.z4[-border - 1]
            c_rate_4_down = self.z4[border]
            c_rate_5_up = self.z5[-border - 1]
            c_rate_5_down = self.z5[border]
            c_rate_6_up = self.z6[-border - 1]
            c_rate_6_down = self.z6[border]

            c_rate_1, c_rate_2, c_rate_3, c_rate_4, c_rate_5, c_rate_6 = [], [], [], [], [], []
            c_rate_1.append(c_rate_1_up), c_rate_1.append(c_rate_1_down)
            c_rate_2.append(c_rate_2_up), c_rate_2.append(c_rate_2_down)
            c_rate_3.append(c_rate_3_up), c_rate_3.append(c_rate_3_down)
            c_rate_4.append(c_rate_4_up), c_rate_4.append(c_rate_4_down)
            c_rate_5.append(c_rate_5_up), c_rate_5.append(c_rate_5_down)
            c_rate_6.append(c_rate_6_up), c_rate_6.append(c_rate_6_down)

            return c_rate_1, c_rate_2, c_rate_3, c_rate_4, c_rate_5, c_rate_6

        if len(z[0]) == 4:
            for i in range(len(z)):
                self.z1.append(z[i][0])
                self.z2.append(z[i][1])
                self.z3.append(z[i][2])
                self.z4.append(z[i][3])

        if len(z[0]) == 4:
            self.z1.sort(), self.z2.sort(), self.z3.sort(), self.z4.sort()
            # print("lenz1",len(self.z1),self.z1)
            c_rate_1_up = self.z1[-border - 1]
            c_rate_1_down = self.z1[border]
            c_rate_2_up = self.z2[-border - 1]
            c_rate_2_down = self.z2[border]
            c_rate_3_up = self.z3[-border - 1]
            c_rate_3_down = self.z3[border]
            c_rate_4_up = self.z4[-border - 1]
            c_rate_4_down = self.z4[border]

            c_rate_1, c_rate_2, c_rate_3, c_rate_4 = [], [], [], []
            c_rate_1.append(c_rate_1_up), c_rate_1.append(c_rate_1_down)
            c_rate_2.append(c_rate_2_up), c_rate_2.append(c_rate_2_down)
            c_rate_3.append(c_rate_3_up), c_rate_3.append(c_rate_3_down)
            c_rate_4.append(c_rate_4_up), c_rate_4.append(c_rate_4_down)

            return c_rate_1, c_rate_2, c_rate_3, c_rate_4

        if len(z[0]) == 3:
            for i in range(len(z)):
                self.z1.append(z[i][0])
                self.z2.append(z[i][1])
                self.z3.append(z[i][2])

        if len(z[0]) == 3:
            self.z1.sort(), self.z2.sort(), self.z3.sort()
            # print("lenz1",len(self.z1),self.z1)
            c_rate_1_up = self.z1[-border - 1]
            c_rate_1_down = self.z1[border]
            c_rate_2_up = self.z2[-border - 1]
            c_rate_2_down = self.z2[border]
            c_rate_3_up = self.z3[-border - 1]
            c_rate_3_down = self.z3[border]

            c_rate_1, c_rate_2, c_rate_3 = [], [], []
            c_rate_1.append(c_rate_1_up), c_rate_1.append(c_rate_1_down)
            c_rate_2.append(c_rate_2_up), c_rate_2.append(c_rate_2_down)
            c_rate_3.append(c_rate_3_up), c_rate_3.append(c_rate_3_down)

            return c_rate_1, c_rate_2, c_rate_3
