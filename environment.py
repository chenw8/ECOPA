import csv
import math
from math import *
import numpy as np
import random
import pandas as pd
import torch

import agents

#ik,

#边缘服务器S=3，用户U=10个

# class User:
#     def __init__(self, user_id, input_size, output_size, cpu_cycles, delay_constraint, priority,t):
#         self.user_id = user_id
#         self.input_size = input_size  # 数据输入大小 din
#         self.output_size = output_size  # 数据输出大小 dou
#         self.cpu_cycles = cpu_cycles  # 任务计算需求 cm
#         self.delay_constraint = delay_constraint  # 延迟约束 lm
#         self.priority = priority  # 任务优先级 pm
#         self.t=t    #时隙

class EdgeServer:
    def __init__(self, server_id, max_capacity, power_coefficient):
        self.server_id = server_id
        self.max_capacity = max_capacity  # 最大计算能力
        self.allocated_capacity = 0  # 当前已分配计算能力
        self.power_coefficient = power_coefficient  # 功率系数

    def allocate_resources(self, cpu_cycles):
        if self.allocated_capacity + cpu_cycles <= self.max_capacity:
            self.allocated_capacity += cpu_cycles
            return True
        return False

    def release_resources(self, cpu_cycles):  #释放cpu_cycles的计算资源
        self.allocated_capacity = max(0, self.allocated_capacity - cpu_cycles)

def read_csv_to_array(filename):  #读取csv文件
    with open(filename, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        data = list(reader)  # 将CSV内容转换为列表
    return data


class Environment:
    def __init__(self,ES_num,user_num,policy):
        self.ES_num = ES_num
        self.user_num = user_num
        # self.limit = limit    #时间限制
        # self.users = users  # 用户任务列表
        # self.servers = servers  # 边缘服务器列表
        self.current_time =0
        self.max_time = 100  # 最大时间步，避免无限运行。好像没用
        self.state=[]
        self.servers=read_csv_to_array('edge_servers.csv')  #初始化服务器数据
        self.tasks=read_csv_to_array('tasks.csv')
        self.loc_ES=[[6000,11000],[10000,2000],[2000,2000]]  #三服务器的初始位置，x,y长度均为[-5000,17000]
        self.ES=[]
        self.t=0  #时隙
        self.Gmax=10  #14(a)的约束。暂定10
        self.Rbh=[[0.,8.7,6.0],      #MB/
                  [5.5,0.,4.1],
                  [3.4,7.2,0.]]
        self.favg=88e+9   #平均计算能力 - Hz 从48，64，96，144中取的中间值
        self.delta=40e+9    #计算能力方差
        self.power_coeff=1e-27     #服务器功率系数，用于计算E
        self.TD3=policy


        #x和y都设到-5000到17000


    def reset(self):
        self.t=1
        task=self.read_s0()                                         #读取最开始的任务信息
        user=self.read_s0_trace()                                   #用户初始位置数组
        #还有初始时临近es信息和预测的未来临近es信息
        um0=self.get_s0_u(self.loc_ES,user)   #self.loc_ES,user是服务器和用户的位置数组。如[[1,2],[1,3],[2,4]]。计算邻近服务器
        Rup=self.get_Rup(4e+7,0.8,0.8e-10,user)
        Tm=self.get_Tm(task,Rup,self.favg,self.delta) # 计算每个任务的处理时间(Tm)，并加上当前时隙得到预计完成时间
        #print(Tm)  #[1, 8, 12, 14, 17, 5, 11, 12, 8, 8]
        for i in range(len(Tm)):
            Tm[i]+=self.t

        user_=self.read_trace(Tm)  #用户在Tm时刻的位置数组。初始状态的预测位置直接读取数据集。之后的使用网络预测
        um_=self.get_s0_u(self.loc_ES,user_)      #传入服务器位置和t+Tm时刻用户位置数组
        task=np.array(task)
        um0=np.array(um0)
        um_=np.array(um_)
        self.state=np.concatenate([task.ravel(),um0,um_]) #task.ravel()是把task变为一维；将任务信息、当前位置关系和预测位置关系组合成状态向量
        #把w也更新进去
        #reset时，创建三个ES服务器，并初始化
        self.ES=self.initial_ES()

        return self.state

    def step(self,action):  #action的形式是([1,3,2……](10个1-3的数),[138803,](30个数，0-9表ES1给每个用户的资源，10-19表示ES2给用户))
        #step里要不要用一个t表示时刻？？ - 有self的时刻，看看参考一下
        #要不要在分配资源的时候减去资源，等到完成时隙再释放资源？？

        """----------时间需要改成整数----------"""

        self.state = np.array(self.state, dtype=np.float32, copy=False)

        tau = self.state[0:70].reshape(10, 7)  # 任务信息，转为10*7的二维数组
        um = self.state[70:80]  # 取值是1，2，3而不是0，1，2
        um_ = self.state[80:]

        cur_t=self.t  #保存一下当前时刻
        lamb = action[1].reshape(3,10)  # lamb是每个ES给每个用户的资源
        sigma = action[0]  # sigma是每个用户选择的ES编号。取值为1，2，3
        w=0   #是否满足约束
        for i in range(len(lamb)):     #判断分配资源是否超过最大能力。如果超过，按比例重新分配并且把w设为1
            sum=0
            for j in range(len(lamb[i])):
                # lamb[i][j]/=10
                sum+=lamb[i][j]
            # sum/=10
            x=self.ES[i].max_capacity#/10
            if sum>x: #如果资源分配超过最大计算能力
                w+=1   #w=1
                #按比例修改资源分配
                lamb[i]=self.allo_res(lamb,i,sum)   #修改lamb某行(某个ES)的资源分配
        t=[]
        for i in range(self.user_num):
            t.append(cur_t)
        user=self.read_trace(t)   #用户此刻的位置

        Tfn,Tc= self.cal_T(action,user)   #重新分配资源后再计算时间
        """----分配的资源是负数，则Tc算出来是负数"""

        #计算Tso
        #查找Tfn+cur_t时用户的位置
        t_com=[]   #每个任务的完成时间
        for i in range(len(Tfn)):
            x=cur_t+Tfn[i]
            x=math.floor(x)   #变为整数找位置
            t_com.append(x)
        #根据t_com查找用户的位置(需要用户移动的数据集)。得到u_m_

        user_=self.read_trace(t_com)   #每个用户在任务完成时的位置数组
        u_m_=self.get_s0_u(self.loc_ES,user_)
        Tso=self.calc_Tso(sigma,tau,self.Rbh,u_m_)
        #计算移动感知增益
        k=self.move_aware_utility(Tso,tau)
        if k>self.Gmax:
            w+=1  #w=1

        #计算U和E
        U=self.calc_U(Tfn)
        lamb_detach=torch.from_numpy(lamb).clone()  #在计算E时会把lamb每项都扩大1e7，而lamb是根据action得到的，也会让action扩大
        E=self.calc_E(self.power_coeff,lamb_detach,sigma,Tc)  #power_coefficient是每个ES功率数组
        U_=0
        E_=0
        for i in range(len(U)):
            U_+=U[i]
            E_+=E[i]
        miu=U_/E_
        #print("miu",miu)  #miu从0到200不等
        """----------------------------------Δp修改----------------------------------"""
        delta_p=10  #Δp暂定10
        reward=miu-delta_p*w+5  #计算奖励

        if cur_t<100:
            done=False
        else:
            done=True


        #更新状态
        if not done:
            self.t+=1
            
            
        task = self.read_s(self.t)   #读取下一时刻的任务信息
        now_t=[]
        for i in range(self.user_num):
            now_t.append(self.t)
        user_next = self.read_trace(now_t) # 用户到下一位置位置数组

        um0 = self.get_s0_u(self.loc_ES, user_next)  # servers,task是服务器和用户的位置数组。如[[1,2],[1,3],[2,4]]
        Rup=self.get_Rup(4e+7,0.8,0.8e-10,user_next)
        Tm = self.get_Tm(task,Rup,self.favg,self.delta)  #预测的每个任务完成时间，是一个数组
        t_next_com=[]
        for i in range(len(Tm)):
            t_next_com.append(Tm[i]+self.t)    #预估每个任务的完成时间

        #预测使用当前时刻t，要估计的时间以及用户当前位置
        user_pre =self.TD3.loc_pre(now_t,t_next_com,user_next,task) # 用户在预测在t+Tm时刻的位置数组。使用神经网络预测位置。需要输入tau，得到用户编号，选择网络
        #user_pre为10维向量，表示预测的位置
        um_ = self.get_s0_u(self.loc_ES, user_pre)  # 传入服务器位置和t+Tm时刻用户位置数组
        task = np.array(task)
        um0 = np.array(um0)
        um_ = np.array(um_)
        self.state = np.concatenate([task.ravel(), um0, um_])  # task.ravel()是把task变为一维
        return self.state,reward,done,t_com,user_pre,miu

    def initial_ES(self):       #初始化ES
        df = pd.read_csv("edge_servers.csv")
        ES=[EdgeServer(row[0], row[1], row[2]) for row in df.itertuples(index=False, name=None)] #读取csv的内容创建ES
        return ES

    def read_s0(self):               #读取初始状态
        task=read_csv_to_array('tasks.csv')
        tau=[]
        for row in task[1:len(task)]:
            if row[1]=='1':
                tau.append(row)
        return tau
    def read_s(self,t):               #传入时刻t，读取状态
        task=read_csv_to_array('tasks.csv')
        tau=[]
        for row in task[1:len(task)]:
            if int(row[1])==t:
                tau.append(row)
        return tau

    def read_s0_trace(self):    #读取初始状态的位置
        location=[]
        for i in range(self.user_num):
            if i<9:
                trace=read_csv_to_array('user_trace/NewYork_30sec_00{}.csv'.format(i+1))
            if i==9:
                trace = read_csv_to_array('user_trace/NewYork_30sec_010.csv')
            x=[]
            x.append(float(trace[1][1]))
            x.append(float(trace[1][2]))
            location.append(x)
        return location

    def read_trace(self,t):    #读取t时刻的位置，t是所有用户各自的预测的t+Tm
        location=[]
        for i in range(self.user_num):
            if i<9:
                trace=read_csv_to_array('user_trace/NewYork_30sec_00{}.csv'.format(i+1))
            if i==9:
                trace = read_csv_to_array('user_trace/NewYork_30sec_010.csv')
            x=[]
            if t[i]>=100:
                x.append(float(trace[-1][1]))
                x.append(float(trace[-1][2]))
            else:
                x.append(float(trace[t[i]][1]))
                x.append(float(trace[t[i]][2]))
            location.append(x)
        return location


    def get_s0_u(self,ES_loca,user_loca):  #根据每个服务器初始位置数组和每个用户的位置，计算每个用户的临近服务器。返回数组um
        #ES_loca时二维数组如[[1,2],[1,3],[2,4]]。user_loca也是这样的二维数组。可以是初始位置也可以是某时刻位置
        um=[]
        for i in range(len(user_loca)):
            x=1  #选定第一个ES为临近
            L=sqrt((ES_loca[0][0]-user_loca[i][0])**2+(ES_loca[0][1]-user_loca[i][1])**2) #计算用户到第一个ES的距离
            for j in range(1,len(ES_loca)):
                l=sqrt((ES_loca[j][0]-user_loca[i][0])**2+(ES_loca[j][1]-user_loca[i][1])**2)
                if l<L:
                    L=l
                    x=j+1   #返回的服务器为1，2，3而不是0，1，2
            um.append(x)

        return um

    def get_Tm(self,task,Rup,favg,delta):    #预估Tm。task是所有用户某个t的任务信息，Rup是传输速率数组,favg,delta是系统参数
        Tm=[]
        #由于无法确定任务要分配到那个服务器，所以使用Rup的平均 -- 可以改为每个用户分别对三个ES的平均，而不是全部的平均。Rup为bit/s
        x=0
        for i in range(len(Rup)):
            for j in range(len(Rup[0])):
                x+=Rup[i][j]
        x=x/(len(Rup)*len(Rup[0]))
        x=x/8/1024/1024  #bit/s转换为MB/s

        y=0
        for i in range(len(self.Rbh)):
            for j in range(len(self.Rbh[0])):
                y+=self.Rbh[i][j]
        Ravg=y/(len(self.Rbh)*len(self.Rbh[0])-len(self.Rbh))   #Ravg为MB/s
        # print("x:",x)   # 0.27809878813601935
        # print("Ravg:",Ravg)  #58.166666666666664

        for i in range(len(task)):
            T=0
            if task[i][6]=='0':
                T=T+float(task[i][2])/x+(float(task[i][2])+float(task[i][3]))/Ravg+float(task[i][4])/(favg+delta)     #task[4]是c，看看用不用换单位
            else:
                T = T + float(task[i][2]) / x + (float(task[i][2]) + float(task[i][3])) / Ravg + float(task[i][4]) / (favg - delta)
            T=math.floor(T)   #T变为小于他的最大整数
            Tm.append(T)
        return Tm

    def uplink_rate(self,bandwidth, power, channel_gain, noise_power):  #计算Rup。power是um的，channel_gain是um与sn的
        return bandwidth * math.log2(1 + (power * channel_gain) / noise_power)

    def calc_Toff(self,tau,Rup,sigma):           #计算Toff。
        Toff=[]
        for i in range(len(Rup)):
            for j in range(len(Rup[0])):
                Rup[i][j]=Rup[i][j]/8/1024/1024*20  #bit/s转换为MB/s（扩大20倍）
        for i in range(len(tau)): #10个用户
            T=float(tau[i][2])/Rup[i][sigma[i]-1]  #sigma[1]得到是[1]
            Toff.append(T)
        return Toff

    def calc_Ttr(self,tau,Rbh,sigma,um):         #计算Ttr。
        #如果执行ES与临近ES不是一个，才会有Ttr。否则，Ttr=0  。sigma是要做的决策？
        Ttr=[]
        for i in range(len(tau)):  # 10个用户
            if int(sigma[i])==int(um[i]):
                Ttr.append(0)
            else:
                din=float(tau[i][2])
                
                # print("s:", len(Rbh[0]))
                # print(len(Rbh[1]))
                # print(sigma[i]-1)
                # print(int(um[i])-1)
                
                
                T=din/Rbh[sigma[i]-1][int(um[i])-1]
                Ttr.append(T)
        return Ttr

    def calc_Tc(self,tau,lamb,sigma):         #计算Tc。
        Tc=[]
        lamb=lamb.reshape(3,10)  #转为3*10的数组。lamb[0][3]表示ES1给用户4的资源
        for i in range(len(tau)):  # 10个用户
            T=tau[i][4]/lamb[sigma[i]-1][i]
            Tc.append(T)
        return Tc

    # def compute_computation_delay(self,cpu_cycles, allocated_capacity):  #计算Tc
    #     if allocated_capacity == 0:
    #         return float('inf')  # 无计算资源
    #     return cpu_cycles / allocated_capacity

    def calc_E(self,power_coeff,lamb,sigma,Tc):               #计算能耗E，power_coeff为服务器功率系数
        lamb = lamb.reshape(3, 10)
        for i in range(len(lamb)):
            for j in range(len(lamb[0])):
                lamb[i][j]*=1e7   #计算的时候把cyale扩大
        E=[]
        for i in range(len(Tc)):
            e=power_coeff*(lamb[sigma[i]-1][i] ** 3)*Tc[i]
            E.append(e)
        return E

    def calc_Tso(self,sigma,tau,Rbh,u_m_):       #Tso。sigma是动作决策的临近服务器，tau是任务信息。u_m_是用户在t+Tfn的临近服务器！
        #Rbh为对称阵
        Tso=[]
        for i in range(len(sigma)):
            if sigma[i]==u_m_[i]:
                Tso.append(0)
            else:
                tso=tau[i][3]/Rbh[sigma[i]-1][u_m_[i]-1]
                Tso.append(tso)
        return Tso

        

    def cal_T(self,action,user):                 #计算Tfn
        self.state = np.array(self.state, dtype=np.float32, copy=False)
        tau=self.state[0:70].reshape(10, 7)     #任务信息，转为10*7的二维数组
        um=self.state[70:80]   #取值是1，2，3而不是0，1，2
        um_=self.state[80:]
        sigma=action[0]  #sigma是每个用户传到那个服务器。取值为1，2，3
        lamb=action[1]   #lamb是每个ES给每个用户的资源，现在是1*30的数组
        # print("lamb",lamb)
        Rup = self.get_Rup(4e+7,0.8,0.8e-10, user)
        Toff=self.calc_Toff(tau,Rup,sigma)   #假设每个用户和每个服务器的Rup是固定的，不随用户位置变化。Rup为每个用户和ES的Rup数组(10*3)
        Ttr=self.calc_Ttr(tau,self.Rbh,sigma,um)     #假设每个用户和每个服务器的Rbh是固定的，不随用户位置变化。
        Tc=self.calc_Tc(tau,lamb,sigma)
        Tfn=[]
        # print("Toff",Toff)
        # print("Ttr",Ttr)
        # print("Tc",Tc)
        for i in range(len(tau)):
            Tfn.append(Toff[i]+Ttr[i]+Tc[i])
        return Tfn,Tc

    def allo_res(self,lamb,i,sum):  #按比例分配资源。i是ES索引，取0、1、2。lamb是改为3*10的资源分配数组。sum是分配的资源总量
        max=self.ES[i].max_capacity
        lamb_=[]
        for j in range(len(lamb[i])):
            x=lamb[i][j]
            x=x/sum*max  #按x在sum中的比例根据max分配资源
            lamb_.append(x)
        return lamb_


    #优先级任务调用
    def high_priority_utility(self,delay_constraint,completion_time, cpu_cycles, penalty): #高。delay_constraint是l，completion_time是Tfn
        if completion_time <= delay_constraint:
            return math.log(1 + delay_constraint - completion_time) + math.log10(1+cpu_cycles)*0.3 #γ暂定0.3。cpu_cycles换单位
        return -penalty

    def low_priority_utility(self,delay_constraint, completion_time,cpu_cycles, alpha, beta):#低
        if completion_time <= delay_constraint:  #Tfn<l
            return 3 + alpha*math.log10(1+cpu_cycles)        #AL暂定为3
        return (3 + alpha*math.log10(1+cpu_cycles)) * math.exp(-beta * (completion_time - delay_constraint))

    #基于优先级的效用函数
    def calc_U(self,Tfn):           #Tfn是完成时间(用了多久，不是完成时刻)
        U=[]
        tau = self.state[0:70].reshape(10, 7)  # 任务信息，转为10*7的二维数组(10个用户)
        for i in range(len(tau)):
            u=0
            if tau[i][-1]==0:   #0是高优先级任务
                u=self.high_priority_utility(tau[i][-2],Tfn[i],tau[i][-3],10)  #惩罚暂定为10
            else:
                u=self.low_priority_utility(tau[i][-2],Tfn[i],tau[i][-3],0.7,0.5)  #a和b暂定为0.7，0.5
            U.append(u)
        return U

    #评价标准
    def move_aware_utility(self,tso,tau):  #每t的移动感知效益。注意d和Ravg的单位
        t=0
        t_=0

        y=0
        for i in range(len(self.Rbh)):
            for j in range(len(self.Rbh[0])):
                y+=self.Rbh[i][j]
        Ravg=y/(len(self.Rbh)*len(self.Rbh[0])-len(self.Rbh))

        for i in range(len(tso)):
            t+=tso[i]
            t_+=tau[i][3]/Ravg
        if t_<=0:
            t_=0.1
        return t/t_

    def get_Rup(self,B,pm,sig,user):    #Rup为10行3列数组,user是用户位置数组
        Rup=[]
        for i in range(self.user_num):
            x=[]
            user_loc=user[i]
            for j in range(self.ES_num):
                ES_loc=self.loc_ES[j]
                d=sqrt((user_loc[0]-ES_loc[0])**2+(user_loc[1]-ES_loc[1])**2)
                FSPL=20*math.log10(d)+20*math.log10(2.4e+09)-147.56
                hmn=10**(-FSPL/10)
                R=B*math.log2(1+pm*hmn/sig)
                x.append(R)
            Rup.append(x)
        return Rup
