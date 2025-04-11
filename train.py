import copy
import os
import gym
import gym_goal
import argparse
import time

import torch

import buffer
from agents import TD3
from embedding import ActionRepresentation_vae
from utils import plot_learning_curve, create_directory
from environment import Environment
from agents import pdqn

import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as torch_utils
from torch.autograd import Variable
from collections import deque

torch.random.manual_seed(41)
np.random.seed(41)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

envpath = '.'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath

parser = argparse.ArgumentParser()# 创建一个 ArgumentParser 对象
parser.add_argument('--ckpt_dir', type=str, default='./checkpoints/TD3/')# 添加参数并设置参数的全局默认值
parser.add_argument('--reward_path', type=str, default='./output_images/avg_reward.png')
# parser.add_argument('--epsilon_path', type=str, default='./output_images/epsilon.png')
parser.add_argument('--max_episodes', type=int, default=2000)

args = parser.parse_args()# 通过 parse_args() 方法解析参数。

# 将离散动作和连续参数合并成完整动作
def pad_action(act, act_param):
    return (act, act_param)

def evaluate(env, policy, action_rep, c_rate, episodes=100):
    returns = []
    epioside_steps = []
    system_gain=[]
    actor_hx = torch.zeros(1, 66).to(device)
    actor_cx = torch.zeros(1, 66).to(device)
    for _ in range(episodes):
        state = env.reset()
        # terminal = False
        t = 0
        total_reward = 0.
        episode_system_gain=0.
        for i in range(100):    ####需要修改成if 100步
            t += 1    #t是每一个episodes状态从s0开始到结束时所走的步数
            state = np.array(state, dtype=np.float32, copy=False)
            """-------------加入hx，cx"""
            discrete_emb, parameter_emb,actor_hx,actor_cx = policy.select_action(state,actor_hx, actor_cx)

            discrete_emb = (  # 为ei增加噪声
                    discrete_emb + np.random.normal(0, 1 * 0.1, size=60)
            ).clip(-1, 1)
            parameter_emb = (  # 为z增加噪声
                    parameter_emb + np.random.normal(0, 1 * 0.1, size=6)
            ).clip(-3, 3)
            """------------------------1,1-----------------"""

            true_parameter_emb = true_parameter_action(parameter_emb, c_rate)
            # print("parameter_emb",parameter_emb)  #[ 6.3583507e-15  9.2696886e-34 -1.0000000e+00  7.0859329e-36 -7.1407355e-29  1.0000000e+00]
            # print("true_parameter_emb",true_parameter_emb)  #[ -1.5793648   6.7578506 -53.840706   -4.9291916   8.109558   59.099037 ]
            # parameter_emb = parameter_emb * c_rate
            # select discrete action
            discrete_action_embedding = copy.deepcopy(discrete_emb)
            discrete_action_embedding = torch.from_numpy(discrete_action_embedding).float().reshape(1, -1)
            discrete_action = action_rep.select_discrete_action(discrete_action_embedding)
            """自己修改部分   -------   把用k得到的ei拼接起来成60维的ei,输入critic或vae的decode"""
            x = action_rep.get_embedding(discrete_action[0] - 1).to(device)
            for i in range(1, 10):
                y = action_rep.get_embedding(discrete_action[i] - 1).to(device)
                x = torch.cat((x, y), dim=0)
            a1 = x.reshape(1, -1).squeeze().long().to(device)
            """自己修改部分"""
            discrete_emb_1 = a1.cpu().view(-1).data.numpy()
            # discrete_emb_1 = action_rep.get_embedding(discrete_action).cpu().view(-1).data.numpy()  #根据选好的k，找到对应的ei

            all_parameter_action = action_rep.select_parameter_action(state, true_parameter_emb,
                                                                      # 输入s，z，ei，decode出原为两个数，现在是30维
                                                                      discrete_emb_1)  # state为90，true_parameter_emb为6，discrete_emb_1应为60
            # if discrete_action == 1 or discrete_action == 2:
            #     all_parameter_action = all_parameter_action[0]
            parameter_action = all_parameter_action
            for i in range(len(parameter_action)):  # 防止动作里有0
                if parameter_action[i] <= 0.:
                    parameter_action[i] = 10.
            action = pad_action(discrete_action, parameter_action)
            next_state, reward, terminal, t_com,pre_loc,miu = env.step(action)
            state=next_state
            # print("action:",action)
            total_reward += reward
            episode_system_gain+=miu
        print("episode:{}:total_reward".format(_),total_reward)

        epioside_steps.append(t)
        returns.append(total_reward)
        system_gain.append(episode_system_gain)
    print("---------------------------------------")
    print(
        f"Evaluation over {episodes} episodes: {np.array(returns[-100:]).mean():.3f} success: {(np.array(returns) == 50.).sum() / len(returns):.3f} epioside_steps: {np.array(epioside_steps[-100:]).mean():.3f}")
    print("---------------------------------------")
    return np.array(returns[-100:]).mean(),np.array(system_gain[-100:]).mean()



def run(args):

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    file_name = f"{args.policy}_"

    U=10  #用户数量
    S=3   #服务器数量
    state_dim=90
    discrete_action_dim=U  #离散动作维度是10，即U
    parameter_action_dim=30  #连续参数有30个
    discrete_emb_dim = 6
    parameter_emb_dim = 6
    max_action = 1.0

    env1 = gym.make('Goal-v0')

    kwargs = {
        "state_dim": state_dim,
        "discrete_action_dim": discrete_emb_dim,
        "parameter_action_dim": parameter_emb_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }
    kwargs["policy_noise"] = args.policy_noise * max_action
    kwargs["noise_clip"] = args.noise_clip * max_action
    kwargs["policy_freq"] = args.policy_freq
    policy = TD3.TD3(**kwargs)
    env = Environment(S, U,policy)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

        # embedding初始部分
    action_rep = ActionRepresentation_vae.Action_representation(state_dim=state_dim,
                                                                    action_dim=discrete_action_dim,  #10
                                                                    parameter_action_dim=30,   #decode的输出的动作维度。原为2，表示选定的连续动作。这里是30
                                                                    # encoder输入的xk和decoder输出的xk是parameter_action_dim个维度
                                                                    reduced_action_dim=discrete_emb_dim,  #ei 6
                                                                    reduce_parameter_action_dim=parameter_emb_dim  #z 6
                                                                    )


    # 双重回放缓冲区：
    #     replay_buffer：用于TD3策略学习
    #     replay_buffer_embedding：用于动作表征学习（VAE）

    replay_buffer = buffer.ReplayBuffer(state_dim, discrete_action_dim=10,  #存储离散k，原为1。这里是U，即10
                                           parameter_action_dim=30,  #原为2
                                           all_parameter_action_dim=parameter_action_dim, #30 - 貌似在sample中没啥用
                                           discrete_emb_dim=discrete_emb_dim, #6
                                           parameter_emb_dim=parameter_emb_dim, #6
                                           max_size=int(1e5))

    replay_buffer_embedding = buffer.ReplayBuffer(state_dim, discrete_action_dim=10,
                                                     parameter_action_dim=30,
                                                     all_parameter_action_dim=parameter_action_dim,
                                                     discrete_emb_dim=discrete_emb_dim,
                                                     parameter_emb_dim=parameter_emb_dim,
                                                     # max_size=int(2e7)
                                                     max_size=int(2e6)
                                                     )
    
    # 初始化了一个强化学习智能体，同时学习离散动作和它们的连续参数，使用经验回放(10,000的缓冲区大小)，为离散动作和参数设置了不同的学习率；
    agent_pre = pdqn.PDQNAgent(
        env1.observation_space.spaces[0], env1.action_space,    #####修改######
        batch_size=128,
        learning_rate_actor=0.001,
        learning_rate_actor_param=0.0001,
        epsilon_steps=1000,
        gamma=0.9,
        tau_actor=0.1,
        tau_actor_param=0.01,
        clip_grad=10.,
        indexed=False,
        weighted=False,
        average=False,
        random_weighted=False,
        initial_memory_threshold=500,
        use_ornstein_noise=False,
        replay_memory_size=10000,
        epsilon_final=0.01,
        inverting_gradients=True,
        zero_index_gradients=False,
        seed=args.seed)

    # ------Use random strategies to collect experience------
    # 使用随机策略收集经验数据，用于后续训练；
    max_steps = 100
    total_reward = 0.
    returns = []
    for i in range(30): #原为20000。改为100。改为50
        state = env.reset()
        state = np.array(state, dtype=np.float32, copy=False)
        act, act_param, all_action_parameters = agent_pre.act(state)  #all_action_parameters用不到
        action = pad_action(act, act_param)

        """----------------action的连续部分还是负的，看看怎么改-------------"""

        # print("action:",action)
        episode_reward = 0.
        agent_pre.start_episode()
        for j in range(max_steps):
            ret = env.step(action)
            next_state, reward, terminal, t_com,pre_loc,miu = ret #t_com完成时间，pre_loc预测完成时间的位置
            next_state = np.array(next_state, dtype=np.float32, copy=False)
            next_act, next_act_param, next_all_action_parameters = agent_pre.act(next_state)
            next_action = pad_action(next_act, next_act_param)
            state_next_state = next_state - state
            # if act == 1 or act == 2:
            #     act_param = np.append(act_param, 0.)
            replay_buffer_embedding.add(state, act, act_param, all_action_parameters, discrete_emb=None,
                                        parameter_emb=None,
                                        next_state=next_state,
                                        state_next_state=state_next_state,
                                        reward=reward, done=terminal,t_com=t_com,pre_loc=pre_loc,
                                        actor_hx=None, actor_cx=None,miu=miu)
            act, act_param, all_action_parameters = next_act, next_act_param, next_all_action_parameters
            action = next_action
            state = next_state
            episode_reward += reward
            if terminal:
                break
        # agent_pre.end_episode()
        returns.append(episode_reward)
        total_reward += episode_reward
        if i % 10 == 0:
            print('per-train-{0:5s} R:{1:.4f} r100:{2:.4f}'.format(str(i), total_reward / (i + 1),
                                                                   np.array(returns[-100:]).mean()))
    s_dir = "result/goal_model"
    save_dir = os.path.join(s_dir, "{}".format(str("embedding")))   # 为embedding模型(VAE动作表示)创建保存路径
    save_dir_rl = os.path.join(s_dir, "{}".format(str("policy")))   # 为policy模型(PDQN策略)创建保存路径
    print("save_dir", save_dir)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir_rl, exist_ok=True)
    rl_save_model=True
    rl_load_model=False

    # rl_save_model=False
    # rl_load_model=True
    # ------VAE训练------  预测训练部分-----    # 将高维连续参数压缩到低维潜在空间
    VAE_batch_size = 64
    vae_save_model = True # 加载之前训练的模型
    vae_load_model = False

    # vae_save_model = False
    # vae_load_model = True

    if vae_load_model:
        print("embedding load model")
        title = "vae" + "{}".format(str(4900))  #4900是2.23 17点训练好的(在7000的基础上训练了5000之后的)
        action_rep.load(title, save_dir)
        # print("load discrete embedding", action_rep.discrete_embedding())
    # print("pre VAE training phase started...")
    recon_s_loss = []
    c_rate, recon_s = vae_train(action_rep=action_rep, train_step=5000, replay_buffer=replay_buffer_embedding,#5000
                                batch_size=VAE_batch_size,
                                save_dir=save_dir, vae_save_model=vae_save_model, embed_lr=1e-4)

    # c_rate = np.load("result\goal_model\embedding\crate4900.npy")

    print("c_rate", c_rate)
    print("discrete embedding", action_rep.discrete_embedding())

    # -------TD3训练------
    print("TD3 train")
    total_reward = 0.   #存储的是所有episode的奖励和，while一次是一个episode，一个episode包含100步if
    total_system_gain = 0.
    returns = []        #存储的是每个episode的奖励(while一次的奖励)
    Reward = []         #每次eval的时候存储一个数，存的是现在的总奖励total_reward/t  。t为while次数
    System_gain = []
    Reward_100 = []     #存储的是returns中最后100次的平均值
    Test_Reward_100 = []    #eval100次，得到的奖励取平均值，存到这里
    Test_system_gain_100=[]
    # Test_epioside_step_100 = []
    # Test_success_rate_100 = []
    # Crate_all=[]
    # pre_loss=[]        #存储每个while的loss平均值(进行100步，然后loss取平均存储。存储while次)
    max_steps = 100
    cur_step = 0
    internal = 10
    total_timesteps = 0
    t = 0
    discrete_relable_rate, parameter_relable_rate = 0, 0

    actor_hx = None
    actor_cx = None
    actor_target_hx=None
    actor_target_cx=None

    # for t in range(int(args.max_episodes)):
    if rl_load_model:
        title = "td3" + "{}".format(str(100000))
        policy.load(title, save_dir_rl)
        print("rl load model")

    while t < args.max_timesteps:  #把max_timesteps设为2000。
        print("t", t, " / 2000")
        print("total_timesteps:", total_timesteps)

        if rl_save_model:
            # if i % 1000 == 0 and i >= 1000:
            if total_timesteps % 100 == 0 :
                title = "td3" + "{}".format(str(total_timesteps))
                policy.save(title, save_dir_rl)
                print("rl save model")
                title = "vae" + "{}".format(str(total_timesteps))
                action_rep.save(title, save_dir)
                print("embedding save model")
                c_rate=np.array(c_rate)
                print("c_rate",c_rate)
                np.save(os.path.join(save_dir, "crate" + "{}".format(str(total_timesteps) + ".npy")), c_rate)

                # ar_load = np.load(os.path.join(save_dir, "crate" + "{}".format(str(total_timesteps) + ".npy")))

        if actor_hx is not None:
            actor_hx.detach()
        else:
            actor_hx = torch.zeros(1,66).to(device)
        if actor_cx is not None:
            actor_cx.detach()
        else:
            actor_cx = torch.zeros(1,66).to(device)

        if actor_target_hx is not None:
            actor_target_hx.detach()
        else:
            actor_target_hx = torch.zeros(1,66).to(device)
        if actor_target_cx is not None:
            actor_target_cx.detach()
        else:
            actor_target_cx = torch.zeros(1,66).to(device)

        state = env.reset()
        state = np.array(state, dtype=np.float32, copy=False)
        
        # hx 是隐藏状态，cx 是细胞状态
        discrete_emb, parameter_emb,next_actor_hx,next_actor_cx = policy.select_action(state, actor_hx, actor_cx)  #根据s0，通过actor得到ei和z(ei是60维，z是6维)
        
        
        
        
        # 探索
        if t < args.epsilon_steps:  #修改epsilon
            epsilon = args.expl_noise_initial - (args.expl_noise_initial - args.expl_noise) * (
                    t / args.epsilon_steps)
        else:
            epsilon = args.expl_noise

        if rl_load_model:
            epsilon=0.0

        discrete_emb = (  #为ei增加噪声
                discrete_emb + np.random.normal(0, max_action * epsilon, size=discrete_emb_dim*U)
        ).clip(-max_action, max_action)
        parameter_emb = (  #为z增加噪声
                parameter_emb + np.random.normal(0, max_action * epsilon, size=parameter_emb_dim)
        ).clip(-3, 3)
        """-------------------------.clip(-max_action, max_action)----------------------"""
        true_parameter_emb = true_parameter_action(parameter_emb, c_rate)   #把z约束到[blower,bupper]
        # parameter_emb = parameter_emb * c_rate


        # select discrete action
        discrete_action_embedding = copy.deepcopy(discrete_emb)
        discrete_action_embedding = torch.from_numpy(discrete_action_embedding).float().reshape(1, -1) #ei转为一行
        discrete_action = action_rep.select_discrete_action(discrete_action_embedding)  #通过欧氏距离在嵌入表中找到合适的k(actor得到的ei输入到嵌入表)
        """自己修改部分   -------   把用k得到的ei拼接起来成60维的ei,输入critic或vae的decode"""
        x = action_rep.get_embedding(int(discrete_action[0]) - 1).to(device)
        for i in range(1,U):
            y = action_rep.get_embedding(discrete_action[i] - 1).to(device)
            x = torch.cat((x, y), dim=0)
        a1 = x.reshape(1, -1).squeeze().long().to(device)
        """自己修改部分"""
        discrete_emb_1=a1.cpu().view(-1).data.numpy()
        # discrete_emb_1 = action_rep.get_embedding(discrete_action).cpu().view(-1).data.numpy()  #根据选好的k，找到对应的ei

        all_parameter_action = action_rep.select_parameter_action(state, true_parameter_emb, #输入s，z，ei，decode出原为两个数，现在是30维
                                                                  discrete_emb_1)  #state为90，true_parameter_emb为6，discrete_emb_1应为60

        """ 修改  ----------------------------------------"""
        # if discrete_action == 1 or discrete_action == 2:
        #     all_parameter_action = all_parameter_action[0]
        parameter_action = all_parameter_action   #选好的连续参数
        for i in range(len(parameter_action)):  #防止动作里有0
            if parameter_action[i] <=0.:
                parameter_action[i] = 10.
        action = pad_action(discrete_action, parameter_action)
        episode_reward = 0.
        episode_system_gain=0.

        # if discrete_action == 1 or discrete_action == 2:
        #     parameter_action = np.append(parameter_action, 0.) #如果连续参数只要第一个，就把第二个设为0

        if t >= args.start_timesteps:  #超过start_timesteps步再训练更新网络 - 2500 - 经过2500后经验池有25个完整100步
            discrete_relable_rate, parameter_relable_rate,x = policy.train(replay_buffer, action_rep, c_rate,
                                                                         recon_s,
                                                                         args.batch_size)
        # if t % 100 == 0:
        #     print("discrete_relable_rate, parameter_relable_rate", discrete_relable_rate, parameter_relable_rate)
        for i in range(max_steps): #设为100个时隙T

            cur_step = cur_step + 1
            total_timesteps += 1
            ret = env.step(action)
            next_state, reward, terminal, t_com,pre_loc,miu = ret
            # if t % 100 == 0:
            # reward_scale
            # r = reward * reward_scale
            next_state = np.array(next_state, dtype=np.float32, copy=False)
            state_next_state = next_state - state
            replay_buffer.add(state, discrete_action=discrete_action, parameter_action=parameter_action,
                              all_parameter_action=None,
                              discrete_emb=discrete_emb,
                              parameter_emb=parameter_emb,
                              next_state=next_state,
                              state_next_state=state_next_state,
                              reward=reward, done=terminal, t_com=t_com,pre_loc=pre_loc,
                              actor_hx=actor_hx, actor_cx=actor_cx,miu=miu)
            replay_buffer_embedding.add(state, discrete_action=discrete_action, parameter_action=parameter_action,
                                        all_parameter_action=parameter_action,
                                        discrete_emb=None,
                                        parameter_emb=None,
                                        next_state=next_state,
                                        state_next_state=state_next_state,
                                        reward=reward, done=terminal, t_com=t_com,pre_loc=pre_loc,
                                        actor_hx=actor_hx, actor_cx=actor_cx,miu=miu)

            next_discrete_emb, next_parameter_emb,next_actor_hx,next_actor_cx = policy.select_action(next_state, actor_hx, actor_cx)

            # if t % 100 == 0:
            #     print("策略输出", next_discrete_emb, next_parameter_emb)
            next_discrete_emb = (
                    next_discrete_emb + np.random.normal(0, max_action * epsilon, size=discrete_emb_dim*U)
            ).clip(-max_action, max_action)
            next_parameter_emb = (
                    next_parameter_emb + np.random.normal(0, max_action * epsilon, size=parameter_emb_dim)
            ).clip(-3, 3)
            """------------------------------.clip(-max_action, max_action)---------------------"""
            # next_parameter_emb = next_parameter_emb * c_rate
            true_next_parameter_emb = true_parameter_action(next_parameter_emb, c_rate)
            # select discrete action
            next_discrete_action_embedding = copy.deepcopy(next_discrete_emb)
            next_discrete_action_embedding = torch.from_numpy(next_discrete_action_embedding).float().reshape(1, -1)
            next_discrete_action = action_rep.select_discrete_action(next_discrete_action_embedding)

            """自己修改部分   -------   把用k得到的ei拼接起来成60维的ei,输入critic或vae的decode"""
            x = action_rep.get_embedding(next_discrete_action[0]-1).to(device)
            for i in range(1, U):
                y = action_rep.get_embedding(next_discrete_action[i] - 1).to(device)
                x = torch.cat((x, y), dim=0)
            a1 = x.reshape(1, -1).squeeze().long().to(device)
            """自己修改部分"""
            next_discrete_emb_1 = a1.cpu().view(-1).data.numpy()

            # next_discrete_emb_1 = action_rep.get_embedding(next_discrete_action).cpu().view(-1).data.numpy()
            # select parameter action
            next_all_parameter_action = action_rep.select_parameter_action(next_state, true_next_parameter_emb,
                                                                           next_discrete_emb_1)
            # if t % 100 == 0:
            #     print("真实动作", next_discrete_action, next_all_parameter_action)
            # env.render()

            # if next_discrete_action == 1 or next_discrete_action == 2:
            #     next_all_parameter_action = next_all_parameter_action[0]
            next_parameter_action = next_all_parameter_action
            for i in range(len(next_parameter_action)):  # 防止动作里有0
                if next_parameter_action[i] <= 0.:
                    next_parameter_action[i] = 10.
            next_action = pad_action(next_discrete_action, next_parameter_action)

            # if next_discrete_action == 1 or next_discrete_action == 2:
            #     next_parameter_action = np.append(next_parameter_action, 0.)
            discrete_emb, parameter_emb, action, discrete_action, parameter_action = next_discrete_emb, next_parameter_emb, next_action, next_discrete_action, next_parameter_action
            state = next_state
            actor_hx = next_actor_hx
            actor_cx = next_actor_cx
            if t >= args.start_timesteps:  #128  - 原为cur_step >= args.start_timesteps
                discrete_relable_rate, parameter_relable_rate,x = policy.train(replay_buffer, action_rep, c_rate, #x是预测损失
                                                                             recon_s,
                                                                             args.batch_size)
            episode_reward += reward   #episode_reward是每个大轮的reward
            episode_system_gain+=miu

            if total_timesteps % args.eval_freq == 0:  #要evaluate。eval_freq=400。这样每次eval时都是terminal，直接进入eval函数
                print("terminal",terminal)
                print(
                    '{0:5s} R:{1:.4f} r100:{2:.4f} sys_gain:{1:.4f}'.format(str(total_timesteps), total_reward / (t + 1),
                                                           np.array(returns[-100:]).mean(),total_system_gain / (t + 1)))
                act_hx = actor_hx
                act_cx = actor_cx
                while not terminal:
                    state = np.array(state, dtype=np.float32, copy=False)
                    discrete_emb, parameter_emb,hx,cx = policy.select_action(state,act_hx, act_cx)
                    true_parameter_emb = true_parameter_action(parameter_emb, c_rate)
                    discrete_action_embedding = copy.deepcopy(discrete_emb)
                    discrete_action_embedding = torch.from_numpy(discrete_action_embedding).float().reshape(1, -1)
                    discrete_action = action_rep.select_discrete_action(discrete_action_embedding)
                    """自己修改部分   -------   把用k得到的ei拼接起来成60维的ei,输入critic或vae的decode"""
                    x = action_rep.get_embedding(discrete_action[0] - 1).to(device)
                    for i in range(1, U):
                        y = action_rep.get_embedding(discrete_action[i] - 1).to(device)
                        x = torch.cat((x, y), dim=0)
                    a1 = x.reshape(1, -1).squeeze().long().to(device)
                    """自己修改部分"""
                    discrete_emb_1 = a1.cpu().view(-1).data.numpy()
                    all_parameter_action = action_rep.select_parameter_action(state, true_parameter_emb,
                                                                                   discrete_emb_1)
                    # discrete_emb_1 = action_rep.get_embedding(discrete_action).cpu().view(-1).data.numpy()

                    # if discrete_action == 1 or discrete_action == 2:
                    #     all_parameter_action = all_parameter_action[0]
                    parameter_action = all_parameter_action
                    for i in range(len(parameter_action)):  # 防止动作里有0
                        if parameter_action[i] <= 0.:
                            parameter_action[i] = 10.
                    action = pad_action(discrete_action, parameter_action)
                    next_state, reward, terminal, t_com,pre_loc,miu = env.step(action)
                    act_hx = hx
                    act_cx = cx

                Reward.append(total_reward / (t + 1))
                System_gain.append(total_system_gain / (t + 1))
                Reward_100.append(np.array(returns[-100:]).mean()) #和total_reward / (t + 1)含义一样

                Test_Reward,test_system_gain = evaluate(env, policy, action_rep, c_rate,episodes=50)
                Test_Reward_100.append(Test_Reward)
                Test_system_gain_100.append(test_system_gain)

            if terminal:
                break

        t = t + 1  #一个while加一
        returns.append(episode_reward)
        total_reward += episode_reward
        total_system_gain += episode_system_gain



        # vae 训练
        # if t % 1000 == 0 and t >= 1000:
        if t % internal == 0 and t >= 10:  #原为t >= 1000
            # print("表征调整")
            # print("vae train")
            c_rate, recon_s = vae_train(action_rep=action_rep, train_step=1, replay_buffer=replay_buffer_embedding,
                                        batch_size=VAE_batch_size, save_dir=save_dir, vae_save_model=vae_save_model,
                                        embed_lr=1e-4)
            recon_s_loss.append(recon_s)
            # print("discrete embedding", action_rep.discrete_embedding())
            # print("c_rate", c_rate)
            # print("recon_s", recon_s)
    print("returns" , returns)
    # print("len_returns" , len(returns))
    print("Reward" , Reward)
    # print("len_Reward" , len(Reward))
    print("Test_Reward_100" , Test_Reward_100)
    # print("len_Test_Reward_100" , len(Test_Reward_100))
    print("System_gain", System_gain)
    print("Test_system_gain_100", Test_system_gain_100)
    #returns和Reward是[tensor(-925.1777)]类型，Test_Reward_100是[-1044.0713]类型
    plot_learning_curve(list(range(len(returns))), [loss.cpu().numpy() if isinstance(loss, torch.Tensor) else loss for loss in returns],"episode","rewards","an episode's reward",
                        "res_fig/episode_reward.png")
    plot_learning_curve(list(range(len(Reward))), [loss.cpu().numpy() if isinstance(loss, torch.Tensor) else loss for loss in Reward],"episode","avg_rewards","average reward",
                        "res_fig/avg_reward.png")
    plot_learning_curve(list(range(len(Test_Reward_100))), [loss.cpu().numpy() if isinstance(loss, torch.Tensor) else loss for loss in Test_Reward_100],"episode","test_100episode_rewards","average of 100 episode's reward",
                        "res_fig/test_100episode_reward.png")
    plot_learning_curve(list(range(len(Test_system_gain_100))),
                        [loss.cpu().numpy() if isinstance(loss, torch.Tensor) else loss for loss in Test_system_gain_100],
                        "episode", "test_100episode_system_gain", "average of 100 episode's system gain",
                        "res_fig/test_100episode_system_gain.png")

    #保存结果数组
    returns_numpy = np.array([r.cpu().numpy() for r in returns])
    np.savetxt('results/returns.csv', returns_numpy, delimiter=',')

    Reward_numpy = np.array([r.cpu().numpy() for r in Reward])
    np.savetxt('results/Reward.csv', Reward_numpy, delimiter=',')

    np.savetxt('results/Test_Reward_100.csv', Test_Reward_100, delimiter=',')

    system_gain_numpy = np.array([r.cpu().numpy() for r in System_gain])
    np.savetxt('results/system_gain_numpy.csv', system_gain_numpy, delimiter=',')

def vae_train(action_rep, train_step, replay_buffer, batch_size, save_dir, vae_save_model, embed_lr):
    initial_losses = []
    for counter in range(int(train_step) + 10):
        losses = []
        state, discrete_action, parameter_action, all_parameter_action, discrete_emb, parameter_emb, next_state, state_next_state, reward, not_done = replay_buffer.sample(
            batch_size) #batch_size=64
        #state是64,90维度
        #discrete_action是64,10维度
        """"get_c_rate要把ei改为60维，因为要输入到vae中"""
        vae_loss, recon_loss_s, recon_loss_c, KL_loss = action_rep.unsupervised_loss(state,
                                                                                     discrete_action,
                                                                                     parameter_action,
                                                                                     state_next_state,
                                                                                     batch_size, embed_lr)
        losses.append(vae_loss)
        initial_losses.append(np.mean(losses))

        if counter % 100 == 0 and counter >= 100:
            # print("load discrete embedding", action_rep.discrete_embedding())
            print("vae_loss, recon_loss_s, recon_loss_c, KL_loss", vae_loss, recon_loss_s, recon_loss_c, KL_loss)
            print("Epoch {} loss:: {}".format(counter, np.mean(initial_losses[-50:])))

        # Terminate initial phase once action representations have converged.
        if len(initial_losses) >= train_step and np.mean(initial_losses[-5:]) + 1e-5 >= np.mean(initial_losses[-10:]):
            # print("vae_loss, recon_loss_s, recon_loss_c, KL_loss", vae_loss, recon_loss_s, recon_loss_c, KL_loss)
            # print("Epoch {} loss:: {}".format(counter, np.mean(initial_losses[-50:])))
            # print("Converged...", len(initial_losses))
            break
        if vae_save_model:
            if counter % 1000 == 0 and counter >= 1000:
                title = "vae" + "{}".format(str(counter))
                action_rep.save(title, save_dir)
                print("embedding save model")

    state_, discrete_action_, parameter_action_, all_parameter_action, discrete_emb, parameter_emb, next_state, state_next_state_, reward, not_done = replay_buffer.sample(
        batch_size=5000)
    c_rate, recon_s = action_rep.get_c_rate(state_, discrete_action_, parameter_action_,
                                            state_next_state_, batch_size=5000, range_rate=2)
    return c_rate, recon_s

# 这个函数计算动作参数的边界中值和偏移量：c_rate参数，它是一个包含动作上下界信息的向量
def count_boundary(c_rate):
    median = (c_rate[0] - c_rate[1]) / 2
    offset = c_rate[0] - 1 * median
    return median, offset

# 这个函数用于将标准化的参数动作转换回实际环境可接受的原始动作范围：
def true_parameter_action(parameter_action, c_rate):
    parameter_action_ = copy.deepcopy(parameter_action)
    for i in range(len(parameter_action)):
        median, offset = count_boundary(c_rate[i])
        parameter_action_[i] = parameter_action_[i] * median + offset
    return parameter_action_

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="P-TD3")  # Policy name (TD3, DDPG or OurDDPG)
    # parser.add_argument("--env", default='Goal-v0')  # platform goal HFO
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=5, type=int)  # Time steps initial random policy is used- 原为128
    parser.add_argument("--eval_freq", default=500, type=int)  # How often (time steps) we evaluate - 原为500
    parser.add_argument("--max_episodes", default=2000, type=int)  # Max time steps to run environment - 好像没用
    parser.add_argument("--max_embedding_episodes", default=1e5, type=int)  # Max time steps to run environment
    parser.add_argument("--max_timesteps", default=150, type=float)  # Max time steps to run environment for

    parser.add_argument("--epsilon_steps", default=10, type=int)  # Max time steps to epsilon environment
    parser.add_argument("--expl_noise_initial", default=1.0)  # Std of Gaussian exploration noise 1.0
    parser.add_argument("--expl_noise", default=0.1)  # Std of Gaussian exploration noise 0.1

    parser.add_argument("--relable_steps", default=1000, type=int)  # Max time steps relable - 好像没用
    parser.add_argument("--relable_initial", default=1.0)  #
    parser.add_argument("--relable_final", default=0.0)  #

    parser.add_argument("--batch_size", default=250, type=int)  # Batch size for both actor and critic - 原为128
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.1)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
    args = parser.parse_args()
    # for i in range(0, 5):
    #     args.seed = i
    run(args)