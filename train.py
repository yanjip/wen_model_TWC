# time: 2024/3/9 10:05
# author: YanJP
# time: 2023/10/30 14:57
# author: YanJP
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import argparse

import para
from normalization import Normalization, RewardScaling
from replaybuffer import ReplayBuffer
from ppo_discrete import PPO_discrete
import envs
from tqdm import tqdm
from Draw_pic import *
import time
# from baseline import *
seed = para.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)
import pickle
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
def save_state_norm(state_norm):
    filepath="runs/model/state_norm/"+curr_time+'state_norm.pkl'
    with open(filepath, 'wb') as file:
        pickle.dump(state_norm, file)

def load_state_norm():
    file=get_file_model(folder_path="runs/model/state_norm")
    file = "runs/model/state_norm/" + file
    with open(file, 'rb') as file_name:
        s_n = pickle.load(file_name)
    return s_n

def evaluate_policy(args, env, agent, state_norm):
    times = 1
    evaluate_reward = 0
    for _ in range(times):
        s = env.reset()
        if args.use_state_norm:  # During the evaluating,update=False
            s = state_norm(s, update=False)
        done = False
        episode_reward = 0
        while not done:
            a = agent.evaluate(s)  # We use the deterministic policy during the evaluating
            s_, r, done, _ = env.step(a)
            if args.use_state_norm:
                s_ = state_norm(s_, update=False)
            episode_reward += r
            s = s_
        evaluate_reward += sum(env.res_p)
    print("reward:",episode_reward/para.K)
    print("sum_bits:",env.sum_bits)
    sum_bits_si = env.Si_rate()
    # return evaluate_reward / times
    return env.user_transcodebit,env.res_birate,env.res_energy_consume,sum_bits_si,episode_reward/para.K

def write_power(bitrates,user_transcodebit,energy_consume,sum_bits_si):
    with open('runs/sum_power.txt', 'a+') as F:
        F.write(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + '\n')
        # F.write("----Power:" + str(para.maxPower) + '\n')
        F.write("Bitrate_list:" + str(bitrates) + "\n\n")
        F.write("Energy_consume:" + str(energy_consume) + "\n\n")
        F.write("sum_bits_si:" + str(np.array(sum_bits_si)/1e6) + "Mbit\n\n")
        F.write("user_transcodebit_left:" + str(user_transcodebit/1e6) + "Mbit\n\n")

def get_file_model(folder_path = "runs/model/"):
    # folder_path = "runs/model/"  # 需要修改成你想要操作的文件夹路径
    file_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    sorted_files = sorted(file_list, key=lambda x: os.path.getmtime(os.path.join(folder_path, x)), reverse=True)
    latest_file = sorted_files[0]
    return latest_file
def main(args, time, seed):
    env = envs.env_()
    env_evaluate = envs.env_()

    args.state_dim = env.observation_space[0]
    # args.action_dim = env.action_space.n
    args.action_dim = env.action_dim
    args.max_episode_steps = env._max_episode_steps  # Maximum number of steps per episode
    # print("env={}".format(env_name))
    print("state_dim={}".format(args.state_dim))
    print("action_dim={}".format(args.action_dim))
    print("max_episode_steps={}".format(args.max_episode_steps))

    rewards=[]
    ma_rewards = []  # 记录所有回合的滑动平均奖励

    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating

    replay_buffer = ReplayBuffer(args)
    agent = PPO_discrete(args)

    # Build a tensorboard
    # writer = SummaryWriter(log_dir='runs/PPO_discrete/time_{}_seed_{}'.format(time, seed))

    state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
    if args.use_reward_norm:  # Trick 3:reward normalization
        reward_norm = Normalization(shape=1)
    elif args.use_reward_scaling:  # Trick 4:reward scaling
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

    for total_steps in tqdm(range(1,args.max_train_steps+1)):
        # para.h=para.generate_h()
    # while total_steps < args.max_train_steps:
        s = env.reset()
        if args.use_state_norm:
            s = state_norm(s)
        if args.use_reward_scaling:
            reward_scaling.reset()
        episode_steps = 0
        done = False
        episode_rewards=[]
        while not done:
            episode_steps += 1
            a, a_logprob = agent.choose_action(s)  # Action and the corresponding log probability
            s_, r, done, _ = env.step(a)
            # a=a_shape
            episode_rewards.append(r)
            if args.use_state_norm:
                s_ = state_norm(s_)
            if args.use_reward_norm:
                r = reward_norm(r)
            elif args.use_reward_scaling:
                r = reward_scaling(r)

            # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
            # dw means dead or win,there is no next state s';
            # but when reaching the max_episode_steps,there is a next state s' actually.
            if done and episode_steps != args.max_episode_steps:
                dw = True  #有next state
            else:
                dw = False

            replay_buffer.store(s, a, a_logprob, r, s_, dw, done)
            s = s_
            # total_steps += 1

            # When the number of transitions in buffer reaches batch_size,then update
            if replay_buffer.count == args.batch_size:
                agent.update(replay_buffer, total_steps)
                replay_buffer.count = 0
        ep_reward=sum(episode_rewards)/para.K
        print("ep_reward:",ep_reward)
        # sum_bits_si=env.Si_rate()
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
            # Evaluate the policy every 'evaluate_freq' steps
        if total_steps % args.evaluate_freq == 0:
            evaluate_num += 1
            user_transcodebit,bitrates,energy_consume,sum_bits_si,_ = evaluate_policy(args, env_evaluate, agent, state_norm)
            # evaluate_rewards.append(evaluate_reward)
            # np.set_printoptions(precision=3)
            # powersum= sum([a * b for a, b in zip(evaluate_power, carrier_allocation)])
            # carrier_sum= sum(carrier_allocation)

            print("num:{} \t Left user_transcodebit:{}\tbirate_seleciton:{}\t energy_consume:{}".format(evaluate_num,user_transcodebit,bitrates,energy_consume))
            # writer.add_scalar('step_rewards', evaluate_rewards[-1], global_step=total_steps)
            # Save the rewards
            # if evaluate_num % args.save_freq == 0:
            #     np.save('./data_train/PPO_discrete_time_{}_seed_{}.npy'.format(time, seed), np.array(evaluate_rewards))
        # if (total_steps + 1) % 2 == 0:
        #     print(f'train_step:{total_steps},power_all:{env.res_p}',)
    write_power(bitrates,user_transcodebit,energy_consume,sum_bits_si)
    path='runs/model/ppo_'+time+'.pth'
    torch.save(agent.actor.state_dict(), path)
    save_state_norm(state_norm)
    rewards=np.array(rewards)
    return {'episodes': range(len(rewards)), 'rewards': rewards, 'ma_rewards': ma_rewards}

def test(args,model_dic='rnn'):
    env = envs.env_()
    para.env=env
    args.state_dim = env.observation_space[0]
    # args.action_dim = env.action_space.n
    args.action_dim = env.action_dim
    args.max_episode_steps = env._max_episode_steps  # Maximum number of steps per episode
    evaluate_num = 0  # Record the number of evaluations
    agent = PPO_discrete(args)
    if model_dic == 'rnn':
        model=get_file_model()
    else:
        model=model_dic
    path='runs/model/'+model
    # path='runs/model/ppo_2023_12_17-20_40_16_B18k.pth'
    agent.load_model(path)
    state_norm =load_state_norm()
    # state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
    for total_steps in range(1,args.max_test_steps+1):
        user_transcodebit, bitrates, energy_consume, sum_bits_si,episode_reward = evaluate_policy(args, env, agent,
                                                                                   state_norm)
        print("sum_bits",sum(sum_bits_si))
        para.energy[0]=energy_consume
        print(" \t energy_consume:{}\tbirate_seleciton:{}".format( energy_consume,bitrates))
        # np.save('runs/simulation_res/bitrates.npy',bitrates)
    return episode_reward
        # ep_reward=sum(episode_rewards)
        # print("ep_reward:",ep_reward)
        # rewards.append(ep_reward)

def test_ppo():
    Nc=np.array([150,160,170,180,190,200])-60
    for i,nc in enumerate(Nc):
        para.N_c=nc
        ppo = test(args)
        print(ppo)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
    parser.add_argument("--max_train_steps", type=int, default=int(2.0e3), help=" Maximum number of training steps")
    parser.add_argument("--max_test_steps", type=int, default=int(1), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=40, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=128, help="Minibatch size") #64
    parser.add_argument("--hidden_width", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.96, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=20, help="PPO parameter") #default=10
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")

    args = parser.parse_args()
    curr_time = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    train=True
    # train=False
    train_log_dir='runs/rewards/'+curr_time
    if train:
        res_dic=main(args, curr_time, seed=seed)
        np.save(train_log_dir + '_reward.npy', np.array(res_dic['rewards']))

        plot_rewards(res_dic['rewards'],curr_time,path='runs/pic')
    else:
        test(args)




