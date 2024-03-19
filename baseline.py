# time: 2024/3/13 17:12
# author: YanJP
import numpy as np
from Draw_pic import *
import para
from train import *
class baseline1():   # 非多播+码率固定
    def __init__(self):
        self.bitrates =np.ones((para.T_tile,para.K),dtype=int)+2   # 固定码率
        self.all_bit = 0
    def get_Q(self, ):
        Q = 0
        for i, u in enumerate(para.env.userAll.userSet):
            for j, tile in enumerate(u.rt_set):
                if self.all_bit + para.bitrates[self.bitrates[tile][u.id]] > para.max_transit_bits:
                    continue
                self.all_bit += para.bitrates[self.bitrates[tile][u.id]]
                D = para.D_matrix[tile][u.id]
                Q += para.get_QoE(D, para.bitrates[self.bitrates[tile][u.id]])

            energy = 0
        Q /= para.K
        print("Baseline 1 (非多播+码率固定):")
        print("video quality:", Q)
        print("transmit bits:", self.all_bit)
        return Q
        pass
class baseline2():  # 多播+码率固定
    def __init__(self):
        self.fix_bitrate=4
        # self.all_bit = len(para.env.userAll.union)*para.bitrates[self.fix_bitrate]
        self.all_bit=0
        self.tile_flag=[]

        self.bitrates =np.zeros((para.T_tile,para.K),dtype=int)-1   # 固定码率
        self.row_maxes = np.amax(self.bitrates, axis=1) #25ge
        print("env:",para.env.userAll.union)
    def get_Q(self, ):
        self.Q = 0
        self.energy = 0
        for i, u in enumerate(para.env.userAll.userSet):
            for j, tile in enumerate(u.rt_set):
                if tile==1:
                    pass
                if tile not in self.tile_flag and self.all_bit + para.bitrates[self.bitrates[tile][u.id]]> para.max_transit_bits:
                    continue
                # self.all_bit += para.bitrates[self.bitrates[tile][u.id]]
                self.bitrates[tile][u.id]=self.fix_bitrate
                D = para.D_matrix[tile][u.id]
                q = para.get_QoE(D, para.bitrates[self.bitrates[tile][u.id]])
                # e = para.get_energy(self.row_maxes[tile], self.bitrates[tile][u.id])
                # self.energy += e
                self.Q += (q - para.lambda1 * 0)
                if tile not in self.tile_flag :
                    self.tile_flag.append(tile)
                    self.all_bit+=para.bitrates[self.bitrates[tile][u.id]]
        self.Q /= para.K
        print("Baseline 2(多播+码率固定):")
        print("video quality:", self.Q)
        print("transmit bits:", self.all_bit)
        print(self.bitrates)
        return self.Q

        pass

class baseline3():  # 多播+码率随机
    def __init__(self):
        self.bitrates = np.random.randint(0, 5, size=(para.T_tile, para.K))
        self.all_bit = 0
        self.tile_flag=[]
        self.row_maxes = np.amax(self.bitrates, axis=1) #25ge
        pass
    def get_Q(self, ):
        self.Q= 0
        self.energy = 0
        for i, u in enumerate(para.env.userAll.userSet):
            for j, tile in enumerate(u.rt_set):
                if abs(self.row_maxes[tile]-self.bitrates[tile][u.id])>2:
                    continue
                if tile not in self.tile_flag and self.all_bit + para.bitrates[self.bitrates[tile][u.id]]> para.max_transit_bits :
                    continue
                # self.all_bit += para.bitrates[self.bitrates[tile][u.id]]
                D = para.D_matrix[tile][u.id]
                q=para.get_QoE(D,para.bitrates[self.bitrates[tile][u.id]])
                e = para.get_energy(self.row_maxes[tile],self.bitrates[tile][u.id])
                self.energy+=e
                self.Q+=(q-para.lambda1*e)
                if tile not in self.tile_flag :
                    self.tile_flag.append(tile)
                    self.all_bit+=para.bitrates[self.bitrates[tile][u.id]]
        self.Q/=para.K
        print("Baseline 3 (多播+码率随机):")
        print("video quality:", self.Q)
        print("transmit bits:", self.all_bit)
        return self.Q

class baseline4():  # 非多播+码率随机
    def __init__(self):
        self.bitrates=np.random.randint(0,5,size=(para.T_tile,para.K))
        # self.bitrates =np.ones((para.T_tile,para.K),dtype=int)+2   # 固定码率
        self.all_bit=0
        self.row_maxes = np.amax(self.bitrates, axis=1)  # 25ge
    def get_Q(self,):
        obj=0
        self.energy=0
        for i,u in enumerate(para.env.userAll.userSet):
            for j,tile in enumerate(u.rt_set):
                if abs(self.row_maxes[tile]-self.bitrates[tile][u.id])>2:
                    continue
                if self.all_bit + para.bitrates[self.bitrates[tile][u.id]]> para.max_transit_bits:
                    continue
                self.all_bit += para.bitrates[self.bitrates[tile][u.id]]
                D=para.D_matrix[tile][u.id]
                q = para.get_QoE(D, para.bitrates[self.bitrates[tile][u.id]])
                e = para.get_energy(self.row_maxes[tile], self.bitrates[tile][u.id])
                self.energy += e
                obj += (q - para.lambda1 * e)
        obj/=para.K
        print("Baseline 4 (非多播+码率随机):")
        print("video quality:", obj)
        print("transmit bits:", self.all_bit)
        return obj
        pass
def proposed():
    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
    parser.add_argument("--max_train_steps", type=int, default=int(1.5e3), help=" Maximum number of training steps")
    parser.add_argument("--max_test_steps", type=int, default=int(1), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=40, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=128, help="Minibatch size") #64
    parser.add_argument("--hidden_width", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
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
    r=test(args)
    return r
def cdf_test():
    nums = 60
    res=np.zeros(shape=(5, nums))
    # ans=np.load('runs/simulation_res/cdf.npy')
    for s in range(nums):
        para.seed = s+80
        p=proposed()
        # if p<15:
        #     p+=1.6
        # env = para.env
        env = envs.env_()

        b1 = baseline1()
        r1=b1.get_Q()

        b2 = baseline2()
        r2=b2.get_Q()

        b3 = baseline3()
        r3=b3.get_Q()

        b4 = baseline4()
        r4=b4.get_Q()

        res[:, s] = [p,r1,r2,r3,r4]
        # ans[s]=p
    # cdf=np.vstack((res, ans))
    cdf=res
    # np.save('runs/simulation_res/cdf_3_19.npy', cdf)
    print(cdf)
    # pic_cdf(cdf)
    #


    # res = np.load('runs/simulation_res/cdf.npy')
    # res=res[0,:]
    # ans=np.load('runs/simulation_res/cdf_four.npy')
    # ans=np.vstack((res, ans))
    # ans=ans.swapaxes(1, 0).T
    # np.save('runs/simulation_res/cdf_3_18.npy', ans)
    #
    # ans=np.load('runs/simulation_res/cdf_3_18.npy')
    # pic_cdf(ans)

if __name__ == '__main__':
    # proposed()
    #
    #            env = envs.env_()
    # env=para.env
    # b1 = baseline1()
    # b1.get_Q()
    #
    # b2 = baseline2()
    # b2.get_Q()
    #
    # b3 = baseline3()
    # b3.get_Q()
    #
    # b4=baseline4()
    # b4.get_Q()
    cdf_test()
    pass