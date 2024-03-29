# time: 2024/3/10 14:49
# author: YanJP
import  numpy as np
import para

class Alluser():
    def __init__(self):
        self.userSet = []
        for i in range(para.K):
            self.userSet.append(User(i))
        self.get_Si()
        self.get_Ki()
    def get_Si(self,):  # 均匀分割
        array=[i for i in range(para.T_tile)]
        # 将数组分成长度为x的子数组
        x=para.T_tile//para.len_Si
        subarrays = [array[i:i + x] for i in range(0, len(array), x)]
        # para.len_Si=len(subarrays)
        self.S_set=[]
        for s in subarrays:
            # self.S_set.append(Tile_multicast(len(s),s))
            self.S_set.append(s)

    def get_Ki(self,):
        self.K_set = [[] for _ in range(len(self.S_set))]
        for i,user in enumerate(self.userSet):
            for j,si in enumerate(self.S_set):
                inter=np.intersect1d(np.array(user.rt_set),np.array(si))
                if len(inter)>0:
                    self.K_set[j].append(user)



class Tile_multicast():  #Si
    def __init__(self,num_tile,tile_set):
        self.num_user=num_tile
        self.user_set=tile_set

class User_multicast():  #Ki
    def __init__(self,num_user,user_set):
        self.num_user=num_user
        self.user_set=user_set



class User():
    def __init__(self,id):
        self.id=id
    # def request_tile(self):
        self.D=np.random.choice(para.Ds)
        self.rt_set=para.request()



class env_():
    def __init__(self):

        self.action_dim=para.action_dim
        self.observation_space=(para.state_dim,)
        self.reward=0
        self._max_episode_steps=para.T_tile*para.K
        self.userAll=Alluser()
        self.h=para.h
        self.Si_last=[]
        for si in self.userAll.S_set:
            self.Si_last.append(si[-1])
        pass
    def reset(self,):
        self.done=0
        self.t=0
        self.steps=0
        self.cur_user=0
        self.Si_birates=[0]*len(self.userAll.S_set)
        self.res_p=[]
        self.res_birate=np.zeros((para.T_tile,para.K))
        self.user_transcodebit=np.array([para.Bit_max]*para.K)
        # obs=np.concatenate((self.now_h,np.array([self.Nc_left_norm,sum(self.salency[self.index])])),axis=0)
        obs = self.user_transcodebit
        return obs
        # state：[time_step, carrier_left, tile_number]  加不加上tilenumber呢，这很有影响
        pass
    def step(self,action):
        for i, si in enumerate(self.userAll.S_set):
            if self.t in si:
                self.group_idx=i
                # print("给定的tile:", self.t, "在Si的第", i + 1, "个子列表中")
                break
        self.Si_birates[self.group_idx]+=para.bitrates[action]
        reward=0
        energy=0
        for u in self.userAll.K_set[self.group_idx]:
            if u.id==self.cur_user:
                D=para.D_matrix[self.t,self.cur_user]
                self.user_transcodebit[self.cur_user]-=para.bitrates[action]-para.bitrates[0]
                if self.user_transcodebit[self.cur_user]<0:
                    self.user_transcodebit[self.cur_user] += para.bitrates[action] - para.bitrates[0]
                    action=0
                    # reward=0
                reward=para.get_QoE(D,para.bitrates[action])
                # energy=para.e*(action)
                # reward-=energy
                break
        # if self.t in self.Si_last:
        #     threshold=self.Si_birates[self.group_idx]
        #     pass
        self.res_birate[self.t,self.cur_user]=action
        self.steps+=1
        self.t+=1
        if self.steps%para.T_tile==0:
            self.cur_user+=1
            self.t=0
        if self.steps==para.T_tile*para.K:
            self.done=1.0
            obs=self.user_transcodebit/1e6
        else:
            # self.now_h = self.h[self.pos:self.pos + para.action_dim, self.index]
            # obs = np.concatenate((self.now_h, np.array([self.Nc_left / para.N_c,sum(self.salency[self.index])])), axis=0)
            # obs=np.array([0.0]*para.state_dim)
            obs = self.user_transcodebit / 1e6

            # obs=np.concatenate((obs,np.sum(self.salency[self.index])),axis=0)
        return obs, reward, self.done, None

if __name__ == '__main__':
    alluser=Alluser()

    pass