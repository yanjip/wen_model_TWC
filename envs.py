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
        self.get_matrix()
        self.get_union()
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

    def get_matrix(self,):
        self.matrix=np.zeros((para.T_tile,para.K))
        for i, user in enumerate(self.userSet):
            for j in user.rt_set:
                self.matrix[j][i] = 1
        self.sum_matrix=np.sum(self.matrix)
    def get_union(self,):
        self.union=np.array([],dtype=int)
        for i, user in enumerate(self.userSet):
            self.union=np.union1d(self.union,user.rt_set)



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
    #     self.D=np.random.choice(para.Ds)
        self.rt_set=para.request()



class env_():
    def __init__(self):

        self.action_dim=para.action_dim
        self.observation_space=(para.state_dim,)
        self.reward=0
        self._max_episode_steps=para.N_fov*para.K
        self.userAll=Alluser()
        self.h=para.h
        self.Si_last=[]
        for si in self.userAll.S_set:
            self.Si_last.append(si[-1])
        self.get_index()
        self.tile_union=self.userAll.union
        pass
    def get_index(self,):
        self.indexs=[]
        for u in self.userAll.userSet:
            self.indexs.append(u.rt_set)
    def get_obs(self,):
        row_maxes = np.amax(self.res_birate, axis=1)
        return row_maxes

    def reset(self,):
        self.done=0
        self.t=0
        self.cur_user=0
        # self.t_index=self.indexs[self.cur_user][self.t]
        self.steps=0
        self.Si_birates=[0]*len(self.userAll.S_set)
        self.res_p=[]
        self.res_energy_consume=0
        self.res_birate=np.zeros((para.T_tile,para.K),dtype=int)-1
        self.user_transcodebit=np.array([para.Bit_max]*para.K)
        self.tile_bit_choose = {element: -1 for element in self.tile_union}
        self.t_index=self.indexs[self.cur_user][self.t]
        self.D = para.D_matrix[self.t_index, self.cur_user]
        # obs = self.user_transcodebit
        self.sum_bits=0
        obs=self.get_obs()
        obs=np.concatenate((obs,np.array([self.D,(para.max_transit_bits-self.sum_bits)/1e8])),axis=0)
        return obs
        # state：[time_step, carrier_left, tile_number]  加不加上tilenumber呢，这很有影响
        pass
    def step(self,action):
        # for i, si in enumerate(self.userAll.S_set):
        #     if self.t in si:
        #         self.group_idx=i
        #         # print("给定的tile:", self.t, "在Si的第", i + 1, "个子列表中")
        #         break
        # self.Si_birates[self.group_idx]+=para.bitrates[action]
        # reward=0
        # D = para.D_matrix[self.t_index, self.cur_user]
        D=self.D
        # self.user_transcodebit[self.cur_user] -= para.bitrates[action] - para.bitrates[0]
        # if self.user_transcodebit[self.cur_user] < 0:
        #     self.user_transcodebit[self.cur_user] += para.bitrates[action] - para.bitrates[0]
        #     action = 0
            # reward=0
        if  self.tile_bit_choose[self.t_index]==-1:
            while self.sum_bits+para.bitrates[action]>para.max_transit_bits:
                action-=1
                if action<0:
                    action=-1
                    break
            if action!=-1:
                self.sum_bits +=para.bitrates[action]
            self.tile_bit_choose[self.t_index]=action  # 你现在就是最大的
        elif self.tile_bit_choose[self.t_index]!=-1:
            if action > self.tile_bit_choose[self.t_index]:
                action = self.tile_bit_choose[self.t_index]
            elif (self.tile_bit_choose[self.t_index]-action)>2:  # 这样写的话，原本选的码率为1，强行改成了4-2=2
                # action=self.tile_bit_choose[self.t_index]-2  #
                action=-1
        if action>-1:
            QoE = para.get_QoE(D, para.bitrates[action])
            energy_consume=para.get_energy(self.tile_bit_choose[self.t_index],action)
            self.res_energy_consume+=energy_consume
            reward=QoE-para.lambda1*energy_consume
        else: reward=0
        # for u in self.userAll.K_set[self.group_idx]:
        #     if u.id==self.cur_user:
        #         D=para.D_matrix[self.t,self.cur_user]
        #         self.user_transcodebit[self.cur_user]-=para.bitrates[action]-para.bitrates[0]
        #         if self.user_transcodebit[self.cur_user]<0:
        #             self.user_transcodebit[self.cur_user] += para.bitrates[action] - para.bitrates[0]
        #             action=0
        #             # reward=0
        #         reward=para.get_QoE(D,para.bitrates[action])
        #         # energy=para.e*(action)
        #         # reward-=energy
        #         break
        # if self.t in self.Si_last:
        #     threshold=self.Si_birates[self.group_idx]
        #     pass
        self.res_birate[self.t_index,self.cur_user]=action
        self.steps+=1
        self.t+=1
        if self.steps%para.N_fov==0:  # 现在是对每个用户依次选择他FoV里面tile的码率
            self.cur_user+=1
            self.t=0
        if self.steps==para.N_fov*para.K:
            self.done=1.0
        #     obs=self.user_transcodebit/1e6
            self.D=0
        else:
            self.t_index = self.indexs[self.cur_user][self.t]
            self.D = para.D_matrix[self.t_index, self.cur_user]
        #     # self.now_h = self.h[self.pos:self.pos + para.action_dim, self.index]
            # obs = np.concatenate((self.now_h, np.array([self.Nc_left / para.N_c,sum(self.salency[self.index])])), axis=0)
            # obs=np.array([0.0]*para.state_dim)
        obs = self.get_obs()
        obs=np.concatenate((obs,np.array([self.D,(para.max_transit_bits-self.sum_bits)/1e8])),axis=0)
            # obs=np.concatenate((obs,np.sum(self.salency[self.index])),axis=0)
        return obs, reward, self.done, None
    def Si_rate(self,):
        self.sum_bits_si=[]
        for i,si in enumerate(self.userAll.S_set):
            sum_bit = 0
            for t in self.tile_union:
                if t in si:
                    sum_bit+=para.bitrates[self.tile_bit_choose[t]]
            self.sum_bits_si.append(sum_bit)
        return self.sum_bits_si
if __name__ == '__main__':
    alluser=Alluser()

    pass