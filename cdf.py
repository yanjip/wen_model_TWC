# time: 2024/3/19 15:18
# author: YanJP
import numpy as np
import para
from baseline import *
def cdf_test():
    nums = 50
    res=np.zeros(shape=(4, nums))
    # ans=np.load('runs/simulation_res/cdf.npy')
    for s in range(nums):
        para.seed = s
        p=proposed()
        # env = para.env
        env = envs.env_()

        b1 = baseline1()
        b1.get_Q()

        b2 = baseline2()
        b2.get_Q()

        b3 = baseline3()
        b3.get_Q()

        b4 = baseline4()
        b4.get_Q()

        res[:, s] = [p,b1,b2,b3,b4]
        # ans[s]=p
    # cdf=np.vstack((res, ans))
    cdf=res
    # np.save('runs/simulation_res/cdf.npy', cdf)
    print(cdf)
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
    cdf_test()