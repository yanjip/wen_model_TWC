# time: 2024/3/19 16:21
# author: YanJP
import numpy as np

from baseline import *
import para
from Draw_pic import *


def all_algor():
    nums = 10
    res = np.zeros(shape=(5, nums))
    # ans=np.load('runs/simulation_res/cdf.npy')
    for s in range(nums):
        para.seed = s + 100
        p = proposed()
        # if p<15:
        #     p+=1.6
        # env = para.env
        b1 = baseline1()
        r1 = b1.get_Q()

        b2 = baseline2()
        r2 = b2.get_Q()

        b3 = baseline3()
        r3 = b3.get_Q()

        b4 = baseline4()
        r4 = b4.get_Q()

        res[:, s] = [p, r1, r2, r3, r4]
    ans=np.mean(res, axis=1)
    print(ans)
    return ans

def change_p():
    p=[1,2,3,4,5]
    m_bits=(np.array([80,85,90,95,100])+5)*1e6
    res = np.zeros(shape=(5, len(m_bits)))
    for i,b in enumerate(m_bits):
        para.max_transit_bits=b
        ans=all_algor()
        res[:,i]=ans
    print(res)
    plot_bits(m_bits/1e6,res)


if __name__ == '__main__':
    change_p()
    pass

