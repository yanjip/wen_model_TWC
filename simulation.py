# time: 2024/3/19 16:21
# author: YanJP
import numpy as np

from baseline import *
import para
from Draw_pic import *


def all_algor():
    nums = 25
    res = np.zeros(shape=(5, nums))
    # ans=np.load('runs/simulation_res/cdf.npy')
    for s in range(nums):
        # para.seed  = (s+1)
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
    p=[34,35,36,37,38,39]
    m_bits=(np.array([80,85,90,95,100,105])-10)*1e6
    # m_bits=(np.array([95,97,99,101,103])-3)*1e6

    res = np.zeros(shape=(5, len(m_bits)))
    for i,b in enumerate(m_bits):
        para.seed  = (i+6)
        np.random.seed(para.seed)
        para.max_transit_bits=b
        ans=all_algor()
        res[:,i]=ans
    print(res)
    np.save('runs/simulation_res/power_3_19.npy',res)
    plot_bits(p,res)

def only_Q():
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

    res = [p, r1, r2, r3, r4]
    Q= [a + para.lambda1*b for a, b in zip(res , para.energy)]
    print("Qulity:",Q)
    print("energy:", para.energy)
    print(res)
    np.save('runs/simulation_res/mix_bar_3_20.npy', [Q,para.energy,res])

    plot_mix_bar([Q,para.energy,res])



if __name__ == '__main__':
    # change_p()
    only_Q()
    pass

