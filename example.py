'''Example quantile comparison

Adopted from http://www.nicebread.de/comparing-all-quantiles-of-two-distributions-simultaneously/.

Author: guangzhi XU (xugzhi1987@gmail.com)
Update time: 2022-08-14 22:08:01.
'''

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from wilcox_quantile import compare_quantiles, compare_quantiles_vec

if __name__ == '__main__':


    # generate sample data
    np.random.seed(33)
    rt1 = np.random.randn(100) * 52 + 350
    rt2 = np.r_[np.random.randn(85) * 55 + 375, np.random.randn(15) * 25 + 220]

    # compare quantiles, serial version
    t1 = time.time()
    comp = compare_quantiles(rt1, rt2, quantiles=np.arange(0.1, 1.0, 0.1))
    t2 = time.time()
    print('time of serial version:', t2-t1)

    # compare quantiles, vectorized version
    t1 = time.time()
    comp2 = compare_quantiles_vec(rt1, rt2, quantiles=np.arange(0.1, 1.0, 0.1))
    t2 = time.time()
    print('time of vectorized version:', t2-t1)

    # compare serial and vectorized version:
    for key in comp.keys():
        v1 = comp[key]
        v2 = comp2[key]
        print('Comparing ', key, 'is all close:', np.allclose(v1, v2))

    #-------------------Plot------------------------
    figure = plt.figure(figsize=(15, 6))
    ax = figure.add_subplot(1,2,1)

    kde1 = gaussian_kde(rt1)
    kde2 = gaussian_kde(rt2)
    plot_x = np.linspace(np.r_[rt1, rt2].min(), np.r_[rt1, rt2].max(), 100)
    pdf1 = kde1.evaluate(plot_x)
    pdf2 = kde2.evaluate(plot_x)

    ax.plot(plot_x, pdf1, 'k-', label='RT1')
    ax.plot(plot_x, pdf2, 'b-', label='RT2')
    ax.grid(axis='both')
    ax.legend()

    ax = figure.add_subplot(1,2,2)
    group1qs = comp['quantiles']

    diff1 = comp['estx-esty']
    ci_lower1 = comp['ci_lower']
    ci_upper1 = comp['ci_upper']

    ax.plot(group1qs, diff1, 'k-o', label='diff1')
    ax.plot(group1qs, ci_lower1, 'k+', label='ci')
    ax.plot(group1qs, ci_upper1, 'k+')

    diff2 = comp2['estx-esty']
    ci_lower2 = comp2['ci_lower']
    ci_upper2 = comp2['ci_upper']

    ax.plot(group1qs, diff2, 'b-o', label='diff2')
    ax.plot(group1qs, ci_lower2, 'b+', label='ci2')
    ax.plot(group1qs, ci_upper2, 'b+')
    ax.legend()
    ax.grid(axis='both')
    ax.set_xlabel('Group 1 quantiles')
    ax.set_ylabel('Est_1 - Est_2')

    figure.show()
    figure.savefig('example_plot.png')
