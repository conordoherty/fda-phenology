import numpy as np

np.random.seed(0)

def sim0(per_class=1000):
    mean1 = np.array([-1,0])
    mean2 = np.array([1,0])
    cov = np.array([[1, 0], [0, 3]])
    
    class1 = np.random.multivariate_normal(mean1, cov, size=per_class)
    class2 = np.random.multivariate_normal(mean2, cov, size=per_class)
    
    x = np.vstack((class1, class2))
    y = np.concatenate((np.ones(per_class), np.zeros(per_class)))

    return x, y
