x1 = np.arange(12).reshape(3,4)
x2 = np.copy(x1[:,:2])
x2 = x2*5
x2 = np.hstack([x2, np.zeros((x2.shape[0],2))])
x3 = x1+x2