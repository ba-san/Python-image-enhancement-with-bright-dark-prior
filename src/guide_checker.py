import os
import numpy as np
from guidedfilter_experiment import boxfilter, guided_filter

I = np.arange(200).reshape((20, 10))
normI = (I - I.min()) / (I.max() - I.min())  # min-max normalize I
r = 5
(M, N) = (20, 10)

#print(I)

sumY = np.cumsum(I, axis=0)
print(sumY)
print('######################')
#print(sumY[r:2*r + 1])
print(sumY[2*r + 1:] - sumY[:M - 2*r - 1])
print(sumY[2*r + 1:]) # from 2*r+1 to end lines
print(sumY[:M - 2*r - 1]) # same lines of above, from start
#print(np.tile(sumY[-1], (r, 1)) - sumY[M - 2*r - 1:M - r - 1])
exit()

#print(I[:r+1])
#print(I[:r+1].shape)
#print(I[r+1:M-r])
#print(I[r+1:M-r].shape)
#print(I[-r:])
#print(I[-r:].shape)

print(I[:, :r + 1])
print(I[:, r + 1:N - r])
print(I[:, -r:])

#boxfilter(normI, 5)
