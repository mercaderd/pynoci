import sys

import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray, float64, matrix, arange
from scipy import signal

from noci.jadeR import jadeR
from skimage.util import colormap


def dw(r):
    # Function validated with MATLAB data
    assert isinstance(r, ndarray), \
        "r (input data matrix) is of the wrong type (%s)" % type(r)
    assert r.ndim == 2, "X has %d dimensions, should be 2" % r.ndim
    r1=np.roll(r,-1,1)
    r2 = r1 - r
    r2 = r2[:, 0:-1]
    num = np.sum(np.power(r2,2),1)
    den = np.sum(np.power(r,2),1)
    dw = num/den
    return dw.T


def dwcriterion(X,maxICs=None,verbose=1):
    assert isinstance(X, ndarray), \
        "X (input data matrix) is of the wrong type (%s)" % type(X)
    X = matrix(X.astype(float64))
    assert X.ndim == 2, "X has %d dimensions, should be 2" % X.ndim
    assert (verbose == 0) or (verbose == 1) or (verbose == 2), \
        "verbose parameter should be either 0 or 1 or 2"
    [n, T] = X.shape
    assert n < T, "number of sensors must be smaller than number of samples"
    if maxICs==None:
        maxICs=n
    assert maxICs<=n,\
        "number of sources (%d) is larger than number of sensors (%d )" % (maxICs,n)
    DWMatrix = np.empty([maxICs,n])
    DWMatrix.fill(np.inf)
    for i in arange(maxICs):
        B=jadeR(X,m=i+1)
        Y = B * matrix(X)
        Xest = np.linalg.pinv(B)*Y
        Xr = X - Xest
        DWMatrix[i,:]=dw(Xr)
    if verbose  > 0:
        plt.imshow(DWMatrix, aspect="auto", clim=(0, 2), cmap ="plasma")
        if maxICs>10:
            yRange = np.arange(0,maxICs,np.floor_divide(maxICs,10))
        else:
            yRange = np.arange(0,maxICs,1)
        plt.yticks(yRange, yRange+1)
        if n>10:
            xRange = np.arange(0,n,np.floor_divide(n,10))
        else:
            xRange = np.arange(0,n,1)
        plt.xticks(xRange, xRange+1)
        plt.title('Durbin-Watson Criterion')
        plt.xlabel("Samples")
        plt.ylabel("# ICs")
        plt.colorbar()
        plt.show()
    return DWMatrix


def ICA_by_two_blocks(X,maxICs=None,verbose=1):
    assert isinstance(X, ndarray), \
        "X (input data matrix) is of the wrong type (%s)" % type(X)
    X = matrix(X.astype(float64))
    assert X.ndim == 2, "X has %d dimensions, should be 2" % X.ndim
    assert (verbose == 0) or (verbose == 1) or (verbose == 2), \
        "verbose parameter should be either 0 or 1 or 2"
    [n, T] = X.shape
    assert n < T, "number of sensors must be smaller than number of samples"
    nblocks = 2
    block_length = np.floor_divide(n, nblocks)
    if maxICs == None:
        maxICs = block_length
    assert maxICs <= block_length, \
        "number of sources (%d) is larger than number of sensors (%d )" % (maxICs, n)
    nblocks=2
    blocks = np.empty([block_length, T, nblocks])
    for i in range(nblocks):
        blocks[:,:, i]=X[(i * block_length):((i+1) * block_length),:]
    correlation_data = []
    print(maxICs)
    for i in arange(maxICs):
        B1 = jadeR(blocks[:, :, 0], m=i + 1)
        B2 = jadeR(blocks[:, :, 1], m=i + 1)
        Y1 = B1 * matrix(blocks[:, :, 0])
        Y2 = B2 * matrix(blocks[:, :, 1])
        CMat = np.sort(np.nanmax(np.abs(np.corrcoef(Y1, Y2)[:i + 1, i + 1:]), axis=1))
        correlation_data.append(CMat[::-1]) #Append correlations in descent order
    if verbose  > 0:
        for correlation in correlation_data:
            plt.plot(correlation,marker='x')
        if maxICs>10:
            xRange = np.arange(0,maxICs,np.floor_divide(maxICs,10))
        else:
            xRange = np.arange(0,maxICs,1)
        plt.xticks(xRange, xRange + 1)
        plt.yticks([0,0.25,0.5,0.75,1])
        plt.title('ICA by 2 blocks')
        plt.xlabel("# ICs")
        plt.ylabel("Correlation")
        plt.show()
    return correlation_data


def lcc(X, threshold = 0.1, verbose=1):
    assert isinstance(X, ndarray), \
        "X (input data matrix) is of the wrong type (%s)" % type(X)
    X = matrix(X.astype(float64))
    assert X.ndim == 2, "X has %d dimensions, should be 2" % X.ndim
    assert (verbose == 0) or (verbose == 1) or (verbose == 2), \
        "verbose parameter should be either 0 or 1 or 2"
    [n, T] = X.shape
    assert n < T, "number of sensors must be smaller than number of samples"
    assert threshold > 0, "threshold must be higher than zero"
    noci = None
    for i in arange(1,n):
        B = jadeR(X, m=i + 1)
        Y = B * matrix(X)
        corr = np.corrcoef(Y)
        maxCorrelation = np.nanmax(np.nanmax(corr-corr.diagonal()*np.eye(i+1)))
        if maxCorrelation > 0.1:
            noci = i
            break
    return noci


def main():
    np.random.seed(0)
    n_samples = 2000
    time = np.linspace(0, 8, n_samples)

    s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
    s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
    s3 = signal.sawtooth(2 * np.pi * time)  # Signal 3: saw tooth signal

    S = np.c_[s1, s2, s3]
    S += 0.1 * np.random.normal(size=S.shape)  # Add noise
    S /= S.std(axis=0)  # Standardize data
    S=S.T
    # Mix data
    A = np.random.random([40, 3]) # Mixing matrix
    X = np.dot(A, S)  # Generate observations
    print(dwcriterion(X,maxICs=15,verbose=1))
    print(ICA_by_two_blocks(X,maxICs=10,verbose=1))
    print(lcc(X,threshold=0.1,verbose=1))

if __name__ == "__main__":
    sys.exit(main())
