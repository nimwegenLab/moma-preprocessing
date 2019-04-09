import numpy as np

def lsm_gauss3D(p,x,y,z,val):
    residual = fun_gauss3D(x,y,z,*p)-val
    return np.ravel(residual)

def fun_gauss3D(x,y,z, A, x0, y0,z0, sigma, sigmaZ, B):
    return A*np.exp(-((x-x0)**2)/(2*sigma)**2)*np.exp(-((y-y0)**2)/(2*sigma)**2)*np.exp(-((z-z0)**2)/(2*sigmaZ)**2) +B


def lsm_gauss3D_cstB(p,x,y,z,B,val):
    #p[-1]=80
    residual = fun_gauss3D_cstB(x,y,z,B,*p)-val
    return np.ravel(residual)

def fun_gauss3D_cstB(x,y,z,B, A, x0, y0,z0, sigma,sigmaZ):
    return A*np.exp(-((x-x0)**2)/(2*sigma)**2)*np.exp(-((y-y0)**2)/(2*sigma)**2)*np.exp(-((z-z0)**2)/(2*sigmaZ)**2) +B

def lsm_gauss3D_cstBsigma(p,x,y,z,B,sigma,sigmaZ,val):
    #p[-1]=80
    residual = fun_gauss3D_cstBsigma(x,y,z,B,sigma,sigmaZ,*p)-val
    return np.ravel(residual)

def fun_gauss3D_cstBsigma(x,y,z,B,sigma,sigmaZ, A, x0, y0,z0):
    return A*np.exp(-((x-x0)**2)/(2*sigma)**2)*np.exp(-((y-y0)**2)/(2*sigma)**2)*np.exp(-((z-z0)**2)/(2*sigmaZ)**2) +B


def lsm_gauss3D_cstsigma(p,x,y,z,sigma,sigmaZ,val):
    #p[-1]=80
    residual = fun_gauss3D_cstsigma(x,y,z,sigma,sigmaZ,*p)-val
    return np.ravel(residual)

def fun_gauss3D_cstsigma(x,y,z,sigma,sigmaZ, A, x0, y0,z0, B):
    return A*np.exp(-((x-x0)**2)/(2*sigma)**2)*np.exp(-((y-y0)**2)/(2*sigma)**2)*np.exp(-((z-z0)**2)/(2*sigmaZ)**2) +B

