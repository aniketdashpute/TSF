import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch

def gkern(l=5, sig=1.):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp((-0.5) * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

def getFINP(Nz, Ny, Nx):
    # fInp - laser source in the middle of the surface
    ## Use Gaussian profile if necessary (uncomment the code below)
    fInp = np.zeros([Ny, Nx])
    halfWindow = 2 # pixels, windowsie = 2*halfWindow+1
    fluxIntensity = 2
    depthPenetration = 2

    # laser_i, laser_j = Ny//2, Nx//2
    # size = halfWindow*2 + 1
    # sigmaBeam = 1.25
    # sigma = sigmaBeam
    # G = fluxIntensity*gkern(size, sigma)
    # fInp[laser_i-halfWindow:laser_i+halfWindow+1, laser_j-halfWindow:laser_j+halfWindow+1] = G

    fInp = np.zeros([Nz, Ny, Nx])
    laser_i, laser_j = Ny//2, Nx//2
    halfWindow = 1 # pixels, windowsie = 2*halfWindow+1
    fInp[:depthPenetration, laser_i-halfWindow:laser_i+halfWindow+1, laser_j-halfWindow:laser_j+halfWindow+1] = fluxIntensity
    
    plt.figure()
    plt.imshow(fInp[0,:,:])
    plt.title("fInp")
    plt.colorbar()

    return fInp

class ForwardFD():
    def __init__(self, params, device):
        self.device = device
        self.Nt, self.Ny, self.Nx = params["Nt"], params["Ny"], params["Nx"]
        self.Nz = params["Nz"]

        self.total_time = params["total_time"]
        self.numCycles = params["numCycles"]
        self.tsCycle = self.Nt//self.numCycles
        self.tsCycleON = params["tsCycleON"]

        # Deltas
        self.delT = params["delT"] # seconds
        self.delTimgf = params["delTimgf"] # multiple of delT for which we have images
        self.delX, self.delY, self.delZ = params["delX"], params["delY"], params["delZ"]
        self.T_init = params["T_init"]
        imgs = params["imgs"]
        self.T_init_tensor = torch.tensor(imgs[0,:,:])

        self.numLayersK = params["numLayersK"]
        self.numLayersEps = params["numLayersEps"]

        self.EpsFactor = params["EpsFactor"]

        print("self.tsCycle: ", self.tsCycle)
        print("self.tsCycleON: ", self.tsCycleON)
        print("self.T_init: ", self.T_init)

    def getLaplacian_torch(self, U_t):
        # Using Neumann Boundary conditions - heat flux = 0 on boundaries
        # Therefore, on edges, left = right (if either of them does not exist)
        # Get x component values
        # Ux-1,y,z
        U_xm1 = torch.roll(U_t, 1, dims=2)
        U_xm1[:,:,0] = U_t[:,:,1]
        # Ux+1,y,z
        U_xp1 = torch.roll(U_t, -1, dims=2)
        U_xp1[:,:,-1] = U_t[:,:,-2]
        # Laplacian in X
        U_lap_x = (U_xp1 + U_xm1 - 2 * U_t)/(self.delX**2)

        # Get y component values
        # Ux,y-1,z
        U_ym1 = torch.roll(U_t, 1, dims=1)
        U_ym1[:,0,:] = U_t[:,1,:]
        # Ux,y+1,z
        U_yp1 = torch.roll(U_t, -1, dims=1)
        U_yp1[:,-1,:] = U_t[:,-2,:]
        # Laplacian in Y
        U_lap_y = (U_yp1 + U_ym1 - 2 * U_t)/(self.delY**2)

        # Get z component values
        # Ux,y1,z-1
        U_zm1 = torch.roll(U_t, 1, dims=0)
        U_zm1[0,:,:] = U_t[1,:,:]
        # U_zm1[0,:,:] = U_t[0,:,:]
        # Ux,y,z+1
        U_zp1 = torch.roll(U_t, -1, dims=0)
        U_zp1[-1,:,:] = U_t[-2,:,:]
        # Laplacian in Y
        U_lap_z = (U_zp1 + U_zm1 - 2 * U_t)/(self.delZ**2)

        U_lap = U_lap_x + U_lap_y + U_lap_z

        return U_lap

    def bIsSourceON(self, timestep):
        t = timestep%self.tsCycle
        if (t<self.tsCycleON):
            return True
        else:
            return False

    def funcFDSim_torch(self, K_top, Eps_top, fInp, Mu, plotting=False):
        u_center = [self.T_init]
        u_center_L = [self.T_init]
        u_center_R = [self.T_init]

        Eps = torch.zeros((self.Nz, self.Ny, self.Nx)).to(self.device)
        topLayers = Eps_top.shape[0]
        Eps[:topLayers,:,:] = Eps_top

        if (K_top.shape[0]>1):
            K = torch.ones((self.Nz, self.Ny, self.Nx))
            K[0,:,:] = K_top[0,:,:]
            K[1:,:,:] = K_top[1,:,:]
        else:
            K = K_top[0,:,:]
        
        K = K.to(self.device)
        Mu = Mu.to(self.device)
        K = torch.clamp(K, min=0.0,max=5.0)
        Mu = torch.clamp(Mu, min=0.0,max=100.0)
        Eps = torch.clamp(Eps, min=0.0, max=100.0)
        # Initial conditions on temperature profile U(xyz,t)
        U_t = (self.T_init * torch.ones([self.Nz, self.Ny, self.Nx])).to(self.device)
        U_tplus1 = (self.T_init * torch.ones([self.Nz, self.Ny, self.Nx])).to(self.device)

        imgs_surface_all = torch.zeros([self.Nt, self.Ny, self.Nx]).to(self.device)
        img_thermal = torch.zeros([self.Ny, self.Nx]).to(self.device)
        imgs_slice_all = torch.zeros([self.Nt, self.Nz, self.Ny]).to(self.device)
        img_slice = torch.zeros([self.Nz, self.Ny]).to(self.device)

        for time in range(self.Nt*self.delTimgf):

            bIsON = self.bIsSourceON(time)            
            U_lap_t = self.getLaplacian_torch(U_t)
            sourceTerm = self.EpsFactor * Eps * fInp
            diffusionTerm = K * 1e-7 * U_lap_t
            sigma0 = 5.67e-8 # Stefan Boltzmann Constant
            radiativeLoss = Mu * 1e-2 * sigma0 * (U_t[0,:,:]**4-self.T_init**4)
            if (bIsON):
                U_tplus1 = U_t + self.delT * self.delTimgf * (diffusionTerm - radiativeLoss + sourceTerm)
            else: #time > tON
                U_tplus1 = U_t + self.delT * self.delTimgf * (diffusionTerm - radiativeLoss)
            
            # Sample plotting after each some timesteps
            if (time % self.delTimgf == 0):
                img_thermal = U_t[0,:,:]
                imgs_surface_all[time//self.delTimgf, :, :] = img_thermal
                img_slice = U_t[:,:,self.Nx//2]
                imgs_slice_all[time//self.delTimgf, :, :] = img_slice
            
            U_t = U_tplus1
            u_center.append(U_t[0, self.Ny//2, self.Nx//2])
            u_center_L.append(U_t[0, self.Ny//2, self.Nx//2-5])
            u_center_R.append(U_t[0, self.Ny//2, self.Nx//2+5])

        if (plotting==True):
            plt.figure()
            plt.plot(u_center, label="center")
            plt.plot(u_center_L, label="left")
            plt.plot(u_center_R, label="right")
            plt.title("Plotting surface center temperature over time")
            plt.legend()

        return imgs_surface_all, imgs_slice_all

    ## Save the simulation data
    def saveSim(self):
        K = 0.8 # *e-7
        K = 1.00*np.ones((self.numLayersK, self.Ny, self.Nx))
        if (self.numLayersK>1): K[1,:,:] = 2.00*K[1,:,:]
        Eps = 1.00*np.ones((self.numLayersEps, self.Ny, self.Nx))
        if (self.numLayersEps>1): Eps[1,:,:] = 0.5*Eps[1,:,:]
        fInp = getFINP(self.Nz, self.Ny, self.Nx)
        imgs_all = self.funcFDSim(K, Eps, fInp)
        print("imgs_all.shape: ", imgs_all.shape)

        imgs_save = {}
        imgs_save["imgs"] = imgs_all
        imgs_save["K"] = K
        imgs_save["Eps"] = Eps
        # imgs_save["time_ON"] = self.time_ON
        imgs_save["total_time"] = self.total_time
        # imgs_save["timestepsON"] = self.tsON1
        imgs_save["delX"] = self.delX
        imgs_save["delY"] = self.delY
        imgs_save["delZ"] = self.delZ
        imgs_save["Nz"] = self.Nz
        imgs_save["delT"] = self.delT
        imgs_save["T_init"] = self.T_init
        imgs_save["fInp"] = fInp
        imgs_save["ObjectName"] = strObjName
        imgs_save["numLayersEps"] = numLayersEps
        imgs_save["numLayersK"] = numLayersK
        imgs_save["Nt"] = self.Nt
        imgs_save["Nx"] = self.Nx
        imgs_save["Ny"] = self.Ny
        imgs_save["Nz"] = self.Nz
        imgs_save["numCycles"] = self.numCycles
        imgs_save["tsCycleON"] = self.tsCycleON

        delTimgf = 1
        imgs_save["delT"] = self.delT/delTimgf
        imgs_save["delTimgf"] = delTimgf


        ## Save the data for future use
        sfilename = "../data/image_mydata3D_" + strObjName + ".pkl"
        with open(sfilename, 'wb') as handle:
            pickle.dump(imgs_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return

## Load the simulated data
def loadSim():
    sfilename = "../data/image_mydata3D_" + strObjName + ".pkl"
    with open(sfilename, 'rb') as handle:
        data_saved = pickle.load(handle)

    im_all = data_saved["imgs"]
    Nt = im_all.shape[0]

    tot_disp = 20
    for iter in range(tot_disp):
        plt.figure()
        plt.imshow(im_all[(iter)*Nt//tot_disp,:,:])
        stitle = "Displaying intermediate images, iter: " + str(iter)
        plt.title(stitle)
        plt.colorbar()

    for iter in range(numLayersK):
        plt.figure()
        plt.imshow(data_saved["K"][iter,:,:])
        stitle = "K, Layer: " + str(iter)
        plt.title(stitle)
        plt.colorbar()

    for iter in range(numLayersEps):
        plt.figure()
        plt.imshow(data_saved["Eps"][iter,:,:])
        stitle = "Eps, Layer: " + str(iter)
        plt.title(stitle)
        plt.colorbar()
