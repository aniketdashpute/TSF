import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

import torch
from torch import nn
from torch.functional import F

from forwardFD import ForwardFD

# Toggle this flag to enable/disable wandb
useWanb = False
if (useWanb == True):
    import wandb
from torch.optim.lr_scheduler import LambdaLR, StepLR

class ModelFD(nn.Module):
    """Custom Pytorch model for gradient optimization.
    """
    def __init__(self, params, fwdModel, device):
        super().__init__()
        self.imgs = torch.tensor(params["imgs"]).to(device)
        self.fInp = torch.tensor(params["fInp"]).to(device)
        self.Nt, self.Ny, self.Nx = self.imgs.shape
        self.Nz = params["Nz"]
        self.numLayersEps = params["numLayersEps"]
        self.numLayersK = params["numLayersK"]
        self.EpsFactor = params["EpsFactor"]
        print("Layers K: ", self.numLayersK)
        print("Layers Eps: ", self.numLayersEps)

        # initialize weights with random numbers - (K, Eps) pairs
        K = 0.5*torch.ones((self.numLayersK, self.Ny, self.Nx)).to(device)
        Eps = 0.5*torch.ones((self.numLayersEps, self.Ny, self.Nx)).to(device)
        Mu = 0.5*torch.zeros((self.Ny, self.Nx)).to(device)
        # make weights torch parameters - consisting of K and Eps
        self.K = nn.Parameter(K)
        self.Eps = nn.Parameter(Eps)
        self.Mu = Mu#nn.Parameter(Mu)

        self.fwdModel = fwdModel
 
    def forward(self):
        """Implement function to be optimised.
        In this case, we predict the surface temperature profiles,
        which is like the image observed by the thermal camera
        """
        pred_imgs_surface_final = self.fwdModel.funcFDSim_torch(self.K, self.Eps, self.fInp, self.Mu)

        return pred_imgs_surface_final

def TVNorm(img, mode="l1"):
    grad_x = img[1:,1:] - img[1:,:-1]
    grad_y = img[1:,1:] - img[:-1,1:]
    if mode=="isotropic":
        return torch.sqrt(grad_x**2 + grad_y**2).mean()
    elif mode=="l1":
        return abs(grad_x).mean() + abs(grad_y).mean()
    return 0
    
def training_loop(model, optimizer, scheduler, Lmd=1.0, total_iterations=1000):
    "Training loop for torch model."
    losses = []
    for i in range(total_iterations):
        print("iter: ", i)
        pred_imgs_surface_all, _ = model()

        # loss 1: predicted profiles = recorded (using MSE loss for LHS-RHS=0)
        loss1 = F.mse_loss(pred_imgs_surface_all, model.imgs.float()).sqrt()
        # loss 2: Total Variation Loss (TV Loss)
        loss2 = TVNorm(model.K[0,:,:], mode='l1')

        if (model.K.shape[0]>1):
            loss2 += TVNorm(model.K[1,:,:], mode='l1')

        loss = loss1 + Lmd * loss2

        loss.backward()

        optimizer.step()
        scheduler.step()
        curr_lr = scheduler.get_last_lr()
        print("Current learning rate: ", curr_lr)
        optimizer.zero_grad()
        losses.append(loss.item())

        if (useWanb == True):
            wandb.log({"loss": loss, "Kcenter: ": model.K[0,model.Ny//2,model.Nx//2],\
                "Kcenter+3: ": model.K[0,model.Ny//2,model.Nx//2+3],\
                "Epscenter: ": model.EpsFactor*model.Eps[0,model.Ny//2,model.Nx//2],\
                "Epscenter+1: ": model.EpsFactor*model.Eps[0,model.Ny//2,model.Nx//2+1]})

    return losses

'''
function: run_optimization(strObjName)
description: use this as a main command for this python file
params: 
    strObjName: string name of the material data (without .pkl)
    bPlotting: Set this to true to enable plotting and saving in wandb
'''
def run_optimization(strObjName, learning_rate = 1e-3, total_iterations=1000, EpsFactor=1.0, bPlotting=False, Description="wandb description"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    ## I. Load the simulated data
    sfilename = "../data/processed/" + strObjName + ".pkl"

    with open(sfilename, 'rb') as handle:
        params = pickle.load(handle)

    imgs = params["imgs"]
    fInp = params["fInp"]

    disp_idx = 6
    stitle="Displaying data, image # "+str(disp_idx)
    
    if (bPlotting == True):
        plt.figure()
        plt.imshow(imgs[disp_idx,:,:])
        plt.title(stitle)
        plt.colorbar()

    ## Unknowns for this optimization problem
    Nt, Ny, Nx = imgs.shape

    params["EpsFactor"] = EpsFactor

    ### II. Instantiate Model
    fwdMod = ForwardFD(params, device)

    model = ModelFD(params, fwdMod, device)
    model = model.to(device)

    ### III. Instantiate optimizer
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    ## Adding a scheduler to vary learning rate
    lambdaF = lambda i: 0.1**min(i/total_iterations, 1)
    scheduler = LambdaLR(opt, lr_lambda=lambdaF)
    # scheduler = StepLR(opt, step_size=25, gamma=0.1)

    ## III.5. WANDB setup
    ## TODO: Make wandb things optional
    if (useWanb == True):
        wandb.init(
            # Set the project where this run will be logged
            project="FD-optim", 
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            name=strObjName, 
            # Track hyperparameters and run metadata
            config={
            "learning_rate": learning_rate,
            "architecture": "Finite Difference",
            "dataset": "Custom",
            "epochs": total_iterations,
            "description": Description,
            "EpsFactor": EpsFactor
        })

    ### IV. Train
    losses = training_loop(model, opt, scheduler, Lmd=0.1, total_iterations=total_iterations)

    ### V. View loss plot
    plt.figure(figsize=(14,7))
    plt.plot(losses)
    plt.ylabel("Loss")
    plt.xlabel("number of iterations")
    plt.title("Loss vs iterations")
    

    ### VI. View more results

    # Sample plotting after some timesteps
    idx = Nt//10

    img_surface_all, img_slice_all = model()
    
    img_thermal = img_surface_all[idx,:,:]
    img = img_thermal.detach().cpu().numpy()

    img_sliceC = img_slice_all[idx,:,:]
    img_slice = img_sliceC.detach().cpu().numpy()

    hw = 10
    if (bPlotting == True):
        # Check if the path exists to write this processed file
        # Create the directory if required
        path = "../results/images/"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)

        plt.figure()
        plt.imshow(img)
        plt.title("Image at timestep="+str(idx+1))
        plt.colorbar()
        fname = "../results/images/Image1.png"
        plt.savefig(fname)
        if (useWanb == True):
            wandb.log({"Image1": wandb.Image(fname)})

        plt.figure()
        plt.imshow(imgs[idx,:,:])
        plt.title("OG Image at timestep="+str(idx+1))
        plt.colorbar()
        fname = "../results/images/Image1OG.png"
        plt.savefig(fname)
        if (useWanb == True):
            wandb.log({"Image1OG": wandb.Image(fname)})

        plt.figure()
        plt.imshow(img[Ny//2-hw:Ny//2+hw+1, Nx//2-hw:Nx//2+hw+1])
        plt.title("Image at timestep="+str(idx+1))
        plt.colorbar()
        fname = "../results/images/Image1zoom.png"
        plt.savefig(fname)
        if (useWanb == True):
            wandb.log({"Image1zoom": wandb.Image(fname)})

        plt.figure()
        plt.imshow(imgs[idx,Ny//2-hw:Ny//2+hw+1, Nx//2-hw:Nx//2+hw+1])
        plt.title("OG Image at timestep="+str(idx+1))
        plt.colorbar()
        fname = "../results/images/Image1OGzoom.png"
        plt.savefig(fname)
        if (useWanb == True):
            wandb.log({"Image1OGzoom": wandb.Image(fname)})

        # View the middle slice through the volume
        plt.figure()
        plt.imshow(img_slice)
        plt.title("Slice at timestep="+str(idx+1))
        plt.colorbar()
        fname = "../results/images/Slice1zoom.png"
        plt.savefig(fname)
        if (useWanb == True):
            wandb.log({"Slice1zoom": wandb.Image(fname)})


    idx = 2*Nt//3
    img_thermal = img_surface_all[idx,:,:]
    img = img_thermal.detach().cpu().numpy()
    img_sliceC = img_slice_all[idx,:,:]
    img_slice = img_sliceC.detach().cpu().numpy()

    if (bPlotting == True):
        plt.figure()
        plt.imshow(img)
        plt.title("Image at timestep="+str(idx+1))
        plt.colorbar()
        fname = "../results/images/Image2.png"
        plt.savefig(fname)
        if (useWanb == True):
            wandb.log({"Image2": wandb.Image(fname)})

        plt.figure()
        plt.imshow(imgs[idx,:,:])
        plt.title("OG Image at timestep="+str(idx+1))
        plt.colorbar()
        fname = "../results/images/Image2OG.png"
        plt.savefig(fname)
        if (useWanb == True):
            wandb.log({"Image2OG": wandb.Image(fname)})

        plt.figure()
        plt.imshow(img[Ny//2-hw:Ny//2+hw+1, Nx//2-hw:Nx//2+hw+1])
        plt.title("Image at timestep="+str(idx+1))
        plt.colorbar()
        fname = "../results/images/Image2zoom.png"
        plt.savefig(fname)
        if (useWanb == True):
            wandb.log({"Image2zoom": wandb.Image(fname)})

        plt.figure()
        plt.imshow(imgs[idx,Ny//2-hw:Ny//2+hw+1, Nx//2-hw:Nx//2+hw+1])
        plt.title("OG Image at timestep="+str(idx+1))
        plt.colorbar()
        fname = "../results/images/Image2OGzoom.png"
        plt.savefig(fname)
        if (useWanb == True):
            wandb.log({"Image2OGzoom": wandb.Image(fname)})

        # View the middle slice through the volume
        plt.figure()
        plt.imshow(img_slice)
        plt.title("Slice at timestep="+str(idx+1))
        plt.colorbar()
        fname = "../results/images/Slice2zoom.png"
        plt.savefig(fname)
        if (useWanb == True):
            wandb.log({"Slice2zoom": wandb.Image(fname)}) 


    idx = 5*Nt//6
    img_thermal = img_surface_all[idx,:,:]
    img = img_thermal.detach().cpu().numpy()
    img_sliceC = img_slice_all[idx,:,:]
    img_slice = img_sliceC.detach().cpu().numpy()

    if (bPlotting == True):
        plt.figure()
        plt.imshow(img)
        plt.title("Image at timestep="+str(idx+1))
        plt.colorbar()
        fname = "../results/images/Image3.png"
        plt.savefig(fname)
        if (useWanb == True):
            wandb.log({"Image3": wandb.Image(fname)})

        plt.figure()
        plt.imshow(imgs[idx,:,:])
        plt.title("OG Image at timestep="+str(idx+1))
        plt.colorbar()
        fname = "../results/images/Image3OG.png"
        plt.savefig(fname)
        if (useWanb == True):
            wandb.log({"Image3OG": wandb.Image(fname)})

        plt.figure()
        plt.imshow(img[Ny//2-hw:Ny//2+hw+1, Nx//2-hw:Nx//2+hw+1])
        plt.title("Image at timestep="+str(idx+1))
        plt.colorbar()
        fname = "../results/images/Image3zoom.png"
        plt.savefig(fname)
        if (useWanb == True):
            wandb.log({"Image3zoom": wandb.Image(fname)})

        plt.figure()
        plt.imshow(imgs[idx,Ny//2-hw:Ny//2+hw+1, Nx//2-hw:Nx//2+hw+1])
        plt.title("OG Image at timestep="+str(idx+1))
        plt.colorbar()
        fname = "../results/images/Image3OGzoom.png"
        plt.savefig(fname)
        if (useWanb == True):
            wandb.log({"Image3OGzoom": wandb.Image(fname)})

        # View the middle slice through the volume
        plt.figure()
        plt.imshow(img_slice)
        plt.title("Slice at timestep="+str(idx+1))
        plt.colorbar()
        fname = "../results/images/Slice3zoom.png"
        plt.savefig(fname)
        if (useWanb == True):
            wandb.log({"Slice3zoom": wandb.Image(fname)})

    # K image - plotting and saving
    K = model.K.detach().cpu().numpy()
    print(K.shape)

    if (bPlotting == True):
        for iter in range(params["numLayersK"]):
            plt.figure()
            plt.imshow(K[iter,:,:])
            stitle = "Layer: " + str(iter+1) + ", K - Diffusivity (scaled)"
            plt.title(stitle)
            plt.colorbar()
            fname = "../results/images/"+"K"+str(iter)+".png"
            plt.savefig(fname)
            if (useWanb == True):
                wandb.log({("K"+str(iter)): wandb.Image(fname)})

            plt.figure()
            hws = 3
            Kzoom = K[iter, Ny//2-hws:Ny//2+1+hws,Nx//2-hws:Nx//2+1+hws]
            plt.imshow(Kzoom)
            stitle = "Layer: " + str(iter+1) + ", average K in central area: "+str(Kzoom.mean())
            plt.title(stitle)
            plt.colorbar()
            fname = "../results/images/"+"Kzoom"+str(iter)+".png"
            plt.savefig(fname)
            if (useWanb == True):
                wandb.log({("Kzoom"+str(iter)): wandb.Image(fname)})


    # EpsP image - plotting and saving
    Eps = model.Eps.detach().cpu().numpy()
    print(Eps.shape)

    ## Also plot while saving
    if (bPlotting == True):
        for iter in range(params["numLayersEps"]):
            plt.figure()
            plt.imshow(Eps[iter,:,:])
            stitle = "Layer: " + str(iter+1) + "Eps*fInp (up to scale)" #+ " x10^-1"
            plt.title(stitle)
            plt.colorbar()
            fname = "../results/images/"+"Eps"+str(iter)+".png"
            plt.savefig(fname)
            if (useWanb == True):
                wandb.log({("Eps"+str(iter)): wandb.Image(fname)})

            hws = 1
            Epszoom = Eps[iter,Ny//2-hws:Ny//2+1+hws,Nx//2-hws:Nx//2+1+hws]
            plt.figure()
            plt.imshow(Epszoom)
            stitle = "Layer: " + str(iter+1) + ", average Eps*fInp in central area: "+str(Epszoom.mean())#+" x10^-1"
            plt.title(stitle)
            plt.colorbar()
            fname = "../results/images/"+"Epszoom"+str(iter)+".png"
            plt.savefig(fname)
            if (useWanb == True):
                wandb.log({("Epszoom"+str(iter)): wandb.Image(fname)})


    if (useWanb == True):
        wandb.finish()

    saveProcData(strObjName, losses, model, params)


# Save the required results in a pickle file
def saveProcData(strObjName, losses, model, params):
    saveD = {}
    saveD["losses"] = losses
    K = model.K.detach().cpu().numpy()
    Eps = model.Eps.detach().cpu().numpy()
    Mu = model.Mu.detach().cpu().numpy()
    
    saveD["K"] = K
    saveD["Eps"] = Eps
    saveD["Mu"] = Mu
    saveD["numLayersEps"] = params["numLayersEps"]
    saveD["numLayersK"] = params["numLayersK"]

    # Check if the path exists to write this processed file
    # Create the directory if required
    path = "../results/"
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)

    ## Save the data for future use
    sfilename = "../results/" + strObjName + "_results.pkl"
    with open(sfilename, 'wb') as handle:
        pickle.dump(saveD, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return

# Load function for extracting results
def loadProcessedData(strObjName, EpsFactor=1.0, bPlotting=False):
    sfilename = "../results/" + strObjName + "_results.pkl"
    with open(sfilename, 'rb') as handle:
        data = pickle.load(handle)

    K = data["K"]
    Eps = data["Eps"]
    numLayersK, Ny, Nx = K.shape
    numLayersEps = data["numLayersEps"]
    # print(numLayersK, numLayersEps, Ny, Nx)

    try:
        T_init = data["T_init"]
    except:
        try:
            sfilename = "../data/image_labdata_cropped_" + strObjName + ".pkl"
            with open(sfilename, 'rb') as handle:
                params = pickle.load(handle)
            T_init = params["T_init"]
        except:
            T_init = 300

    if (bPlotting == True):
        plt.figure()
        plt.imshow(K[0,:,:])
        plt.title("Diffusivity loaded")
        plt.colorbar()

        iter = 0
        hws = 3
        Kzoom = K[iter, Ny//2-hws:Ny//2+1+hws,Nx//2-hws:Nx//2+1+hws]
        stitle = "Layer: " + str(iter+1) + ", average K in central area: "+str(Kzoom.mean())
        plt.figure()
        plt.imshow(Kzoom)
        plt.title(stitle)
        plt.colorbar()

        plt.figure()
        plt.imshow(Eps[0,:,:])
        plt.title("Eps*fInp loaded")
        plt.colorbar()

        hws = 1
        Epszoom = Eps[iter, Ny//2-hws:Ny//2+1+hws,Nx//2-hws:Nx//2+1+hws]
        stitle = "Layer: " + str(iter+1) + ", average Eps*fInp in central area: "+str(Epszoom.mean())
        plt.figure()
        plt.imshow(Epszoom)
        plt.title(stitle)
        plt.colorbar()

    hwK = 3
    Kzoom = K[0, Ny//2-hwK:Ny//2+1+hwK,Nx//2-hwK:Nx//2+1+hwK]
    K_collection = []
    K_collection.append(Kzoom.mean())

    hwE = 1
    Epszoom = Eps[0, Ny//2-hwE:Ny//2+1+hwE,Nx//2-hwE:Nx//2+1+hwE]
    Eps_collection = []
    Eps_collection.append(Epszoom.mean())

    
    cy = Ny//2
    cx = Nx//2
    hWin = 2
    features_K = K[0, cy-hWin:cy+hWin+1, cx-hWin:cx+hWin+1]
    features_Eps0 = Eps[0, cy-hWin:cy+hWin+1, cx-hWin:cx+hWin+1]
    # features_Eps1 = Eps[1, cy-hWin:cy+hWin+1, cx-hWin:cx+hWin+1]
    K_Eps_center = np.array([K[0, cy, cx], Eps[0, cy, cx], Eps[1, cy, cx]])

    features = np.append(K_Eps_center, features_K.flatten())
    features = np.append(features, features_Eps0.flatten())
    # features = np.append(features, features_Eps1.flatten())
    

    '''
    features = np.array([K[0,Ny//2,Nx//2], K[0,Ny//2+1,Nx//2], K[0,Ny//2-1,Nx//2],
                         K[0,Ny//2,Nx//2+1], K[0,Ny//2,Nx//2-1],
                         Eps[0,Ny//2,Nx//2], Eps[0,Ny//2+1,Nx//2], Eps[0,Ny//2-1,Nx//2],
                         Eps[0,Ny//2+2,Nx//2], Eps[0,Ny//2-2,Nx//2]])
    '''
    
    return K, Eps, numLayersK, numLayersEps, features
