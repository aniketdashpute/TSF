import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

def processData(strObjName, params, bPlotting=False):
    print("----------")
    print("Now analysing: ", strObjName)
    print("----------")

    numLayersK = params.numLayersK
    numLayersEps = params.numLayersEps
    Nz = params.Nz
    total_time_captured = params.total_time_captured
    total_time = params.total_time
    numCycles = params.numCycles
    timeCycleON = params.timeCycleON
    time_ON = params.time_ON
    tremove = params.tremove
    delTimgf = params.delTimgf
    strFolderName = params.strCommon + strObjName
    
    ## Part 0. Convert data into single npy file
    results_dir = "../data/"+strFolderName+"/"
    lst = os.listdir(results_dir) # your directory path
    number_files = len(lst)
    N_images = number_files-1
    print("N_images: ", N_images)

    img0 = np.load(results_dir+"image_data20.npy")
    height, width = img0.shape

    if (bPlotting == True):
        plt.figure()
        plt.imshow(img0)
        plt.colorbar()

    someFig = N_images//2-2

    # Check if the path exists to write this processed file
    # Create the directory if required
    path = "../data/processed/"
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    
    sfilename = "../data/processed/" + strObjName + ".pkl"

    bProcessResults = False
    if (img0[height//2, width//2]<100): bProcessResults = True

    if (bProcessResults == True):
        ## If the temperature is in C, convert to K
        ## Needed only once
        U_all = np.zeros([N_images, height, width])
        for iter in range(N_images):
            filename = results_dir+"/image_data" + str(iter) + ".npy"
            img = np.load(filename)
            img = img + 273.15
            U_all[iter,:,:] = img
        
        bProcessResults = False

    ## Part I. Load the data
    imgsOG = U_all

    # Remove some initial data before laser starts
    tsStart = int(np.rint(imgsOG.shape[0]*tremove/total_time_captured))

    imgs = imgsOG
    tsTotal = int(np.rint(imgsOG.shape[0]*total_time/total_time_captured))
    print(tsTotal)
    # example: tsTotal/tsTotalCaptured = 40/60 -> taking first 40 seconds instead of 60
    # because data becomes redundant and noisy after that
    imgs = imgsOG[tsStart:tsTotal,:,:]
    Nt, Ny, Nx = imgs.shape


    imgNum2time = total_time_captured/(imgsOG.shape[0]) ## Because data captured over 60 seconds
    tsON = int(np.rint(time_ON/imgNum2time))
    print("timesteps ON: ", tsON)

    # Deltas
    delT = total_time_captured/Nt # seconds
    delX, delY, delZ = params.delX, params.delY, params.delZ #mm

    if (bPlotting == True):
        plt.figure()
        plt.imshow(imgs[someFig,:,:])
        plt.title("An intermediate image of temperature distribution")
        plt.colorbar()

    ## Part II. Getting the center point manually
    ## Part a. Obtain the center point
    xlowlim, xhighlim = 300, 400
    ylowlim, yhighlim = 240, 360
    ylen = yhighlim - ylowlim
    image = imgs[someFig,ylowlim:yhighlim,xlowlim:xhighlim]
    maxP = image.argmax()
    print("x: ", (maxP%image.shape[1]+xlowlim), " y: ", (maxP/image.shape[1]+ylowlim))

    ## Part b. Plot the center point
    xint = [int(maxP%image.shape[1])+xlowlim, int(np.floor(maxP/image.shape[1]))+ylowlim]
    print("Chosen xint: ", xint)
    Cy, Cx = xint[1], xint[0]
    if (bPlotting == True):
        plt.figure()
        plt.imshow(imgs[someFig,:,:])
        plt.title("Display a sample image for reference")
        plt.colorbar()
        plt.plot()
        plt.plot(xint[0], xint[1], "xk", markersize=1, mew=3)

    if (bPlotting == True):
        plt.figure()
        plt.plot(imgs[:,Cy,Cx], label="Center pixel")
        plt.plot(imgs[:,Cy,Cx+5], label="Center pixel+5")
        plt.plot(imgs[:,Cy,Cx-5], label="Center pixel-5")
        plt.legend()
        plt.title("Plotting temporal variations at some points")

    ## Part III. 
    # Crop the image set to keep the center and a window around it
    hwSize = 50 # Half Window Size
    totalSize = 2 * hwSize + 1
    imgs_crop = imgs[:,Cy-hwSize:Cy+hwSize+1, Cx-hwSize:Cx+hwSize+1]

    if (bPlotting == True):
        plt.figure()
        plt.imshow(imgs_crop[someFig,:,:])
        plt.title("Intermediate image - zoomed")
        plt.colorbar()
        plt.plot()

    if (bPlotting == True):
        plt.figure()
        plt.imshow(imgs_crop[someFig,:,:])
        plt.title("Sample image with center marked")
        plt.colorbar()
        plt.plot()
        plt.plot(hwSize, hwSize, "xk", markersize=1, mew=3)

    if (bPlotting == True):
        plt.figure()
        x_labels = np.arange(Nt)*imgNum2time
        plt.plot(x_labels, imgs_crop[:,hwSize,hwSize], label="Center pixel")
        plt.plot(x_labels, imgs_crop[:,hwSize,hwSize+5], label="Center pixel+5")
        plt.plot(x_labels, imgs_crop[:,hwSize,hwSize-5], label="Center pixel-5")
        plt.legend()
        plt.xlabel('time (s)')
        plt.ylabel('Temperature (K)')
        plt.title("Plotting temporal variations at some points")

    T_init = imgs_crop[0,hwSize,hwSize-5] # K
    print("T_init: ", T_init)

    print("FINALLY GOING IN: imgs_crop.size(): ", imgs_crop.shape)

    outParams = {}
    outParams["ObjectName"] = strObjName
    outParams["imgs"] = imgs_crop
    outParams["Cx"] = Cx
    outParams["Cy"] = Cy
    outParams["time_ON"] = time_ON
    outParams["total_time_captured"] = total_time_captured
    outParams["total_time"] = total_time
    outParams["timestepsON"] = tsON
    outParams["delX"] = delX
    outParams["delY"] = delY
    outParams["delZ"] = delZ
    outParams["delT"] = delT
    outParams["T_init"] = T_init
    Nt, Ny, Nx = imgs_crop.shape
    outParams["Nt"] = Nt
    outParams["Nx"] = Nx
    outParams["Ny"] = Ny
    outParams["Nz"] = Nz
    outParams["numLayersK"] = numLayersK
    outParams["numLayersEps"] = numLayersEps

    outParams["delT"] = delT/delTimgf
    outParams["delTimgf"] = delTimgf

    outParams["imgNum2time"] = outParams["total_time"]/outParams["Nt"]
    outParams["numCycles"] = numCycles
    outParams["tsCycleON"] = int(np.rint(timeCycleON/outParams["imgNum2time"]))
    outParams["sfilename"] = sfilename

    print("Time ON timesteps: ", outParams["tsCycleON"])

    # Finally save all the processed data
    saveLabData(outParams, bPlotting)

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
    ## Use Gaussian profile if necessary
    fInp = np.zeros([Nz, Ny, Nx])
    halfWindow = 5 # pixels, windowsie = 2*halfWindow+1
    fluxIntensity = 1
    depthPenetration = 5

    # laser_i, laser_j = Ny//2, Nx//2
    # size = halfWindow*2 + 1
    # sigmaBeam = 0.5
    # sigma = sigmaBeam
    # G = fluxIntensity*gkern(size, sigma)
    # fInp[:depthPenetration, laser_i-halfWindow:laser_i+halfWindow+1, laser_j-halfWindow:laser_j+halfWindow+1] = G

    laser_i, laser_j = Ny//2, Nx//2
    fInp[:depthPenetration, laser_i-halfWindow:laser_i+halfWindow+1, laser_j-halfWindow:laser_j+halfWindow+1] = fluxIntensity
    
    return fInp

## Save the processed data
def saveLabData(imgs_save, bPlotting=False):
    Nz, Ny, Nx = imgs_save["Nz"], imgs_save["Ny"], imgs_save["Nx"]
    fInp = getFINP(Nz, Ny, Nx)
    imgs_save["fInp"] = fInp
    
    if (bPlotting == True):
        plt.figure()
        plt.imshow(fInp[0,:,:])
        plt.title("fInp")
        plt.colorbar()

    imgs_crop = imgs_save["imgs"]
    print("imgs_crop.shape: ", imgs_crop.shape)
    
    if (bPlotting == True):
        plt.figure()
        plt.imshow(imgs_crop[5,:,:])
        plt.title("Early stage figure for reference")
        plt.colorbar()

    sfilename = imgs_save["sfilename"]
    ## Save the data for future use
    with open(sfilename, 'wb') as handle:
        pickle.dump(imgs_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

## Load the processed data
def loadLabData(sfilename):
    with open(sfilename, 'rb') as handle:
        data_saved = pickle.load(handle)

    im_all = data_saved["imgs"]

    Nt, Ny, Nx = im_all.shape

    plt.figure()
    plt.imshow(im_all[2*Nt//5,:,:])
    plt.title("Displaying intermediate images")
    plt.colorbar()

    plt.figure()
    plt.imshow(im_all[3*Nt//5,:,:])
    plt.title("Displaying intermediate images")
    plt.colorbar()

    return data_saved