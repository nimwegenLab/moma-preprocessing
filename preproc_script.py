import os, re, time, sys, shutil
import numpy as np
import matplotlib.pyplot as plt

import scipy.optimize
import scipy.ndimage.interpolation as ndii
import skimage.external.tifffile
import skimage.transform
import skimage.measure
import skimage.filters
from skimage.feature import match_template

from colicycle.MMdata import MMData

import colicycle.preprocessing as pre
import colicycle.imreg as imreg

data_folder = sys.argv[1]
folder_to_save = sys.argv[2]
positions = [int(sys.argv[3])]
maxframe = int(sys.argv[4])

print("Input folder:")
print(data_folder)
print("Output folder:")
print(folder_to_save)
print("Positions:")
print(str(positions[0]))
print("Max frames:")
print(str(maxframe))

dataset = MMData(data_folder)
base_name = dataset.get_first_tiff().split('.')[0]

folder_analyzed = os.path.normpath(folder_to_save)+'/'+base_name

kymo_folder = folder_analyzed+'/Kymographs/'
if not os.path.exists(kymo_folder):
    os.makedirs(kymo_folder)
    
GL_folder = folder_analyzed+'/GrowthLanes/'
if not os.path.exists(GL_folder):
    os.makedirs(GL_folder)

colors = dataset.get_channels()
phase_channel_index = 0

half_width = 50


#define metadata for imagej
metadata = {'channels':len(colors),'slices':1,'frames':maxframe,'hyperstack':True,'loop':False}

start1 = time.time()
for indp in positions:

    current_saveto_folder = folder_to_save+'/'+base_name+'/GrowthLanes/'+base_name+'_Pos'+str(indp)

    if not os.path.exists(current_saveto_folder):
        os.makedirs(current_saveto_folder)

    #load first phase image
    image_base = dataset.get_image_fast(channel=phase_channel_index,frame=0,position=indp)
    
    #find rotation and channels
    image_rot, angle, mincol, maxcol, channel_centers = pre.split_channels_init(image_base)
    
    '''#find region of numbers as high variance region
    numbers_var = np.var(image_base[:,0:mincol-50],axis = 1)
    window = 100
    peaks = np.array([x for x in np.arange(window,len(numbers_var)-window) 
                      if np.all(numbers_var[x]>numbers_var[x-window:x])&np.all(numbers_var[x]>numbers_var[x+1:x+window])])
    peaks = peaks[numbers_var[peaks]>100*np.min(numbers_var)]
    highvar = peaks[np.argmin(peaks-image_base.shape[0]/2)]
    templ = image_base[highvar-100:highvar+100,int(mincol/2)-100:int(mincol/2)+100]'''
    
    #find regions with large local derivatives in BOTH directions, which should be "number-regions".
    #keep the one the most in the middle
    #take a sub-region around that point as a template on which to do template matching
    large_feature = skimage.filters.gaussian(np.abs(skimage.filters.scharr_h(image_base)*skimage.filters.scharr_v(image_base)),sigma=5)
    mask = np.zeros(image_base.shape)
    mask[large_feature>0.5*np.max(large_feature)]=1
    mask_lab = skimage.measure.label(mask)
    mask_reg = skimage.measure.regionprops(mask_lab)
    middle_num_pos = mask_reg[np.argmin([np.linalg.norm(np.array(x.centroid)-np.array(image_base.shape)/2) for x in mask_reg])].centroid
    mid_row = np.int(middle_num_pos[0])
    
    hor_space = int(mincol)+100
    hor_mid = int(hor_space/2)
    hor_width = int(0.3*hor_space)
    
    templ = image_base[mid_row-100:mid_row+100,0:hor_space]
    
    #create empty kymographs to fill
    kymo = np.zeros((maxcol-mincol+60,maxframe, len(colors), len(channel_centers)))
    metadataK = {'channels':len(colors),'slices':1,'frames':len(channel_centers),'hyperstack':True,'loop':False}


    #calculate channel spacing 
    fourier_ch = np.abs(np.fft.fft(np.sum(image_base[:,mincol:maxcol],axis=1)))
    fourier_sort = np.sort(fourier_ch)
    channel_spacing = image_base.shape[0]/np.where(fourier_ch==fourier_sort[-2])[0][0]

    frame_counter = np.zeros(len(channel_centers))
    #go through time-lapse and cut out channels
    for t in range(maxframe):

        if np.mod(t,10)==0:
            print('time: '+str(t))
        #load image and align
        image = dataset.get_image_fast(channel=phase_channel_index,frame=t,position=indp)
        
        '''#register images using FFT. Sometimes fails, probably because of border effect
        mindim = np.int(np.min([int(image.shape[0]/3),mincol-50])/2)
        t0, t1 = pre.fft_align(image_base[0:int(image.shape[0]/3),0:mincol-50],
                                   image[0:int(image.shape[0]/3),0:mincol-50],pixlim =mindim)
        image = np.roll(image,(t0,t1),axis=(0,1))
        t0b, t1b = pre.fft_align(image_base[0:int(image.shape[0]/3),mincol-50:mincol+200],
                                   image[0:int(image.shape[0]/3),mincol-50:mincol+200],pixlim=20)
        image = np.roll(image,(t0b,t1b),axis=(0,1))
        t0 = t0+t0b
        t1 = t1+t1b
        '''
        
        #do template matching of the "number-region"
        tomatch = image[mid_row-50:mid_row+50,hor_mid-hor_width:hor_mid+hor_width]

        result = match_template(templ, tomatch, pad_input=True)
        ij = np.unravel_index(np.argmax(result), result.shape)
        t1, t0 = ij[::-1]
        t0 = int(t0-templ.shape[0]/2)
        t1 = int(t1-templ.shape[1]/2)
        image = np.roll(image,(t0,t1),axis=(0,1))

        #rotate image
        image_rot = skimage.transform.rotate(image,angle,cval=0)

        #find channels in new image
        channels = pre.find_channels(image_rot, mincol, maxcol)

        #load all colors
        image_stack = np.zeros((image_base.shape[0],image_base.shape[1],len(colors)))
        for i in range(len(colors)):
            image = dataset.get_image_fast(channel=i,frame=t,position=indp)
            image = np.roll(image,(t0,t1),axis=(0,1))
            image_rot = skimage.transform.rotate(image,angle,cval=0)
            image_stack[:,:,i] = image_rot

        #go through all channels, check if there's a corresponding one in the new image. If yes go through all colors,
        #cut out channel, and append to tif stack. Completel also the kymograph for each color.
        for c in channels:
            if (np.min(c-channel_centers)<5)&(int(c)-half_width>0)&(int(c)+half_width+1<image.shape[0]):
                gl = np.argmin(np.abs(c-channel_centers))
                frame_counter[gl]+=1
                gl_str='0'+str(gl) if gl<10 else str(gl)
                pos_gl_name = dataset.get_first_tiff().split('.')[0]+'_Pos'+str(indp)+'_GL'+gl_str
                
                if not os.path.exists(current_saveto_folder+'/'+pos_gl_name):
                    os.makedirs(current_saveto_folder+'/'+pos_gl_name)
        
                filename = current_saveto_folder+'/'+pos_gl_name+'/'+pos_gl_name+'.tif'

                for i in range(len(colors)):
                    imtosave = image_stack[:,:,i][int(c)-half_width:int(c)+half_width+1,mincol-30:maxcol+30]
                    imtosave_flip = np.flipud(imtosave.T)
                    skimage.external.tifffile.imsave(filename,imtosave_flip.astype(np.uint16),append = 'force',imagej = True, metadata = metadata)
                    kymo[:,t,i,gl] = np.mean(imtosave, axis = 0)
    
    #remove growth lanes that don't have all time points (e.g. because of drift)
    incomplete_GL = np.where(frame_counter<maxframe)[0]
    for inc in incomplete_GL:
        gl_str='0'+str(inc) if inc<10 else str(inc)
        pos_gl_name = dataset.get_first_tiff().split('.')[0]+'_Pos'+str(indp)+'_GL'+gl_str
        filename = current_saveto_folder+'/'+pos_gl_name
        if os.path.exists(filename):
            shutil.rmtree(filename)
    
    print(incomplete_GL)
    #save kymograph
    for gl in range(len(channel_centers)):
        if gl not in incomplete_GL:

            for c in range(len(colors)):
                filename = kymo_folder+'/'+dataset.get_first_tiff().split('.')[0]+'_Pos'+str(indp)+'_GL'+str(gl)+'_col'+str(c)+'_kymo.tif'
                skimage.external.tifffile.imsave(filename,kymo[:,:,c,gl].astype(np.uint16),append = 'force',imagej = True, metadata = metadataK)



end1 = time.time()
print(end1 - start1)

