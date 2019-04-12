from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Reshape, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import pandas as  pd

from skimage.measure import label, regionprops
from skimage import morphology, segmentation
import scipy

import trackpy

from . import time_mat_operations as tmo


K.set_image_data_format('channels_last')  # TF dimension ordering in this code
smooth = 1.
output = None

#definition of the dice_coefficient to estimate quality of training for binary segmentation
def dice_coef(y_true, y_pred):
    smooth = 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


#definition of the classical Unet architecture (https://arxiv.org/abs/1505.04597)
#for compilation (last line), an argument sample_weight_mode='temporal' is used. This is 
#what allows to use pixel weigths for the training to learn especially well cell boundaries as done
#in the original paper.
def get_unet(dims,img_rows,img_cols):
    inputs = Input((img_rows, img_cols, dims))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    
    conv11 = Reshape((img_rows*img_cols,1),input_shape=(img_rows,img_cols,1))(conv10)

    model = Model(inputs=[inputs], outputs=[conv11])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef], sample_weight_mode='temporal')
        
    return model

#Python generator yielding a triplet of arrays (images, masks, weights) where 
#each array is a bactch of images
def train_generator(folder, numtot):
    num = 0
    while num > -1:
        img_load = np.load(folder+'/imgs_train_'+str(num)+'.npy')
        mask_load = np.load(folder+'/imgs_mask_train_'+str(num)+'.npy')
        weight_load = np.load(folder+'/imgs_weight_train_'+str(num)+'.npy')
        img_load = img_load.astype('float32')

        #weight_load[:]=1
        #img_load = img_load-np.mean(img_load)
        #img_load = img_load/np.std(img_load)
        #for i in range(img_load.shape[0]):
        #    img_load[i,:,:,:] = img_load[i,:,:,:]*np.random.rand()*2
        
        
        mask_load = mask_load[..., np.newaxis]
        mask_load = mask_load.astype('float32')
        #mask_load /= 255.  # scale masks to [0, 1]
    
        weight_load = weight_load.astype('float32')
    
        yield (img_load,mask_load,weight_load)
        num += 1
        if num==numtot:
            num=0

#Python generator yielding a triplet of arrays (images, masks, weights) where 
#each array is a bactch of images
def valid_generator(folder, numtot):
    num = 0
    while num > -1:
        img_load = np.load(folder+'/imgs_valid_'+str(num)+'.npy')
        mask_load = np.load(folder+'/imgs_mask_valid_'+str(num)+'.npy')
        weight_load = np.load(folder+'/imgs_weight_valid_'+str(num)+'.npy')
        img_load = img_load.astype('float32')
        
        #weight_load[:]=1
        #img_load = img_load-np.mean(img_load)
        #img_load = img_load/np.std(img_load)
        #for i in range(img_load.shape[0]):
        #    img_load[i,:,:,:] = img_load[i,:,:,:]*np.random.rand()*2
            
            
        mask_load = mask_load[..., np.newaxis]
        mask_load = mask_load.astype('float32')
        #mask_load /= 255.  # scale masks to [0, 1]
    
        weight_load = weight_load.astype('float32')
    
        yield (img_load,mask_load,weight_load)
        num += 1
        if num==numtot:
            num=0
            
#training of the CNN with all images saved in a single array
#folder: path to training data
#img_rows, img_cols: image dimensions
#dims: "third" dimension of the images (e.g. 3 for RGB)
#batch_size: batch size for training
#epochs: number of training epochs
#weights: path to existing weights file
def coli_mom_deeptrain(folder, img_rows,img_cols, dims, batch_size = 32, epochs = 100, weights = None):

    #load training data
    imgs_train, imgs_mask_train, imgs_weight_train  = load_train_data(folder)
    imgs_train = imgs_train.astype('float32')

    imgs_mask_train = imgs_mask_train[..., np.newaxis]
    imgs_mask_train = imgs_mask_train.astype('float32')
    #imgs_mask_train /= 255.  # scale masks to [0, 1]
    
    imgs_weight_train = imgs_weight_train.astype('float32')

    #create u-net model
    coli_model = get_unet(dims,img_rows,img_cols)
    model_checkpoint = ModelCheckpoint(folder+'weights.h5', monitor='val_loss', save_best_only=True)
    
    #load weights if provided
    if weights:
        coli_model.load_weights(weights)
    
    #run training
    coli_model.fit(imgs_train, imgs_mask_train, batch_size=batch_size, epochs=epochs, verbose=1, shuffle=True,
              validation_split=0.2,sample_weight = imgs_weight_train,
              callbacks=[model_checkpoint])
    


    '''imgs_test, imgs_id_test = load_test_data(folder)

    imgs_test = imgs_test.astype('float32')
    
    plate_model.load_weights(folder+'weights.h5')

    imgs_mask_test = plate_model.predict(imgs_test, verbose=1)
    #np.save('imgs_mask_test.npy', imgs_mask_test)

    test_dir = folder+'test'
    if not os.path.exists(test_dir):
        os.mkdir(pred_dir)
    for image, image_id in zip(imgs_mask_test, imgs_id_test):
        image = np.reshape(image,(img_rows,img_cols))
        image = (image * 255.).astype(np.uint8)
        imsave(os.path.join(test_dir, str(image_id) + '_pred.png'), image)'''


#training of the CNN with images saved in batches and recovered with generators
#folder: path to training data
#img_rows, img_cols: image dimensions
#dims: "third" dimension of the images (e.g. 3 for RGB)
#batch_size: batch size for training
#epochs: number of training epochs
#weights: path to existing weights file
def coli_mom_deeptrain_batches(folder, img_rows,img_cols, dims, train_batch_nb = 230, validation_batch_nb = 58, epochs = 100, weights = None):

    #create u-net model
    coli_model = get_unet(dims,img_rows,img_cols)
    model_checkpoint = ModelCheckpoint(folder+'weights.h5', monitor='val_loss', save_best_only=True)
    
    #load weights if provided
    if weights:
        coli_model.load_weights(weights)
    
    #run training with function dedicated to batches
    coli_model.fit_generator(train_generator(folder, train_batch_nb), steps_per_epoch=train_batch_nb, epochs=epochs,
                              validation_data=valid_generator(folder, validation_batch_nb),validation_steps = validation_batch_nb,verbose=1,
                              callbacks=[model_checkpoint])


        
def load_train_data(folder):
    imgs_train = np.load(folder+'imgs_train.npy')
    imgs_mask_train = np.load(folder+'imgs_mask_train.npy')
    imgs_weight_train = np.load(folder+'imgs_weight_train.npy')
    return imgs_train, imgs_mask_train, imgs_weight_train

def load_test_data(folder):
    imgs_test = np.load(folder+'imgs_test.npy')
    imgs_id = np.load(folder+'imgs_id_test.npy')
    return imgs_test, imgs_id


def find_phase_channel_limits(mom):
    
    image = mom.load_moma_im()
    back= np.mean(image[:,np.sum(image==0,axis=0)<image.shape[0]/2][int(image.shape[0]/2)::,0:5])
    backstdv = np.std(image[:,np.sum(image==0,axis=0)<image.shape[0]/2][int(image.shape[0]/2)::,0:5])
    mid_im = int(image.shape[1]/2)
    #topad = 512-image.shape[0]
    limtop = np.where(np.mean(image[:,np.sum(image==0,axis=0)<image.shape[0]/2][:,0:15],axis=1)>back+5*backstdv)[0][-1]#+topad
    limbottom = np.where(np.mean(image[:,mid_im-2:mid_im+3],axis = 1)>back+5*backstdv)[0][-1]#+topad

    return np.array([limtop,limbottom])

def deep_single_image(momobj, weights_file, normalize = True):
    im_height = 512
    im_width = 32
    
    
    #create model and load weights
    model = get_unet(1,im_height,im_width)
    model.load_weights(weights_file)

    image = momobj.load_moma_im()
    mid_im = int(image.shape[1]/2)
    topad = im_height-image.shape[0]

        
    if normalize:
        #image = image-1.0*np.mean(image)
        #image = image/np.std(image)
        
        image = image-np.mean(image[150:400,mid_im-2:mid_im+3])
        image = image/np.std(image[150:400,mid_im-2:mid_im+3])

    image = np.pad(image,((topad,0),(0,0)),'constant')
    image = image[:,mid_im-int(im_width/2):mid_im+int(im_width/2)]

    imgs_test = image.astype('float32')

    imgs_test = imgs_test[np.newaxis,...,np.newaxis]
    if output:
        imgs_mask_test = model.predict(imgs_test, verbose=1)
    else:
        imgs_mask_test = model.predict(imgs_test, verbose=0)

    imgs_mask_test = np.reshape(imgs_mask_test,image.shape)

    imgs_mask_test = imgs_mask_test[topad::,:]

    return imgs_mask_test
        


def deep_map(momobj, weights_file, min_time=0, max_time=None,limits = None, normalize = True, projection = 'max',threshold = 0.9, show_plot = False):
    
    all_coords = []
    
    if not max_time:
        max_time = momobj.get_max_time()
        
    im_height = 512
    im_width = 32
    
    #image = momobj.load_moma_im()
    topad = im_height-momobj.height
    
    #create model and load weights
    model = get_unet(1,im_height,im_width)
    model.load_weights(weights_file)

    #create deep learned kymograph
    tot_time = max_time-min_time+1
    segmNN = np.zeros((im_height,tot_time))
    segm_min = np.zeros((im_height,tot_time))
    segm_max = np.zeros((im_height,tot_time))
    time_count = 0
    for t in range(min_time,max_time+1):
        if output:
            print('time: '+str(t))
        momobj.time = t
        image = momobj.load_moma_im()
        mid_im = int(image.shape[1]/2)
        
        factor = 1
        if normalize:
            #image = image-1.0*np.mean(image)
            #image = image/np.std(image)
            
            image = image-np.mean(image[150:400,mid_im-2:mid_im+3])
            image = image/np.std(image[150:400,mid_im-2:mid_im+3])

        image = np.pad(image,((topad,0),(0,0)),'constant')
        image = image[:,mid_im-int(im_width/2):mid_im+int(im_width/2)]*factor

        imgs_test = image.astype('float32')

        imgs_test = imgs_test[np.newaxis,...,np.newaxis]
        if output:
            imgs_mask_test = model.predict(imgs_test, verbose=1)
        else:
            imgs_mask_test = model.predict(imgs_test, verbose=0)
            
        imgs_mask_test = np.reshape(imgs_mask_test,image.shape)
        
        '''#standard
        imgs_mask = imgs_mask_test.copy()
        imgs_mask[imgs_mask<threshold]=0
        imgs_mask[imgs_mask>=threshold]=1
        imgs_mask = morphology.binary_opening(imgs_mask, morphology.disk(2)).astype(int)
        imgs_label = label(imgs_mask)'''
        
        #using watershed
        mapped_th = np.zeros(imgs_mask_test.shape)
        mapped_th2 = np.zeros(imgs_mask_test.shape)
        mapped_th[imgs_mask_test>threshold]=1
        mapped_th = morphology.binary_opening(mapped_th, morphology.disk(2)).astype(int)
        mapped_th2[imgs_mask_test>0.01]=1
        mapped_th2 = morphology.binary_opening(mapped_th2, morphology.disk(3)).astype(int)
        
        mapped_lab = label(mapped_th)
        imgs_label = morphology.watershed(image,mapped_lab,mask = mapped_th2,watershed_line=True)

        
        cell_info = regionprops(imgs_label)

        newMask = np.zeros(imgs_label.shape)
        for x in cell_info:
            if np.abs(x.centroid[1]-im_width/2)<10:
                newMask[x.coords[:,0],x.coords[:,1]]=1
                all_coords.append([t, x.coords[:,0].max()-x.coords[:,0].min(), x.centroid[0],
                                  x.coords[:,0].min(),x.coords[:,0].max()])
        
        if show_plot:
            fig,ax = plt.subplots(figsize=(20,20))
            plt.subplot(1,2,1)
            plt.imshow(image, cmap = 'gray')
            plt.imshow(newMask, cmap = 'Reds', alpha = 0.1)
            plt.subplot(1,2,2)
            plt.imshow(image, cmap = 'gray')
            plt.show()
        
        if projection == 'max':
            segmNN[:,time_count] = np.max(imgs_mask_test,axis = 1)
        elif projection == 'sum':
            segmNN[:,time_count] = np.sum(imgs_mask_test,axis = 1)
        elif projection == 'labelled':
            for x in cell_info:
                if np.abs(x.centroid[1]-im_width/2)<10:
                    #label kymograph
                    segm_min[x.coords[:,0].min():x.coords[:,0].max(),time_count] = x.coords[:,0].min()-topad
                    segm_max[x.coords[:,0].min():x.coords[:,0].max(),time_count] = x.coords[:,0].max()-topad
                    
                    tempcol = segmNN[:,time_count].copy()
                    tempcol[x.coords[:,0].min():x.coords[:,0].max()]+=1
                    if np.any(tempcol==2):
                        tempcol[tempcol==2] = 0
                        segmNN[:,time_count] = tempcol                        
                    else:
                        segmNN[x.coords[:,0].min()+1:x.coords[:,0].max()-1,time_count]+=1
        else:    
            print('unknown projection')
            break
        time_count+=1
    
    all_coords = np.array(all_coords)
    all_coords[:,[2,3,4]] = all_coords[:,[2,3,4]]-topad
    segmNN = segmNN[topad::,:]
    segm_min = segm_min[topad::,:]
    segm_max = segm_max[topad::,:]
    
    
    if limits is None:
        return segmNN, segm_min, segm_max, all_coords
    else:
        return segmNN[limits[0]:limits[1],:], all_coords
    
    
def deep_segment(mom, kymo, kymo_min, kymo_max):
    
    lims = find_phase_channel_limits(mom)
    
    kymo_max[kymo_max>lims[1]]=lims[1]
    
    segm_binary = kymo.copy()
    segm_part = kymo.copy()

    lowerpart = 1-segm_binary[lims[1]-20::,:]
    lowerpart_lab = morphology.label(lowerpart)
    lowerpart_clear = segmentation.clear_border(lowerpart_lab)

    sizes = np.bincount(lowerpart_clear.ravel())
    mask_sizes = sizes < 100
    mask_sizes[0] = 0
    tosuppress = mask_sizes[lowerpart_clear]
    lowerpart[tosuppress]=0
    segm_binary[lims[1]-20::,:] = 1-lowerpart

    segm_binary_lab = morphology.label(segm_binary)
    sizes = np.bincount(segm_binary_lab.ravel())
    mask_sizes = sizes < 100
    tosuppress = mask_sizes[segm_binary_lab]
    segm_binary[tosuppress]=0

    #segm_binary[lims[1]::,:]=0


    for i in range(segm_binary.shape[1]):
        init = np.where(segm_binary[:,i]==1)[0][-1]
        segm_binary[init-10:init,i]=1

    #for each time point find cell regions and calculate their properties
    lim=3
    int_lim = 0.6
    all_pos = []
    for r in range(segm_binary.shape[1]):
        lane_label = label(segm_binary[:,r])
        #suppress the top cell it it touches the image border to avoid getting 
        #cut cells
        if lane_label[0]==1:
            lane_label = lane_label-1
            lane_label[lane_label==-1]=0

        #measure features of cell regions
        seg_sum = np.array([np.sum(lane_label==i+1) for i in range(lane_label.max())])
        seg_mean = np.array([np.median(segm_part[:,r][lane_label==i+1]) for i in range(lane_label.max())])
        seg_pos = np.array([np.mean(np.argwhere(lane_label==i+1)) for i in range(lane_label.max())])
        seg_minpos = np.array([np.argwhere(lane_label==i+1)[0][0] for i in range(lane_label.max())])
        seg_maxpos = np.array([np.argwhere(lane_label==i+1)[-1][0] for i in range(lane_label.max())])
        
        '''#alternative using complete segmentation information
        seg_minpos = np.array([kymo_min[lane_label==i+1,r][0] for i in range(lane_label.max())])
        seg_maxpos = np.array([kymo_max[lane_label==i+1,r][0] for i in range(lane_label.max())])
        seg_pos = 0.5*(seg_minpos+seg_maxpos)
        seg_sum = seg_maxpos-seg_minpos
        seg_mean = np.array([np.median(segm_part[:,r][lane_label==i+1]) for i in range(lane_label.max())])'''
        
        seg_pos = np.c_[seg_pos,seg_sum,seg_mean,seg_minpos,seg_maxpos]
        #suppress minuscule cells
        seg_pos = seg_pos[(seg_pos[:,1]>lim),:]
        #seg_pos = seg_pos[(seg_pos[:,1]>lim),:]

        all_pos.append(np.flipud(seg_pos))
    
    #suppress isolated small cells
    small_cell = [np.any(x[0:-1,1]<10) for x in all_pos]
    for i in range(1,len(small_cell)-1):
        if small_cell[i]:
            if not (small_cell[i+1]) or not (small_cell[i-1]):
                all_pos[i] = all_pos[i][all_pos[i][:,1]>=10,:]
                
            
    return all_pos


def deep_linking(pos_time_list):
    all_pos = pos_time_list.copy()
    #initialize the dictionary that will contain the cell tracks
    t_init = 0
    t_max = len(all_pos)-1#segm_part.shape[1]-1
    current = []
    for i in range(all_pos[t_init].shape[0]):
        temp = {}
        #addNameToDictionary(assembly, i,{})
        addNameToDictionary(temp, 'pos',[all_pos[t_init][i,0]])
        addNameToDictionary(temp, 'length',[all_pos[t_init][i,1]])
        addNameToDictionary(temp, 'finish',False)
        addNameToDictionary(temp, 'full_cellcycle',False)
        addNameToDictionary(temp, 'born',t_init)
        addNameToDictionary(temp, 'pix_min',[all_pos[t_init][i,3]])
        addNameToDictionary(temp, 'pix_max',[all_pos[t_init][i,4]])
        addNameToDictionary(temp, 'genealogy',str(i))

        current.append(temp)
    current = filter(lambda cell: cell['finish']==False, current)
    current = sorted(current, key=lambda k: k['pos'][-1],reverse=True)


    #track cells over time
    finished = []
    for t in range(t_init,t_max):
        p=0#all_pos[t+1].shape[0]-1
        current = list(filter(lambda cell: cell['finish']==False, current))
        current = sorted(current, key=lambda k: k['pos'][-1],reverse=True) 
        for i in range(len(current)):

            #no further cell
            if p==all_pos[t+1].shape[0]:
                current[i]['finish'] = True                
                continue

            #measure distance between cells and length ratio. Ratio ~2 means cell divided
            ratio_len = current[i]['length'][-1]/all_pos[t+1][p,1]

            #measure futher length ratios between neighboring cells and candidates
            ratio_len2=None
            sum_ratio = None
            if (p<all_pos[t+1].shape[0]-1):
                ratio_len2 = current[i]['length'][-1]/all_pos[t+1][p+1,1]
                sum_ratio = current[i]['length'][-1]/(all_pos[t+1][p,1]+all_pos[t+1][p+1,1])
            ratio_len3=None
            if (i<len(current)-1):
                ratio_len3 = current[i+1]['length'][-1]/all_pos[t+1][p,1]

            
            #if current candidate is not last one, check whether next one is consistent with dividing cell
            if (ratio_len>1)&(ratio_len2!=None):  
                
                #a cell has divided
                if ((ratio_len>1.5)&(ratio_len2>1.5))or((ratio_len>1)&(np.abs(1-sum_ratio)<0.1)):
                    
                    current[i]['finish'] = True
                    current[i]['full_cellcycle'] = True

                    temp = {}
                    addNameToDictionary(temp, 'pos',[all_pos[t+1][p,0]])
                    addNameToDictionary(temp, 'pix_min',[all_pos[t+1][p,3]])
                    addNameToDictionary(temp, 'pix_max',[all_pos[t+1][p,4]])
                    addNameToDictionary(temp, 'length',[all_pos[t+1][p,1]])
                    addNameToDictionary(temp, 'finish',False)
                    addNameToDictionary(temp, 'full_cellcycle',False)
                    addNameToDictionary(temp, 'born',t+1)
                    addNameToDictionary(temp, 'genealogy',current[i]['genealogy']+'B')
                    current.append(temp)
                    temp = {}
                    p+=1
                    if p<all_pos[t+1].shape[0]:
                        temp = {}
                        addNameToDictionary(temp, 'pos',[all_pos[t+1][p,0]])
                        addNameToDictionary(temp, 'pix_min',[all_pos[t+1][p,3]])
                        addNameToDictionary(temp, 'pix_max',[all_pos[t+1][p,4]])
                        addNameToDictionary(temp, 'length',[all_pos[t+1][p,1]])
                        addNameToDictionary(temp, 'finish',False)
                        addNameToDictionary(temp, 'full_cellcycle',False)
                        addNameToDictionary(temp, 'born',t+1)
                        addNameToDictionary(temp, 'genealogy',current[i]['genealogy']+'T')
                        current.append(temp)
                        p+=1
                
                elif (ratio_len>1.5)&(ratio_len2<1.5):
                    p+=1
                    current[i]['pos'].append(all_pos[t+1][p,0])
                    current[i]['length'].append(all_pos[t+1][p,1])
                    current[i]['pix_min'].append(all_pos[t+1][p,3])
                    current[i]['pix_max'].append(all_pos[t+1][p,4])
                    p+=1
                
                else:
                    current[i]['pos'].append(all_pos[t+1][p,0])
                    current[i]['length'].append(all_pos[t+1][p,1])
                    current[i]['pix_min'].append(all_pos[t+1][p,3])
                    current[i]['pix_max'].append(all_pos[t+1][p,4])
                    p+=1
            
            #if the current cell is not the last one
            elif (ratio_len<0.75)&(ratio_len3!=None):
                #a cell has merged
                if (ratio_len<0.75)&(ratio_len3<0.75):
                    
                    '''print('time: '+str(t))
                    print('ratio_len: '+str(ratio_len))
                    print('ratio_len2: '+str(ratio_len2))
                    print('ratio_len3: '+str(ratio_len3))'''
                    #print('here time: '+str(t))
                    all_pos[t+1] = np.insert(all_pos[t+1],p,all_pos[t+1][p,:],axis = 0)
                    #recalculated division point between fused cells
                    temp_pos = all_pos[t+1][p,0]+all_pos[t+1][p,1]/2-current[i]['length'][-1]

                    all_pos[t+1][p,1] = current[i]['length'][-1]
                    all_pos[t+1][p+1,1] = current[i+1]['length'][-1]
                    #set new mid-positions
                    all_pos[t+1][p+1,0] = temp_pos-0.5*all_pos[t+1][p+1,1]
                    all_pos[t+1][p,0] = temp_pos+0.5*all_pos[t+1][p,1]

                    all_pos[t+1][p+1,3] = all_pos[t+1][p+1,0]-0.5*all_pos[t+1][p+1,1]
                    all_pos[t+1][p+1,4] = all_pos[t+1][p+1,0]+0.5*all_pos[t+1][p+1,1]
                    all_pos[t+1][p,3] = all_pos[t+1][p,0]-0.5*all_pos[t+1][p,1]
                    all_pos[t+1][p,4] = all_pos[t+1][p,0]+0.5*all_pos[t+1][p,1]

                    current[i]['pos'].append(all_pos[t+1][p,0])
                    current[i]['length'].append(all_pos[t+1][p,1])
                    current[i]['pix_min'].append(all_pos[t+1][p,3])
                    current[i]['pix_max'].append(all_pos[t+1][p,4])
                    p+=1

                
                else:
                    
                    current[i]['pos'].append(all_pos[t+1][p,0])
                    current[i]['length'].append(all_pos[t+1][p,1])
                    current[i]['pix_min'].append(all_pos[t+1][p,3])
                    current[i]['pix_max'].append(all_pos[t+1][p,4])
                    p+=1

            else:
                
                current[i]['pos'].append(all_pos[t+1][p,0])
                current[i]['length'].append(all_pos[t+1][p,1])
                current[i]['pix_min'].append(all_pos[t+1][p,3])
                current[i]['pix_max'].append(all_pos[t+1][p,4])
                p+=1

        finished = finished+list(filter(lambda cell: cell['finish']==True, current))
    for x in range(len(finished)):
        finished[x]['length'] = np.array(finished[x]['length'])
        finished[x]['pix_min'] = np.array(finished[x]['pix_min'])
        finished[x]['pix_max'] = np.array(finished[x]['pix_max'])
        finished[x]['pos'] = np.array(finished[x]['pos'])
        
    for x in range(len(current)):
        current[x]['length'] = np.array(current[x]['length'])
        current[x]['pix_min'] = np.array(current[x]['pix_min'])
        current[x]['pix_max'] = np.array(current[x]['pix_max'])
        current[x]['pos'] = np.array(current[x]['pos'])

    time_mat = finished+current
    time_mat_pd = pd.DataFrame(time_mat)
    
    time_mat_pd['pix_max']=time_mat_pd.pix_max.apply(lambda x: np.array(x))
    time_mat_pd['pix_min']=time_mat_pd.pix_min.apply(lambda x: np.array(x))
    time_mat_pd['pos']=time_mat_pd.pos.apply(lambda x: np.array(x))
    time_mat_pd['length']=time_mat_pd.length.apply(lambda x: np.array(x))
    
    time_mat_pd = tmo.essential_props(time_mat_pd)
    
    return time_mat_pd


def cleanup_kymo(kymo):
    footp = np.ones((5,2)).astype(bool)
    kymolab = label(1-kymo)
    reginfo = regionprops(kymolab)
    for c in reginfo:
        if c.coords[:,1].max()-c.coords[:,1].min()<6:
            kymocopy = 1-kymo.copy()
            kymo_single = np.zeros(kymo.shape)
            kymo_single[c.coords[:,0],c.coords[:,1]]=1
            kymo_single = scipy.ndimage.filters.maximum_filter(kymo_single, footprint=footp)
            size_single = np.sum(kymo_single)
            lab_to_supp = np.unique(kymolab[:,c.coords[:,1].min():c.coords[:,1].max()+1])
            if lab_to_supp.shape[0]>0:
                for x in lab_to_supp:
                    if (x>0)&(x!=c.label):
                        kymocopy[kymolab==x]=0

            #kymocopy[:,c.coords[:,1].max()+2::]=0
            #kymocopy[:,0:c.coords[:,1].min()-1]=0
            
            maxfilt = scipy.ndimage.filters.maximum_filter(kymocopy, footprint=footp)
            maxlab = label(maxfilt)
            reginfo = regionprops(maxlab)
            to_consider = maxlab[c.coords[0,0],c.coords[0,1]]
            if to_consider==0:
                kymo[c.coords[:,0],c.coords[:,1]]=1
            for x in reginfo:
                if x.label == to_consider:
                    if (x.area<=size_single)or(x.coords[:,1].max()-x.coords[:,1].min()<=6):
                        kymo[c.coords[:,0],c.coords[:,1]]=1
                        
            '''fig, ax = plt.subplots(figsize = (30,30))
            plt.subplot(1,4,1)
            plt.imshow(kymocopy,interpolation=None,cmap = 'gray',vmin=0.0,alpha =0.8)
            plt.subplot(1,4,2)
            plt.imshow(maxfilt,interpolation=None,cmap = 'gray',vmin=0.0,alpha =0.8)
            #plt.subplot(1,4,3)
            #plt.imshow(kymo_test,interpolation=None,cmap = 'gray',vmin=0.0,alpha =0.8)
            #plt.subplot(1,4,4)
            #plt.imshow(supp_mask,interpolation=None,cmap = 'gray',vmin=0.0,alpha =0.8)
            plt.show()'''
    return kymo


def deep_spot_detect(momobj, weights_file, min_time=0, max_time=None,limits = None, threshold = 0.5):
    
    if max_time is None:
        max_time = momobj.get_max_time()
        
    im_height = 512
    im_width = 32
    #create model and load weights
    model = get_unet(1,im_height,im_width)
    model.load_weights(weights_file+'/weights.h5')
    
    topad = 512-momobj.height
    
    tot_time = max_time-min_time+1
    
    spot_time = np.zeros((im_height,tot_time))
    
    '''if limits is None:
        spot_time = np.empty((im_height,tot_time))
    else:
        spot_time = np.empty((limits[1]-limits[0],tot_time))'''
    spot_time_raw = spot_time.copy()
    deep_raw = spot_time.copy()
    spot_coord = []
    spot_amp = []
    
    for time_count, t in enumerate(range(min_time,max_time+1)):
        
        if output:
            print(t)
            
        momobj.time = t
        im_fluo = momobj.load_moma_im()

        im_fluo = im_fluo.astype('float32')
        
        center = np.round(im_fluo.shape[1]/2).astype(int)
        im_fluo = np.pad(im_fluo,((im_height-im_fluo.shape[0],0),(0,0)),mode='constant')[:,center-int(im_width/2):center+int(im_width/2)]

        imgs_test = im_fluo.copy()

        imgs_test = imgs_test[np.newaxis,...,np.newaxis]
        if output:
            imgs_mask_test = model.predict(imgs_test, verbose=1)
        else:
            imgs_mask_test = model.predict(imgs_test, verbose=0)

        imgs_mask_test = np.reshape(imgs_mask_test,im_fluo.shape)
        
        if limits is not None:
            imgs_mask_test = imgs_mask_test[limits[0]:limits[1],:]
            im_fluo = im_fluo[limits[0]:limits[1],:]
        
        image_mask_thresshold = np.zeros(imgs_mask_test.shape)
        image_mask_clean = np.zeros(imgs_mask_test.shape)
        image_mask_thresshold[imgs_mask_test>threshold]=1
        image_mask_label = label(image_mask_thresshold)
        
        mask_info = regionprops(image_mask_label,im_fluo)
        temp_spots = []
        temp_amp = []
        for reg in mask_info:
            if reg.area>5:
                image_mask_clean[image_mask_label==reg.label]=1
                temp_spots.append(reg.centroid)
                temp_amp.append(reg.mean_intensity)
        spot_coord.append(temp_spots)
        spot_amp.append(temp_amp)

        center = np.round(imgs_mask_test.shape[1]/2).astype(int)

        deep_raw[:,time_count] = np.mean(imgs_mask_test[:,center-16:center+16],axis =1)
        spot_time[:,time_count] = np.max(image_mask_clean[:,center-16:center+16],axis =1)
        spot_time_raw[:,time_count] = np.sum(im_fluo[:,center-16:center+16],axis =1)
        
    deep_raw = deep_raw[topad::,:]
    spot_time = spot_time[topad::,:]
    spot_time_raw = spot_time_raw[topad::,:]

    spot_coord = [np.array(x)-np.repeat([[topad,0]],len(x),axis = 0) if len(x)>0 else [] for x in spot_coord]


    return spot_time, spot_time_raw, spot_coord, spot_amp, deep_raw


def deep_assign_spots_to_cells(time_mat_pd,spot_coord,spot_amp):
    cond_thresh = 0
    time_mat_pd['spots'] = np.nan
    all_spots = []
    for index in time_mat_pd.index:
        spot_in_cell = []
        for t in range(time_mat_pd.loc[index].pix_max.shape[0]):#range(len(time_mat_pd.loc[index].cell_len)):

            born = time_mat_pd.loc[index].born

            if t+born>=len(spot_amp):
                spot_in_cell.append(np.empty((0,4)))
                continue
            temp_coord = np.array(spot_coord[t+born])
            if temp_coord.shape[0]==0:
                #spot_in_cell.append(np.array([]))
                spot_in_cell.append(np.empty((0,4)))
                continue

            temp_amp = np.array(spot_amp[t+born])

            temp_tot = np.c_[temp_coord,temp_amp]

            index1 = int(time_mat_pd.loc[index].pix_min[t])
            index2 = int(time_mat_pd.loc[index].pix_max[t])

            condition = (temp_tot[:,2]>cond_thresh)&(temp_coord[:,1]>6)&(temp_coord[:,1]<26)&(temp_coord[:,0]>index1)&(temp_coord[:,0]<index2)
            temp_coord_sel = temp_tot[condition,:]

            temp_coord_sel = temp_coord_sel-np.repeat([[(index1+index2)/2,0,0]],[temp_coord_sel.shape[0]],axis=0)
            temp_coord_sel = np.c_[temp_coord_sel,t*np.ones(temp_coord_sel.shape[0])]
            spot_in_cell.append(temp_coord_sel)
        all_spots.append(spot_in_cell)
    return all_spots


def deep_make_spot_tracks(time_mat_pd):
    time_mat_pd['spots_num'] = np.nan
    time_mat_pd['spots_num'] = time_mat_pd['spots_num'].astype(object)
    time_mat_pd['spots_tracks'] = np.nan
    time_mat_pd['spots_tracks'] = time_mat_pd['spots_tracks'].astype(object)

    for index in time_mat_pd.index:#range(31,80):
        if time_mat_pd.iloc[index].full_cellcycle:
            mother_id = time_mat_pd.iloc[index].mother_id
            if (not np.isnan(mother_id)):

                #recover mother info
                mother_id = int(mother_id)
                mother_time = len(time_mat_pd.iloc[mother_id]['spots'])
                mother_Ld = time_mat_pd.iloc[mother_id].length[-1]

                daughter_Ld = time_mat_pd.iloc[index].length[-1]
                daughter_time = len(time_mat_pd.iloc[index]['spots'])

                gdaughter1_id = time_mat_pd.index[time_mat_pd.genealogy == time_mat_pd.iloc[index].genealogy+'B']
                gdaughter1_id = gdaughter1_id[0] if len(gdaughter1_id)>0 else None
                gdaughter2_id = time_mat_pd.index[time_mat_pd.genealogy == time_mat_pd.iloc[index].genealogy+'T']
                gdaughter2_id = gdaughter2_id[0] if len(gdaughter2_id)>0 else None

                if (gdaughter1_id is None) or (gdaughter2_id is None):
                    continue
                gdaughter1_time = len(time_mat_pd.iloc[gdaughter1_id]['spots'])
                gdaughter2_time = len(time_mat_pd.iloc[gdaughter2_id]['spots'])

                #proceed if mother cell is ok
                if (not np.isnan(mother_Ld))&(mother_time>5)&(gdaughter1_time>5)&(gdaughter2_time>5):

                    #find out if daughter is top/bottom
                    if time_mat_pd.iloc[index].genealogy[-1]=='B':
                        sign = -1
                        displace = -1/4
                    else: 
                        sign = 1
                        displace = 1/4


                    combined = np.empty((0,4))

                    for x in range(mother_time-5,mother_time):
                        mother_spots = time_mat_pd.iloc[mother_id]['spots'][x]
                        mother_spots = mother_spots[np.sign(mother_spots[:,0])==sign]#mother_Ld/2]
                        mother_spots = np.c_[(x-mother_time)*np.ones((mother_spots[:,0].shape[0],1)),mother_spots[:,0:3]]
                        combined = np.concatenate((combined,mother_spots))

                    daughter_time = len(time_mat_pd.iloc[index]['spots'])
                    for x in range(daughter_time):
                        daughter_spots = np.c_[(x)*np.ones((time_mat_pd.iloc[index]['spots'][x][:,0].shape[0],1)),
                                               time_mat_pd.iloc[index]['spots'][x][:,0]+displace*mother_Ld,time_mat_pd.iloc[index]['spots'][x][:,1:3]]
                        combined = np.concatenate((combined,daughter_spots))

                    for x in range(5):
                        gdaughter_spots = np.c_[(x+daughter_time)*np.ones((time_mat_pd.iloc[gdaughter1_id]['spots'][x][:,0].shape[0],1)),
                                               time_mat_pd.iloc[gdaughter1_id]['spots'][x][:,0]+(displace*mother_Ld)-1/4*daughter_Ld,time_mat_pd.iloc[gdaughter1_id]['spots'][x][:,1:3]]
                        combined = np.concatenate((combined,gdaughter_spots))

                    for x in range(5):
                        gdaughter_spots = np.c_[(x+daughter_time)*np.ones((time_mat_pd.iloc[gdaughter2_id]['spots'][x][:,0].shape[0],1)),
                                               time_mat_pd.iloc[gdaughter2_id]['spots'][x][:,0]+(displace*mother_Ld)+1/4*daughter_Ld,time_mat_pd.iloc[gdaughter2_id]['spots'][x][:,1:3]]
                        combined = np.concatenate((combined,gdaughter_spots))

                    
                    combined = combined[:,0:3]
                    spot_frame = pd.DataFrame(combined,columns=('frame','x','y'))
                    #spot_frame = pd.DataFrame(combined,columns=('frame','x','z','y'))
                    spot_tracked = trackpy.link_df(spot_frame,20,memory=5,link_strategy='nonrecursive',pos_columns=['x', 'y'])
                    spot_tracked = trackpy.filter_stubs(spot_tracked,5)


                    #fig, ax = plt.subplots()
                    
                    tracks = []
                    for tr in spot_tracked.particle.unique():
                        curr_track = spot_tracked[spot_tracked.particle==tr]
                        #interpolate missing values
                        tracks.append(np.c_[[np.interp(np.arange(curr_track.frame.min(),curr_track.frame.max()+1),curr_track.frame,curr_track[key]) for key in ['frame','x','y']]].transpose())
                        #plt.plot(spot_tracked[spot_tracked.particle==tr].frame,spot_tracked[spot_tracked.particle==tr].x,'-')

                    #limit tracks to daughter segment
                    cut_tracks = [x[(x[:,0]>=0)&(x[:,0]<daughter_time),:] for x in tracks]
                    cut_tracks = [x for x in cut_tracks if len(x)>0]
                    #calculate number of spots as a f. of time
                    num_spots = np.zeros(daughter_time)
                    for x in range(len(cut_tracks)):
                        num_spots[int(cut_tracks[x][0,0]):int(cut_tracks[x][-1,0])+1]+=1
                    time_mat_pd.at[index,'spots_num'] = num_spots
                    time_mat_pd.at[index,'spots_tracks'] = cut_tracks
                    
                    
                    #plt.plot(combined[:,0],combined[:,1],'o')
                    #plt.plot(combined[(combined[:,0]>0)&(combined[:,0]<daughter_time)][:,0],
                    #         combined[(combined[:,0]>0)&(combined[:,0]<daughter_time)][:,1],'ro')  
                    #ax.set_title('index: '+str(index))
                    #plt.show()
                '''else:
                    print('index: '+str(index)+' mother_exit: ')#+time_mat_pd.iloc[mother_id].finish +' mother_time: '+ str(mother_time)+' d2exit: '+ time_mat_pd.iloc[gdaughter2_id].finish + ' d1exit: ' +time_mat_pd.iloc[gdaughter1_id].finish)
            else:
                print('index: '+str(index)+'mothernan')
        else:
            print('index: '+str(index)+' daugher_exit: ')#+time_mat_pd.iloc[index].finish)'''
    return time_mat_pd



def deep_alternate_spot_tracks(time_mat_pd):
    time_mat_pd['spots_num'] = np.nan
    time_mat_pd['spots_num'] = time_mat_pd['spots_num'].astype(object)
    time_mat_pd['spots_tracks'] = np.nan
    time_mat_pd['spots_tracks'] = time_mat_pd['spots_tracks'].astype(object)
    time_mat_pd['full_tracks'] = np.nan
    time_mat_pd['full_tracks'] = time_mat_pd['full_tracks'].astype(object)
    
    for index in time_mat_pd.index:#range(31,80):
        maxnumspots = np.max([len(x) for x in time_mat_pd.loc[index].spots])
        if (time_mat_pd.iloc[index].full_cellcycle)&(maxnumspots<10):
            mother_id = time_mat_pd.iloc[index].mother_id
            if mother_id>0:

                #recover mother info
                mother_id = int(mother_id)
                mother_time = len(time_mat_pd.iloc[mother_id]['spots'])
                mother_Ld = time_mat_pd.iloc[mother_id].length[-1]

                daughter_Ld = time_mat_pd.iloc[index].length[-1]
                daughter_time = len(time_mat_pd.iloc[index]['spots'])

                gdaughter1_id = time_mat_pd.index[time_mat_pd.genealogy == time_mat_pd.iloc[index].genealogy+'B']
                gdaughter1_id = gdaughter1_id[0] if len(gdaughter1_id)>0 else None
                gdaughter2_id = time_mat_pd.index[time_mat_pd.genealogy == time_mat_pd.iloc[index].genealogy+'T']
                gdaughter2_id = gdaughter2_id[0] if len(gdaughter2_id)>0 else None

                if (gdaughter1_id is None) or (gdaughter2_id is None):
                    continue
                gdaughter1_time = len(time_mat_pd.iloc[gdaughter1_id]['spots'])
                gdaughter2_time = len(time_mat_pd.iloc[gdaughter2_id]['spots'])

                #proceed if mother cell is ok
                if (not np.isnan(mother_Ld))&(mother_time>10)&(gdaughter1_time>5)&(gdaughter2_time>5):

                    #find out if daughter is top/bottom
                    if time_mat_pd.iloc[index].genealogy[-1]=='B':
                        sign = -1
                        displace = -1/4
                    else: 
                        sign = 1
                        displace = 1/4


                    combined = np.empty((0,4))

                    for x in range(mother_time-10,mother_time):
                        mother_spots = time_mat_pd.iloc[mother_id]['spots'][x]
                        mother_spots = mother_spots[np.sign(mother_spots[:,0])==sign]#mother_Ld/2]
                        mother_spots = np.c_[(x-mother_time)*np.ones((mother_spots[:,0].shape[0],1)),mother_spots[:,0:3]]
                        combined = np.concatenate((combined,mother_spots))

                    daughter_time = len(time_mat_pd.iloc[index]['spots'])
                    for x in range(daughter_time):
                        daughter_spots = np.c_[(x)*np.ones((time_mat_pd.iloc[index]['spots'][x][:,0].shape[0],1)),
                                               time_mat_pd.iloc[index]['spots'][x][:,0]+displace*mother_Ld,time_mat_pd.iloc[index]['spots'][x][:,1:3]]
                        combined = np.concatenate((combined,daughter_spots))

                    for x in range(5):
                        gdaughter_spots = np.c_[(x+daughter_time)*np.ones((time_mat_pd.iloc[gdaughter1_id]['spots'][x][:,0].shape[0],1)),
                                               time_mat_pd.iloc[gdaughter1_id]['spots'][x][:,0]+(displace*mother_Ld)-1/4*daughter_Ld,time_mat_pd.iloc[gdaughter1_id]['spots'][x][:,1:3]]
                        combined = np.concatenate((combined,gdaughter_spots))

                    for x in range(5):
                        gdaughter_spots = np.c_[(x+daughter_time)*np.ones((time_mat_pd.iloc[gdaughter2_id]['spots'][x][:,0].shape[0],1)),
                                               time_mat_pd.iloc[gdaughter2_id]['spots'][x][:,0]+(displace*mother_Ld)+1/4*daughter_Ld,time_mat_pd.iloc[gdaughter2_id]['spots'][x][:,1:3]]
                        combined = np.concatenate((combined,gdaughter_spots))


                    combined = combined[:,0:3]
                    spot_frame = pd.DataFrame(combined,columns=('frame','x','y'))
                    #spot_frame = pd.DataFrame(combined,columns=('frame','x','z','y'))
                    spot_tracked = trackpy.link_df(spot_frame,20,memory=5,link_strategy='nonrecursive',pos_columns=['x', 'y'])
                    spot_tracked = trackpy.filter_stubs(spot_tracked,5)


                    #fig, ax = plt.subplots()

                    tracks = []
                    for tr in spot_tracked.particle.unique():
                        curr_track = spot_tracked[spot_tracked.particle==tr]
                        #interpolate missing values
                        tracks.append(np.c_[[np.interp(np.arange(curr_track.frame.min(),curr_track.frame.max()+1),curr_track.frame,curr_track[key]) for key in ['frame','x','y']]].transpose())
                        #plt.plot(spot_tracked[spot_tracked.particle==tr].frame,spot_tracked[spot_tracked.particle==tr].x,'-')

                    time_mat_pd.at[index,'full_tracks'] = tracks
                    
                    #limit tracks to daughter segment
                    cut_tracks = [x[(x[:,0]<daughter_time),:] for x in tracks]
                    cut_tracks = [x for x in cut_tracks if len(x)>0]
                    #calculate number of spots as a f. of time
                    num_spots = np.zeros(daughter_time+10)
                    for x in range(len(cut_tracks)):
                        num_spots[int(cut_tracks[x][0,0])+10:int(cut_tracks[x][-1,0])+11]+=1
                    time_mat_pd.at[index,'spots_num'] = num_spots
                    time_mat_pd.at[index,'spots_tracks'] = cut_tracks

                    if (num_spots[0]==1)&(len(np.argwhere(num_spots==2))>0):
                        time_mat_pd.at[index,'Ti'] = np.argwhere(num_spots==2)[0][0]-10

                    '''fig, ax = plt.subplots()
                    plt.plot(combined[:,0],combined[:,1],'o')
                    plt.plot(combined[(combined[:,0]>0)&(combined[:,0]<daughter_time)][:,0],
                             combined[(combined[:,0]>0)&(combined[:,0]<daughter_time)][:,1],'ro')  
                    ax.set_title('index: '+str(index))
                    plt.show()'''

                    #break

    return time_mat_pd


def addNameToDictionary(d, name, emptystruct):
    if name not in d:
        d[name] = emptystruct