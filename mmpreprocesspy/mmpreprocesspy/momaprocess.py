import os
import re
import glob
import numpy as np
import pandas as pd

#from skimage import feature
from skimage.feature import hessian_matrix, hessian_matrix_eigvals, match_template
import scipy

from . import tools_GW as tgw


def addNameToDictionary(d, name, emptystruct):
    if name not in d:
        d[name] = emptystruct

#time of frames is 0 based. Cells on the first frame are born at t = -1
def parse_exported(moma_path): 
    
    if not os.path.exists(moma_path):
        print('no such file')
        return None
        
    file = open(moma_path, 'r')
    tline = file.readline()
    while not re.search('trackRegionInterval',tline):
        tline = file.readline()
    pixlim = re.findall('(\d+)',tline)
    tracklim = int(pixlim[0])
    
    time_mat={}
    while tline:
        if re.search('id=',tline):
            index = int(re.search('(\d+)',tline).group(0))
            addNameToDictionary(time_mat, index,{})
            addNameToDictionary(time_mat[index], 'tracklim',[])
            addNameToDictionary(time_mat[index], 'born',[])
            addNameToDictionary(time_mat[index], 'genealogy',[])
            addNameToDictionary(time_mat[index], 'pixlim',[])
            addNameToDictionary(time_mat[index], 'pos_GL',[])
            addNameToDictionary(time_mat[index], 'exit_type',[])
            
            time_mat[index]['tracklim'] = tracklim
            time_mat[index]['born'] = int(re.search('(?<=birth_frame=)-*(\d+)',tline).group(0))
            tline = file.readline()
            while re.search('(frame=)|(output=)',tline):
                if re.search('frame=',tline):
                    frame = int(re.search('(?<=frame=)(\d+)',tline).group(0))+1
                    time_mat[index]['genealogy'] =re.search('(?<=genealogy=)([0-9TB]*)',tline).group(0)
                    pix_low = int(re.findall('pixel_limits=\[(\d*),',tline)[0])
                    pix_high = int(re.findall('pixel_limits=\[\d*,(\d*)\]',tline)[0])
                    time_mat[index]['pixlim'].append([pix_low,pix_high])

                    pos_GL = int(re.findall('pos_in_GL=\[(\d*),',tline)[0]);#position from top in GL
                    num_GL = int(re.findall('pos_in_GL=\[\d*,(\d*)\]',tline)[0]);#total cells in GL
                    time_mat[index]['pos_GL'].append([pos_GL,num_GL])
                tline = file.readline()
            if re.search('DIVISION',tline):
                time_mat[index]['exit_type'] = 'DIVISION'
            elif re.search('EXIT',tline):
                time_mat[index]['exit_type'] = 'EXIT'
            elif re.search('USER_PRUNING',tline):
                time_mat[index]['exit_type'] = 'USER_PRUNING'
            elif re.search('ENDOFDATA',tline):
                time_mat[index]['exit_type'] = 'ENDOFDATA'
            time_mat[index]['pixlim'] = np.array(time_mat[index]['pixlim'])
            time_mat[index]['pos_GL'] = np.array(time_mat[index]['pos_GL'])
        else:
            tline = file.readline()
    time_mat = pd.DataFrame(time_mat).T
    return time_mat

def moma_cleanup(time_mat):
    
    time_mat = time_mat[time_mat.pos_GL.apply(lambda x: 1 not in x[:,0])]
    time_mat = time_mat[time_mat.born>0]
    
    return time_mat


def length_hessian(mom):

    time_mat = mom.tracks
    model = np.ones([9,9])
    modmidle = int((9-1)/2)

    for i in range(model.shape[0]):
        for j in range(model.shape[1]):
            if ((2-i)**2+(modmidle-j)**2)**0.5<modmidle+1:
                model[i,j]=0.0
    model = 1-model
    
    time_mat['cell_len'] = np.nan
    time_mat['mid_val'] = np.nan
    time_mat['mid_pos'] = np.nan
    
    time_mat['cell_len'] = time_mat.cell_len.astype('object')
    time_mat['mid_val'] = time_mat.mid_val.astype('object')
    time_mat['mid_pos'] = time_mat.mid_pos.astype('object')

    for c in time_mat.index:

        #get real frame times
        times = time_mat.at[c,'born']+np.arange(time_mat.at[c,'Td'])

        cell_len = []
        mid_val = []
        mid_pos = []
        for t in range(len(times)):
            if time_mat.at[c,'pos_GL'][t,0]==time_mat.at[c,'pos_GL'][t,1]:
                bottom_basic =1
            else:
                bottom_basic =0

                mom.time=times[t]
                image = mom.load_moma_im()
                im_size = image.shape
                im_middle = int((im_size[1]-1)/2)
                #reduce to image center
                image = image[::,im_middle-10:im_middle+11]
                im_size = image.shape
                im_middle = int((im_size[1]-1)/2)

                #recover MoMA box and extend it by 20px
                index1 = time_mat.at[c,'pixlim'][t,0]+time_mat.at[c,'tracklim']-20
                index2 = time_mat.at[c,'pixlim'][t,1]+time_mat.at[c,'tracklim']+21

                mean_middle = np.mean(np.array(image[index1:index2,im_middle-10:im_middle+11]),1)

                Hxx, Hxy, Hyy = hessian_matrix(image[index1:index2,::], sigma=0.1, order='rc')
                hessian = hessian_matrix_eigvals(Hxx, Hxy, Hyy)

                im_hess = hessian[1][::,im_middle-10:im_middle+11]
                im_hess2 = hessian[0][::,im_middle-10:im_middle+11]
                im_size = im_hess.shape
                im_middle = int((im_size[1]-1)/2)


                match_bottom = match_template(im_hess, model,pad_input=True,mode='constant')
                match_top = match_template(im_hess, np.flipud(model),pad_input=True,mode='constant')

                if bottom_basic == 1:
                    proj_bottom = np.max(match_bottom[15:-20,im_middle-5:im_middle+6],axis=1)
                    lim_proj = 0.2
                else:
                    proj_bottom = np.max(match_bottom[15:-15,im_middle-5:im_middle+6],axis=1)
                    lim_proj = 0.2
                proj_top = np.max(match_top[15:-15,im_middle-5:im_middle+6],axis=1)

                f = scipy.interpolate.interp1d(np.arange(0,proj_bottom.shape[0]), proj_bottom,kind='quadratic')
                x_bottom = np.arange(0,proj_bottom.shape[0]-1,0.1)
                proj_bottom = f(x_bottom)

                f = scipy.interpolate.interp1d(np.arange(0,proj_top.shape[0]), proj_top,kind='quadratic')
                x_top = np.arange(0,proj_top.shape[0]-1,0.1)
                proj_top = f(x_top)

                pos_max_proj_top = np.nonzero(tgw.locmax(proj_top,5))[0]
                pos_max_proj_top = x_top[pos_max_proj_top[proj_top[pos_max_proj_top]>0.2]]

                pos_max_proj_bottom = np.nonzero(tgw.locmax(proj_bottom,5))[0]
                pos_max_proj_bottom = x_bottom[pos_max_proj_bottom[proj_bottom[pos_max_proj_bottom]>lim_proj]]

                
                if (pos_max_proj_top.shape[0]>0) & (pos_max_proj_bottom.shape[0]>0):
                    lim_top = pos_max_proj_top[0]
                    lim_bottom = pos_max_proj_bottom[-1]

                    mid = int(np.round(15+0.5*(lim_bottom+lim_top)))+index1

                    cell_len.append(lim_bottom-lim_top+6)
                    mid_val.append(np.mean(mean_middle[mid-index1-1:mid-index1+2]))
                    mid_pos.append(mid)
                else:
                    cell_len.append(np.nan)
                    mid_val.append(np.nan)
                    mid_pos.append(np.nan)


        time_mat.at[c,'cell_len'] = cell_len
        time_mat.at[c,'mid_val'] = mid_val
        time_mat.at[c,'mid_pos'] = mid_pos
    return time_mat

                        