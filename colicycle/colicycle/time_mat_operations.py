import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats
import trackpy

from . import tools_GW as tgw
from . import spot_detection as sp

def essential_props(time_mat):
    #time_mat['length'] = time_mat.pixlim.apply(lambda x: x[:,1]-x[:,0])
    time_mat['Td'] = time_mat.length.apply(lambda x: x.shape[0])
    time_mat['Ld'] = time_mat.length.apply(lambda x: x[-1])
    time_mat['Lb'] = time_mat.length.apply(lambda x: x[0])
    time_mat['Tb'] = time_mat.born.apply(lambda x: x if x>=0 else 0)
    
    time_mat['mother_id'] = time_mat.apply(lambda row: 
                                                 int(time_mat.index[time_mat.genealogy == row.genealogy[0:-1]][0]) 
                                                 if len(row.genealogy[0:-1])>0 else np.nan,axis = 1)
    
    return time_mat

def exponential_fit(row):
    tau_fit= np.nan
    Lb_fit = np.nan
    Ld_fit = np.nan
    pearson_lin = np.nan
    pearson_log = np.nan
    if len(row['length'])>5:
        xdata = range(len(row['length']))
        tau0 = row['Td']
        ydata = row['length']
        popt, pcov = scipy.optimize.curve_fit(tgw.fun_expgrowht2, xdata, ydata, p0=[15,tau0])
        tau_fit = popt[1]
        Lb_fit = popt[0]
        Ld_fit = tgw.fun_expgrowht2(row['Td'],popt[0],popt[1])
        pearson_lin = scipy.stats.pearsonr(xdata, ydata)[0]
        pearson_log = scipy.stats.pearsonr(xdata, np.log(ydata))[0]

        
    return pd.Series({'tau_fit': tau_fit, 'Lb_fit': Lb_fit,'Ld_fit': Ld_fit,'pearson_lin': pearson_lin,'pearson_log': pearson_log})


#return a matrix whose first column contains all time points at which cell indices of the second column exist
def get_cell_id_time_matrix(time_mat_pd):
    series_temp = time_mat_pd.pixlim.apply(lambda x: np.arange(len(x)))+time_mat_pd.born
    series_temp2 = time_mat_pd.apply(lambda row: row.name,axis = 1)*time_mat_pd.pixlim.apply(lambda x: np.ones(len(x)))
    celltime =np.stack((np.concatenate(series_temp.values),np.concatenate(series_temp2.values))).T
    return celltime


#matrix, with column#1=absolute time,column#2=cell time, column#3=index, column#4=pix_max, column#5=pix_min
#pix_min and _max are corrected for tracklim and thus directly usable with images
def get_long_mat(time_mat_pd):
    tracklim = time_mat_pd.iloc[0].tracklim
    series_time = time_mat_pd.pixlim.apply(lambda x: np.arange(len(x)))+time_mat_pd.born
    series_time2 = time_mat_pd.pixlim.apply(lambda x: np.arange(len(x)))#cell time scale
    series_index = time_mat_pd.apply(lambda row: row.name,axis = 1)*time_mat_pd.pixlim.apply(lambda x: np.ones(len(x)))
    celltime =np.stack((np.concatenate(series_time.values),np.concatenate(series_time2.values),np.concatenate(series_index.values),
                      np.concatenate(time_mat_pd.pix_max.values)+tracklim,
                        np.concatenate(time_mat_pd.pix_min.values)+tracklim)).T
    return celltime


#load the cellid image as specified in the mom object using information of time_mat_pd
#of course if the cell doesn't exist at the given mom.time, an error occurs
def get_cell(time_mat_pd, cellid, mom, image = None):
    if image == None:
        image = mom.load_moma_im()
    
    im_size = image.shape
    im_middle = int((im_size[1]-1)/2)
    
    t= mom.time-time_mat_pd.iloc[cellid]['born']
    
    index1 = time_mat_pd.iloc[cellid]['pixlim'][t,0]+time_mat_pd.iloc[cellid]['tracklim']+1
    index2 = time_mat_pd.iloc[cellid]['pixlim'][t,1]+time_mat_pd.iloc[cellid]['tracklim']-1

    return image[index1:index2,im_middle-5:im_middle+6]


#detect spots in all cells
#to turn the output list into something understandable use:
#pd.DataFrame(time_mat_pd.iloc[16].spots,
    #columns = ['x', 'y', 'A', 'b', 'A_fit', 'x_fit', 'y_fit', 'sigma_fit', 'RSS', 'xcorr', 'x_box', 'y_box', 'time','cell_time', 'x_cell', 'y_glob'])
def find_spots(time_mat_pd, mom, sigmaXY, maxtime = None):
    #set spot fitting parameters
    #sigmaXY = 1.5
    fitbox = [int(np.ceil(4*sigmaXY)),int(np.ceil(4*sigmaXY))]
    gfilt = sp.make_g_filter(modelsigma=sigmaXY)
    glogfilt = sp.make_laplacelog(modelsigma=sigmaXY)

    #get matrix with cellids and boxes at all time points
    celltime = get_long_mat(time_mat_pd)

    #create an array to store a spot list for each cell
    spots = [[] for x in time_mat_pd.index]
    
    if maxtime is None:
        maxtime = mom.get_max_time()

    all_fits = []
    for t in range(0,maxtime):
        #print('time: '+str(t))
        mom.col = 2
        mom.time = t
        image = mom.load_moma_im()
        im_size = image.shape
        im_middle = int((im_size[1]-1)/2)
        image = image[:,im_middle-15:im_middle+16]

        conv = sp.spot_filter_convfft(image=image,templ=gfilt,logfilt=-glogfilt)

        spot_prop = sp.get_candidates(conv,image)
        fit_res = sp.fit_candidates(image, spot_prop, sigmaXY, show_fit = False, fit_type = 'B')
        fit_pd = pd.DataFrame(np.c_[fit_res,np.ones(len(spot_prop))*fitbox[0],np.ones(len(spot_prop))*fitbox[1],np.ones(len(spot_prop))*t],
                 columns=['A_fit','x_fit','y_fit','sigma_fit','RSS','xcorr','x_box','y_box','time'])

        spot_mat = pd.concat([spot_prop,fit_pd],axis=1)

        #absolute positions of spots
        abs_pos_x = (spot_mat.x+spot_mat.x_fit-spot_mat.x_box).values
        abs_pos_y = (spot_mat.y+spot_mat.y_fit-spot_mat.y_box).values
        #cell limits 
        cell_lims = celltime[celltime[:,0]==t,:]
        #find the index of the cell to which each spot belongs
        assigned = [cell_lims[[np.logical_and(p > x[4], p < x[3]) for x in cell_lims],:] for p in abs_pos_x]
        #assign spot to cell
        for ind, x in enumerate(assigned):
            if len(x)>0:
                toappend = np.append(spot_mat.iloc[ind].values,[x[0][1], 0.5*(x[0][3]+x[0][4])-abs_pos_x[ind],abs_pos_y[ind]])
                spots[int(x[0][2])].append(toappend)

        #all_fits.append(fit_res)
        '''fig,ax = plt.subplots(figsize=(20,20))
        plt.imshow(image)
        plt.plot(spot_prop.y,spot_prop.x,'ro')
        plt.show()'''
    spots = [np.stack(x) if len(x)>0 else [] for x in spots]
    time_mat_pd['spots'] = pd.Series(spots)
    
    return time_mat_pd


#track spots stored in time_mat_pd.spots
def track_spots(time_mat_pd):
    time_mat_pd['spots_num'] = np.nan
    time_mat_pd['spots_num'] = time_mat_pd['spots_num'].astype(object)
    time_mat_pd['spots_tracks'] = np.nan
    time_mat_pd['spots_tracks'] = time_mat_pd['spots_tracks'].astype(object)
    time_mat_pd['full_tracks'] = np.nan
    time_mat_pd['full_tracks'] = time_mat_pd['full_tracks'].astype(object)
    
    for index in time_mat_pd.index:#range(14,17):#
        if len(time_mat_pd.loc[0].spots)==0:
            continue
        unique, counts = np.unique(time_mat_pd.loc[0].spots[:,-2], return_counts=True)
        maxnumspots = np.max(counts)
        
        if (time_mat_pd.iloc[index].full_cellcycle)&(maxnumspots<10):
            mother_id = time_mat_pd.iloc[index].mother_id
            if mother_id>0:

                #recover mother info
                mother_id = int(mother_id)
                mother_time = time_mat_pd.iloc[mother_id]['Td']
                mother_Ld = time_mat_pd.iloc[mother_id].length[-1]

                daughter_Ld = time_mat_pd.iloc[index].length[-1]
                daughter_time = time_mat_pd.iloc[index]['Td']

                gdaughter1_id = time_mat_pd.index[time_mat_pd.genealogy == time_mat_pd.iloc[index].genealogy+'B']
                gdaughter1_id = gdaughter1_id[0] if len(gdaughter1_id)>0 else None
                gdaughter2_id = time_mat_pd.index[time_mat_pd.genealogy == time_mat_pd.iloc[index].genealogy+'T']
                gdaughter2_id = gdaughter2_id[0] if len(gdaughter2_id)>0 else None

                if (gdaughter1_id is None) or (gdaughter2_id is None):
                    continue
                gdaughter1_time = time_mat_pd.iloc[gdaughter1_id]['Td']
                gdaughter2_time = time_mat_pd.iloc[gdaughter2_id]['Td']
                
                len_mother_spots = len(time_mat_pd.iloc[mother_id]['spots'])
                len_gdaughter_spots = len(time_mat_pd.iloc[gdaughter1_id]['spots'])
                len_gdaughter_spots2 = len(time_mat_pd.iloc[gdaughter2_id]['spots'])

                #proceed if mother cell is ok
                if (not np.isnan(mother_Ld))&(mother_time>10)&(gdaughter1_time>5)&(gdaughter2_time>5)&(len_mother_spots>0)&(len_gdaughter_spots>0)&(len_gdaughter_spots2>0):
                    #find out if daughter is top/bottom
                    if time_mat_pd.iloc[index].genealogy[-1]=='B':
                        sign = -1
                        displace = -1/4
                    else: 
                        sign = 1
                        displace = 1/4


                    combined = np.empty((0,4))
                    
                    #keep time, x, y, A and limit the time points in mothe and grand-daughters
                    mother_spots = time_mat_pd.iloc[mother_id]['spots'][:,[13,14,15,4]]
                    mother_spots=mother_spots[mother_spots[:,0]>=mother_time-10]
                    daughter_spots = time_mat_pd.iloc[index]['spots'][:,[13,14,15,4]]
                    gdaughter_spots = time_mat_pd.iloc[gdaughter1_id]['spots'][:,[13,14,15,4]]
                    gdaughter_spots=gdaughter_spots[gdaughter_spots[:,0]<5]
                    gdaughter_spots2 = time_mat_pd.iloc[gdaughter2_id]['spots'][:,[13,14,15,4]]
                    gdaughter_spots2=gdaughter_spots2[gdaughter_spots2[:,0]<5]
                    
                    
                    if ((len(mother_spots)==0)or(len(gdaughter_spots)==0)or(len(gdaughter_spots2)==0)):
                        continue
                    
                    #create a composite spot ensemble made of mother-daugher-grand-daughters
                    #correct for positions (1/4 positions, 1/2 position etc.)
                    mother_spots = mother_spots[np.sign(mother_spots[:,1])==sign]
                    mother_spots[:,0] = mother_spots[:,0]-mother_time
                    combined = np.concatenate((combined,mother_spots))
                    
                    
                    daughter_spots[:,1] = daughter_spots[:,1]+displace*mother_Ld
                    combined = np.concatenate((combined,daughter_spots))
                    
                    
                    gdaughter_spots[:,1] = gdaughter_spots[:,1]+(displace*mother_Ld)-1/4*daughter_Ld
                    gdaughter_spots[:,0] = gdaughter_spots[:,0]+daughter_time
                    combined = np.concatenate((combined,gdaughter_spots))
                    
                    
                    gdaughter_spots2[:,1] = gdaughter_spots2[:,1]+(displace*mother_Ld)+1/4*daughter_Ld
                    gdaughter_spots2[:,0] = gdaughter_spots2[:,0]+daughter_time
                    combined = np.concatenate((combined,gdaughter_spots2))

                    #do the tracking
                    combined = combined[:,0:3]
                    spot_frame = pd.DataFrame(combined,columns=('frame','x','y'))
                    #spot_frame = pd.DataFrame(combined,columns=('frame','x','z','y'))
                    spot_tracked = trackpy.link_df(spot_frame,20,memory=5,link_strategy='nonrecursive',pos_columns=['x', 'y'])
                    spot_tracked = trackpy.filter_stubs(spot_tracked,5)


                    tracks = []
                    for tr in spot_tracked.particle.unique():
                        curr_track = spot_tracked[spot_tracked.particle==tr]
                        #interpolate missing values
                        tracks.append(np.c_[[np.interp(np.arange(curr_track.frame.min(),curr_track.frame.max()+1),curr_track.frame,curr_track[key]) for key in ['frame','x','y']]].transpose())

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

    return time_mat_pd


def length_fit_at_T(row, time_var):
    len_fit = np.nan
    if (~np.isnan(row['tau_fit']))&(~np.isnan(row[time_var])):
        len_fit = tgw.fun_expgrowht2(row[time_var],row['Lb_fit'],row['tau_fit'])
    return len_fit


#find the inter-initiation added volume per origin
def inter_initiations(row, time_mat_pd,t_var,l_var):
    if row['mother_id']>0:
    
        if (row[t_var]<0) and (time_mat_pd.loc[row['mother_id']][t_var]>0):
            L_inter_init = row[l_var]-0.5*time_mat_pd.loc[row['mother_id']][l_var]
            L_i_old = 0.5*time_mat_pd.loc[row['mother_id']][l_var]
        elif time_mat_pd.loc[row['mother_id']][t_var]>0:
            L_inter_init = row[l_var]-row['Lb_fit']+0.5*(time_mat_pd.loc[row['mother_id']].Ld_fit-time_mat_pd.loc[row['mother_id']][l_var])
            L_i_old = 0.5*time_mat_pd.loc[row['mother_id']][l_var]
        else:
            L_inter_init = np.nan
            L_i_old = np.nan
        return (L_inter_init, L_i_old)
    else:
        return (np.nan,np.nan)
    
    
'''def mother_var(row, time_mat_pd,var):
    if type(row['mother_id'])==str:
        return time_mat_pd.loc[row['mother_id']][var]
    else:
        return np.nan'''
    
    
def mother_var(row, time_mat_pd,var):
    if row['mother_id']>0:
        if row['mother_id'] in time_mat_pd.index:
            return time_mat_pd.loc[row['mother_id']][var]
        else:
            return np.nan
    else:
        return np.nan
    
    
def kymo(time_mat_pd, mom):
    
    all_images = []
    all_images2 = []
    if (time_mat_pd.at[c,'born']==0) or (time_mat_pd.at[c,'born']==-1) or (np.sum(time_mat_pd.at[c,'pos_GL'][:,0]==1)>0):
        kymo = np.zeros((100,100))
    else:
        proj_time = {}
        times = [x+time_mat_pd.at[c,'born'] for x in range(time_mat_pd.at[c,'pixlim'].shape[0])]
        for t in range(len(times)):
            
            mom.time=times[t]
            im_fluo = mom.load_moma_im(momainfo)
            
            momainfo['col'] = col1
            im_phase = ii.load_moma_im(momainfo)
            
            filtered = imp.spot_filter(im_fluo,momainfo)
            if ncc == False:
                im_fluo = filtered['amplitude']
            else:
                im_fluo = filtered['NCC']

            im_size = im_fluo.shape
            im_middle = int((im_size[1]-1)/2)

            if time_mat[c]['cell_len'][t]<5:
                index1 = time_mat[c]['pixlim'][t,0]+time_mat[c]['tracklim']+1
                index2 = time_mat[c]['pixlim'][t,1]+time_mat[c]['tracklim']-1
            else:
                index1 = time_mat[c]['mid_pos'][t] - int(time_mat[c]['cell_len'][t]/2)
                index2 = time_mat[c]['mid_pos'][t] + int(time_mat[c]['cell_len'][t]/2)

            proj_time[t] = np.max(np.array(im_fluo[index1:index2,im_middle-10:im_middle+11]),axis=1)
            all_images.append(im_fluo[index1:index2,im_middle-10:im_middle+11])
            all_images2.append(im_phase[index1:index2,im_middle-10:im_middle+11])

        maxlen = max([proj_time[k].shape[0] for k in proj_time])
        kymo = np.zeros((maxlen,len(times)))
        kymo[:] = np.NAN
        im_middle = int((kymo.shape[0]-1)/2)
        for k in range(len(times)):
            cur_len = proj_time[k].shape[0]
            if np.mod(cur_len,2):
                kymo[im_middle-int(cur_len/2):im_middle+int(cur_len/2)+1,k] = proj_time[k]
            else:
                kymo[im_middle-int((cur_len+1)/2)+1:im_middle+int((cur_len+1)/2)+1,k] = proj_time[k]

        
    return kymo, all_images, all_images2