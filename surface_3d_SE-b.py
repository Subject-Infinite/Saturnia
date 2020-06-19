import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
#from PIL import Image
import itertools # import permutations, product
from matplotlib.animation import FuncAnimation
from scipy.signal import find_peaks, peak_prominences, find_peaks_cwt
from PIL import Image, ImageDraw
#from skimage.morphology import extrema
#from skimage.feature import peak_local_max
#import hdbscan
import pandas as pd
from sklearn.cluster import DBSCAN
#from sklearn.neighbors import NearestNeighbors
from itertools import chain
#import peakutils

'''
def animate(frames,X,img2):
	#ax1.clear()
	#ax2.clear()
	#fig.clear()
	#ax1 = fig.add_subplot(1,2,1)
	ax1.plot(list(range(0,1920)),img2[X])
	#ax2 = fig.add_subplot(1,2,2)
	#plt.subplot(1,2,2)
	ax2.imshow(img2)
	x = list(range(0,1920))
	y = [X]*len(x)
	ax2.plot(x,y)
	#ax2.clear()
	#ax1.clear()
	#fig.clear()
'''

#################
#Detect features#
#################

im = "nm_t011_z140_c002.tif"
#im = "nm_t016_z001_c002.tif"
#im = "nm_t065_z211_c002.tif"
#im = "nm_t011_z244_c002.tif"
#im = "nm_t011_z082_c002.tif"
img2 = np.array(cv.imread(im,-1))
gausB_img2=cv.GaussianBlur(img2, (31,31),0) #denoise to make peak detection easier

nucleus_x=[]
nucleus_y=[]
nucleus_intense=[]
row_len = list(range(0,1920))
for xRow in range(0,1920):
	row = xRow
	xVal = gausB_img2[row]
	'''
	p_l_m=peak_local_max(xVal,threshold_rel=0.9,min_distance=100)
	plm_list=[]
	for a in p_l_m:
		plm_list.append(a[0])
	plm_list.reverse()
	for peak_index in plm_list:
		nucleus_x.append(peak_index)
		nucleus_y.append(xVal[peak_index])'''
	peaks = find_peaks(xVal,height=375,width=(20,900),distance=75,prominence=150)
	#print("peaks = {}".format(peaks))
	peaks_x = peaks[0]
	peaks_y = peaks[1].get('peak_heights')
	#####
	'''
	peak_cwt =  find_peaks_cwt(xVal, widths=np.arange(1,200),min_length=170)
	'''
	'''
	h_m = extrema.h_maxima(xVal,500)


	hm_list = []
	a_index = 0

	for a in h_m:
		if a==1:
			hm_list.append(row_len[a_index])
		else:
			pass
		a_index+=1
	if len(hm_list)>0:
		for a in hm_list:
			nucleus_x.append(a)
			nucleus_y.append(row)

	'''
	prominences = peak_prominences(xVal,peaks[0])
	#print("prominences = {}".format(prominences))
	contour_heights = xVal[peaks_x] - prominences[0]
	#print(peaks_x)
	#print(len(nucleus_x[0]))
	if len(peaks_x)>0:
		for a in peaks_x:
			nucleus_x.append(a)
			nucleus_y.append(row)
			nucleus_intense.append(xVal[a])
	else:
		continue

########################
#set up watershed seeds#
########################

#nucleus_x=list(np.array(nucleus_x).flat)
#nucleus_y=list(np.array(nucleus_y).flat)
#print(nucleus_x)
#print(nucleus_y)

peak_crd = pd.DataFrame(list(zip(nucleus_x,nucleus_y,nucleus_intense)),columns=['x','y','intensity'])

#print(peak_crd)
#clusterer = hdbscan.HDBSCAN(min_cluster_size=7,min_samples=1).fit(peak_crd)

clusterer = DBSCAN(eps=50,min_samples=5,algorithm='auto').fit(peak_crd)
#clusterer = NearestNeighbors(n_neighbors=5,radius=30).fit(peak_crd)
#print(clusterer)
#print(clusterer.labels_)
peak_crd['labels'] = clusterer.labels_
print(peak_crd)

def ellbound(dataframe,label,r_index):
	boundr=int(dataframe[str(label)][r_index])
	return boundr

#lab_grp=peak_crd.groupby(['labels']).mode()
#print("lab_group = ", lab_grp)

mid_val = peak_crd.groupby(['labels']).median() #find median (in y) per cluster (nucleus) to draw for watershed seed

#print("midval = ",mid_val,"end")

dbscan_labels =mid_val.index.tolist()
print("dbscan_labels",dbscan_labels)
labels_values_col=list(peak_crd.columns)
bright_window=pd.DataFrame(columns=labels_values_col)
for labels in dbscan_labels:
	#if labels==-1 or labels==0:
	if 7==8:
	#if labels==0:
		continue
	else:
		print("labels = ", labels)
		labels_values=peak_crd.loc[peak_crd['labels']==labels]
		#print("labels_values = {}".format(labels_values))
		#labels_values_col=list(labels_values.columns)
		print("labels_values_col",labels_values_col)
		loop_timer=0
		print("labels_values={},len(labels_values)={})".format(labels_values,len(labels_values)))
		median_list=[]
		median_iteration_list=[]
		for a in range(0,len(labels_values)-4):
			#print("loop_timer=",loop_timer)
			aa=labels_values['intensity'].iloc[a]
			ab=labels_values['intensity'].iloc[a+1]
			ac=labels_values['intensity'].iloc[a+2]
			ad=labels_values['intensity'].iloc[a+3]
			ae=labels_values['intensity'].iloc[a+4]

			print("ae={},ab={},ac={},ad={},ae={},loop_iteration={}".format(aa,ab,ac,ad,ae,loop_timer))
			slid_win_array = np.array([aa,ab,ac,ad,ae])
			sliding_window_median = np.median(slid_win_array)
			#sliding_window_max = np.amax(slid_win_array)

			print("sliding_window_median = ", sliding_window_median)

			sliding_window_max=np.amax(slid_win_array)
			median_list.append(sliding_window_median)
			median_iteration_list.append(a)


			indexes_for_new_frame = labels_values.index[labels_values['intensity']==sliding_window_max]
			print("indexes_for_new_frame = ",indexes_for_new_frame)
			input_dict={'x':labels_values.loc[indexes_for_new_frame[0],'x'],'y':labels_values.loc[indexes_for_new_frame[0],'y'],'intensity':labels_values.loc[indexes_for_new_frame[0],'intensity'],'labels':labels_values.loc[indexes_for_new_frame[0],'labels']}
			print("input_dict = ",input_dict)
			print(labels_values.loc[indexes_for_new_frame[0],'x'])
			#bright_window=pd.DataFrame(columns=labels_values_col)
			loop_timer+=1
			if loop_timer==(len(labels_values)-4):
				print("median_list = ", median_list)
				max_median_list=max(median_list)
				enumeration_l=[]
				list_mid_ticker=0
				for val in median_list:
					if val==max_median_list:
						enumeration_l.append(list_mid_ticker)
					list_mid_ticker+=1
				median_max_index=median_iteration_list[int(np.median(enumeration_l))]
				print("index of interest = ", median_max_index)
				print("max_median_list = ", max_median_list)
				a = median_max_index
				aa=labels_values['intensity'].iloc[a]
				ab=labels_values['intensity'].iloc[a+1]
				ac=labels_values['intensity'].iloc[a+2]
				ad=labels_values['intensity'].iloc[a+3]
				ae=labels_values['intensity'].iloc[a+4]

				slid_win_array = np.array([aa,ab,ac,ad,ae])
				sliding_window_max=np.amax(slid_win_array)
				print("sliding_window_max = ",sliding_window_max)
				sliding_window_df = pd.DataFrame(columns=labels_values_col)
				swd_input_dicta={'x':labels_values['x'].iloc[a],'y':labels_values['y'].iloc[a],'intensity':labels_values['intensity'].iloc[a],'labels':labels_values['labels'].iloc[a]}
				swd_input_dictb={'x':labels_values['x'].iloc[a+1],'y':labels_values['y'].iloc[a+1],'intensity':labels_values['intensity'].iloc[a+1],'labels':labels_values['labels'].iloc[a+1]}
				swd_input_dictc={'x':labels_values['x'].iloc[a+2],'y':labels_values['y'].iloc[a+2],'intensity':labels_values['intensity'].iloc[a+2],'labels':labels_values['labels'].iloc[a+2]}
				swd_input_dictd={'x':labels_values['x'].iloc[a+3],'y':labels_values['y'].iloc[a+3],'intensity':labels_values['intensity'].iloc[a+3],'labels':labels_values['labels'].iloc[a+3]}
				swd_input_dicte={'x':labels_values['x'].iloc[a+4],'y':labels_values['y'].iloc[a+4],'intensity':labels_values['intensity'].iloc[a+4],'labels':labels_values['labels'].iloc[a+4]}
				sliding_window_df=sliding_window_df.append(swd_input_dicta,ignore_index=True)
				sliding_window_df=sliding_window_df.append(swd_input_dictb,ignore_index=True)
				sliding_window_df=sliding_window_df.append(swd_input_dictc,ignore_index=True)
				sliding_window_df=sliding_window_df.append(swd_input_dictd,ignore_index=True)
				sliding_window_df=sliding_window_df.append(swd_input_dicte,ignore_index=True)

				indexes_for_new_frame = labels_values.index[labels_values['intensity']==sliding_window_max]
				print("sliding_window_df",sliding_window_df)

				#sliding_window_max2=sliding_window_df[sliding_window_df['intensity']==sliding_window_df['intensity'].max()]
				#print("sliding_window_max2",sliding_window_max2)
				#sliding_window_max3=sliding_window_max2['intensity']
				#print("sliding_window_max3=",sliding_window_max3)
				#mid_swm2=sliding_window_max2.median()
				#print("mid_swm2 = ", mid_swm2)
				input_dict={'x':labels_values.loc[indexes_for_new_frame[0],'x'],'y':labels_values.loc[indexes_for_new_frame[0],'y'],'intensity':labels_values.loc[indexes_for_new_frame[0],'intensity'],'labels':labels_values.loc[indexes_for_new_frame[0],'labels']}
				#bright_window.append({'x':labels_values.loc[indexes_for_new_frame[0],'x'],'y':labels_values.loc[indexes_for_new_frame[0],'y'],'intensity':labels_values.loc[indexes_for_new_frame[0],'intensity'],'labels':labels_values.loc[indexes_for_new_frame[0],'labels']},ignore_index=True)
				#pd.concat(bright_window,labels_values.loc[indexes_for_new_frame[0],:])
				bright_window=bright_window.append(input_dict,ignore_index=True)
				print("bright_window",bright_window)
#print("mid_val['x']={},mid_val['y']={}".format(mid_val['x'][0],mid_val['y'][0]))
#print(len(mid_val))

mid_val = bright_window

seed_kernel=np.ones((3,3),np.uint8)
i = Image.new("1",(1920,1920),"black")
wshed_seed = ImageDraw.Draw(i)
#print(int(mid_val['y'][13]))

#local area intensity scan

'''
			  o(perph_north)
			  |
			  |
	(perph_east)o-----O-----o(perph_west)
			  |
			  |
			  o(perph_south)


'''

i2 = Image.new("1",(1920,1920),"black") #set up background binary image. Using foreround detected points to generate isolation zones as 'unknown' areas.
bkgrd_seed = ImageDraw.Draw(i2)

'''
def mean_compare(modifier,cen,per1,per2,per3):
	perph_list = [per1,per2,per3]
	perph_list = modifier*np.array(perph_list)
	return cen>perph_list[0] and cen>perph_list[1] and cen>perph_list[2]
'''

def mean_compare_flex(modifier,central,perph_compare,compare_index): #to compare mean intensity of cental zone with mean intensity of peripheral zones (perph). If central intensity is brighter than 3(set by compare_index, is variable) of surrounding 4 perphs, return TRUE. perph_compare must be an iterable (lkisst/tuple), this represents a list of the peripheral zone means to compare central means against. modifier is a measure of how much brighter the central zone must be above the peripheral mean being compared, to be considered bright enough. A good ball park figure is 10% brighter, so multiply perph by 1.1.
	larger_than_perph=0
	for perph_zone_mean in perph_compare:
		if central>modifier*perph_zone_mean:
		#if central>(perph_zone_mean*(1+(central/modifier))):
		#if (central/perph_zone_mean)>=(1.02+(((10/59)*central)-(5079/59))/100):
		#if (central/perph_zone_mean)>(1+(central/modifier)):
			larger_than_perph+=1
		print("central={},perph_zone_mean={},larger_than_perph={}".format(central,perph_zone_mean,larger_than_perph))
	if larger_than_perph>=compare_index:
		return True
'''
	set_up_iterations=itertools.combinations(perph_compare,compare_index)
	iteration_list=[]
	for iteration_result in set_up_iterations:
		print(iteration_result)
		iteration_list.append(iteration_result)
	is_larger_than_tuple=0
	print("central=",central)
	for iteration_tuple in iteration_list:
		print("print iteration_tuple = {}".format(iteration_tuple))
		smaller_perph_mean=0
		for perph_zone_mean in iteration_tuple:
			print("print perph_zone_mean = {}".format(perph_zone_mean))
			if central>(modifier*perph_zone_mean):
				smaller_perph_mean+=1
			else:
				smaller_perph_mean = 0
		print("smaller_perph_mean= ",smaller_perph_mean)
		if smaller_perph_mean==len(iteration_tuple):
			is_larger_than_tuple+=1
		else:
			is_larger_than_tuple = 0
	print("is_larger_than_tuple=",is_larger_than_tuple)
	if is_larger_than_tuple>=1:
		return is_larger_than_tuple>=1
'''
mask=np.zeros(img2.shape,np.uint8)
for a in range(0,len(mid_val)-1):
	#mask=np.zeros(img2.shape,np.uint8)
	#print(nucleus_x[a],nucleus_y[a])
	markerdilate=3
	bk_dil=30 #radius of unknown exclusion zone for backgound isolation
	#print(int(mid_val['x'][a]),int(mid_val['y'][a]))
	y_b=ellbound(mid_val,'y',a)
	x_b=ellbound(mid_val,'x',a)
	#print(x_b,y_b)
	#wshed_seed.ellipse((int(mid_val['x'][a])-markerdilate,int(mid_val['y'][a])-markerdilate,int(mid_val['x'][a])+markerdilate,int(mid_val['y'][a])+markerdilate),"white")
	c_x = x_b
	c_y = y_b
	rad_cen=(c_x,c_y)
	print("rad_cen=",rad_cen)
	rad_len_cen=9
	rad_len_perph=9
	perph_circ_dist=29
	perph_circ_cent_north=(c_x,c_y-perph_circ_dist)
	perph_circ_cent_east=(c_x+perph_circ_dist,c_y)
	perph_circ_cent_south=(c_x,c_y+perph_circ_dist)
	perph_circ_cent_west=(c_x-perph_circ_dist,c_y)

	cen_circ=cv.circle(mask,rad_cen,rad_len_cen,255,-1)
	north_circ=cv.circle(mask,perph_circ_cent_north,rad_len_perph,254,-1)
	east_circ=cv.circle(mask,perph_circ_cent_east,rad_len_perph,253,-1)
	south_circ=cv.circle(mask,perph_circ_cent_south,rad_len_perph,252,-1)
	west_circ=cv.circle(mask,perph_circ_cent_west,rad_len_perph,251,-1)

	where_cen=np.where(mask==255)
	where_north=np.where(mask==254)
	where_east=np.where(mask==253)
	where_south=np.where(mask==252)
	where_west=np.where(mask==251)
#	print("where_cen={}".format(where_cen))
	img_where_cen=img2[where_cen[0],where_cen[1]]
	img_where_north=img2[where_north[0],where_north[1]]
	img_where_east=img2[where_east[0],where_east[1]]
	img_where_south=img2[where_south[0],where_south[1]]
	img_where_west=img2[where_west[0],where_west[1]]
#	print("img_where_cen = {}".format(img_where_cen))
	mean_cen=np.mean(img_where_cen)
	concat_perph=list(chain(img_where_north,img_where_east,img_where_south,img_where_west))
#	print("concat_perph = {}".format(concat_perph))
	mean_perph=np.mean(concat_perph)
	print("mean_cen = {}, mean_perph = {}".format(mean_cen,mean_perph))
	mean_north=np.mean(img_where_north);mean_east=np.mean(img_where_east);mean_south=np.mean(img_where_south);mean_west=np.mean(img_where_west)
#	mean_north=np.median(img_where_north);mean_east=np.median(img_where_east);mean_south=np.median(img_where_south);mean_west=np.median(img_where_west)

	peripheral_means=[mean_north,mean_east,mean_south,mean_west]
	modifier=1.1
	print("mean_compare_flex: ", mean_compare_flex(modifier,mean_cen,peripheral_means,3))

#	if mean_compare(modifier,mean_cen,mean_north,mean_east,mean_south) or mean_compare(modifier,mean_cen,mean_north,mean_east,mean_west) or mean_compare(modifier,mean_cen,mean_north,mean_west,mean_south) or mean_compare(modifier,mean_cen,mean_east,mean_south,mean_west):

	if mean_compare_flex(modifier,mean_cen,peripheral_means,3):
		print("wololo")
#	if mean_cen>(mean_perph*1.15):
		pass
	else:
		continue

	wshed_seed.ellipse((x_b-markerdilate,y_b-markerdilate,x_b+markerdilate,y_b+markerdilate),'white')
	bkgrd_seed.ellipse((x_b-bk_dil,y_b-bk_dil,x_b+bk_dil,y_b+bk_dil),'white')
	#wshed_seed.point((ellbound(mid_val,'x',a),ellbound(mid_val,'y',a)),'white')
#group by cluster label
#find median y per goup and rest of coordinate with it
#put these coords in new list and use these coordinates to plot watershed seeds

#markerdilate = 1
#wshed_seed.ellipse((mid_val['x']-markerdilate,mid_val['y']-markerdilate,mid_val['x']+markerdilate,mid_val['y']+markerdilate),'white')



wshed_seed = np.array(i)
wshed_seed_th = np.array(wshed_seed)
wshed_seed_th = np.multiply(wshed_seed_th,1).astype(np.uint8)


bkgrd_seed = np.array(i2)
bkgrd_seed_th = np.array(bkgrd_seed)
bkgrd_seed_th = np.multiply(bkgrd_seed_th,1).astype(np.uint8)
#wshed_seed_2=cv.distanceTransform(wshed_seed,cv.DIST_L2,5)
#wshed_seed_th=cv.threshold(wshed_seed_th,12,255,cv.THRESH_BINARY)
#wshed_seed_th=wshed_seed_th[1]
#wshed_seed_th=np.multiply(wshed_seed_th,1).astype(np.uint8)
#wshed_seed=cv.erode(wshed_seed,seed_kernel,iterations=3)
#print(wshed_seed)

'''
seed_contours,_=cv.findContours(wshed_seed,cv.RETR_EXTERNAL,2)
c_length=len(seed_contours)
status=np.zeros((c_length,1))
for i,c1 in enumerate(seed_contours):
	x=i
	if i != c_length-1:
		for j,c2 in enumerate(seed_contours[i+1:]):
			x+=1
			dist = find_if_close(c1,c2)
			if dist == True:
				val = min(status[i],status[x])
				status[x] = status[i] = val
			else:
				if status[x]==status[i]:
					status[x] = i+1
unified=[]
maximum=int(status.max())+1
for i in xrange(maximum):
	pos=np.where(status==i)[0]
	if pos.size != 0:
		cont = np.vstack(seed_contours[i] for i in pos)
		hull = cv.convexHull(cont)
		unified.append(hull)
contour_show = cv.drawContours(img2,unified,-1,(0,255,0),2)
#contour_another = cv.drawContours()
'''

##################
#watershed module#
##################


#print(wshed_seed)
#print("img2 dtype = {}, wshed_seed dtype = {}".format(img2.dtype,wshed_seed.dtype))

back_img2=(img2/256).astype(np.uint8)
back_img2=cv.threshold(back_img2,1,255,cv.THRESH_BINARY)

kernel = np.ones((3,3),np.uint8)

#print("img2={}".format(back_img2))

back_img2=cv.dilate(back_img2[1],kernel,iterations=2)

bkgrd_seed_th = bkgrd_seed_th*255
back_img2=bkgrd_seed_th
#print("bkgrd_seed= ",bkgrd_seed_th)



#print("back_img2.shape={}, wshed_seed.shape={}".format(back_img2.shape,wshed_seed.shape))
#print("back_img2 dtype = {}, wshed_seed dtype = {}".format(back_img2.dtype,wshed_seed.dtype))


unknown=cv.subtract(back_img2,wshed_seed_th)

_, markers = cv.connectedComponents(wshed_seed_th)
markers = markers+10

markers[unknown==255]=0
eightB_img2=(img2/255).astype("uint8")
#print(img2.dtype)
#print(markers.dtype)
#print(img2.shape)
#print(markers)
eightB_img2=cv.cvtColor(eightB_img2,cv.COLOR_GRAY2RGB) #my images are single channel but opencv watershed requires 3 channel input
wshed_show = cv.watershed(eightB_img2,markers)
wshed_show_og = wshed_show
print(wshed_show)
print(len(np.unique(wshed_show)))
uniq_wshed = np.unique(wshed_show)

#determine the amount of pixels per watershed region, determines area. we want to remove watersheds above certain value. Turn pixel values above certain value to -1
#tidying watershed output


wshed_mask=np.zeros(img2.shape,np.uint8)
corona_wshed_delta=1.05
for a in uniq_wshed:
	occur=np.count_nonzero(wshed_show==a)
	print("{} frequency = {}".format(a,occur))
	if occur > 5000 or occur < 30: ######################################################## size threshold
		wshed_show = np.where(wshed_show==a,-1,wshed_show)
		pass
	wshed_mask_im=np.where(wshed_show==a,np.uint8(255),np.uint8(0))
	wshed_mask=np.where(wshed_show==a)
	#print(wshed_mask)
	wshed_intensities=img2[wshed_mask[0],wshed_mask[1]]
	#print(wshed_intensities)
	wshed_mask_dilate=cv.dilate(wshed_mask_im,kernel,iterations=10)
	wshed_corona=cv.subtract(wshed_mask_dilate,wshed_mask_im)
	wshed_corona_mask=np.where(wshed_corona>0)
	print("wshed_corona = ", wshed_corona_mask)
	wshed_corona_intensities=img2[wshed_corona_mask[0],wshed_corona_mask[1]]
	print("wshed_corona_intensities = ", wshed_corona_intensities)
	print("mean wshed intensity = {}, mean corona intensity = {}".format(np.mean(wshed_intensities),np.mean(wshed_corona_intensities)))
	if np.mean(wshed_intensities)<(corona_wshed_delta*np.mean(wshed_corona_intensities)):
		wshed_show = np.where(wshed_show==a,-1,wshed_show)

print(uniq_wshed)
#wshed_show_size = measure.regionprops(wshed_show)
#print(wshed_show)
#print(wshed_show_size[1])
wshed_bound=cv.inRange(wshed_show,-1,-1)
#wshed_show_gray = cv.cvtColor(wshed_show,cv.COLOR_RGB2GRAY)
#print(wshed_show.dtype)
#print(wshed_show)
wshed_show_g = wshed_show
wshed_show = wshed_show.astype(np.uint8)
wshed_show = cv.threshold(wshed_show,254,255,cv.THRESH_BINARY)
#wshed_show=cv.dilate(wshed_show[1],seed_kernel,iterations=1)
'''
print("wshed_show",wshed_show)
for a in range(0,len(wshed_show[1])):
	print(wshed_show[1][a])
'''
contours_func=cv.findContours(wshed_show[1],cv.RETR_LIST,cv.CHAIN_APPROX_NONE)
contours=contours_func[0]
#print(contours)
#print(len(contours[0]))
#print(contours[10])
contour_draw=cv.drawContours(eightB_img2,contours,-1,(255,255,255),1)
#cv.imshow("contours",contours[1].astype("uint8"))
#cv.waitKey(0)


'''
wshed_comp=cv.connectedComponentsWithStats(wshed_bound,connectivity=4)
stats = wshed_comp[3]
print((stats))
#print(len(wshed_comp))
#wshed_size_th = wshed_show_size['area'<500]
'''
####################
####Plot outputs####
####################
fig = plt.figure()

spec_G=gridspec.GridSpec(ncols=3,nrows=2,figure=fig)

#spec_G=gridspec.GridSpec(ncols=2,nrows=1,figure=fig)
#ax4=fig.add_subplot(spec_G[0,0])
#ax2=fig.add_subplot(spec_G[0,1])
#ax0.imshow(img2)

ax0 = fig.add_subplot(spec_G[0,0])
ax1 = fig.add_subplot(spec_G[0,1])
ax2 = fig.add_subplot(spec_G[0,2])
ax0.imshow(img2)


'''
ax1.plot(list(range(0,1920)),xVal)
ax1.plot(peaks_x,peaks_y,"x")
ax1.vlines(x=peaks_x,ymin=contour_heights,ymax=xVal[peaks_x])
ax1.hlines(y=peaks[1].get("width_heights"),xmin=peaks[1].get("left_ips"),xmax=peaks[1].get("right_ips"))
'''
ax1.imshow(gausB_img2)
x = list(range(0,1920))
y = [row]*len(x)
#y_plot = [row]*len(peaks[0])
ax2.plot(x,y)
#ax2.plot(peaks[0],y_plot,"x")
ax2.scatter(peak_crd['x'],peak_crd['y'],marker='x',c=peak_crd['labels'])
ax2.imshow(img2)

#ani = FuncAnimation(fig,animate,frames=1920,fargs=(x,img2),interval=150)
ax3 = fig.add_subplot(spec_G[1,0])

ax3.imshow(wshed_seed)

ax4 = fig.add_subplot(spec_G[1,1])
#ax4.imshow(wshed_show_g)
#ax4.imshow(wshed_bound)
ax4.imshow(back_img2)
ax5 = fig.add_subplot(spec_G[1,2])

ax5.imshow(contour_draw)
#ax5.imshow(wshed_corona)

plt.show()
plt.clf
