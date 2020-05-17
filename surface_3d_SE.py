import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
from PIL import Image
import itertools # import permutations, product
from matplotlib.animation import FuncAnimation
from scipy.signal import find_peaks, peak_prominences
from PIL import Image, ImageDraw
#from skimage import measure
#import hdbscan
import pandas as pd
from sklearn.cluster import DBSCAN
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
img2 = np.array(cv.imread(im,-1))
gausB_img2=cv.GaussianBlur(img2, (31,31),0) #denoise to make peak detection easier

nucleus_x=[]
nucleus_y=[]

for xRow in range(0,1920):
	row = xRow
	xVal = gausB_img2[row]

	peaks = find_peaks(xVal,height=500,width=(20,100),distance=75,prominence=150)
	#print("peaks = {}".format(peaks))
	peaks_x = peaks[0]
	peaks_y = peaks[1].get('peak_heights')

	prominences = peak_prominences(xVal,peaks[0])
	#print("prominences = {}".format(prominences))
	contour_heights = xVal[peaks_x] - prominences[0]
	#print(peaks_x)
	#print(len(nucleus_x[0]))
	if len(peaks_x)>0:
		for a in peaks_x:
			nucleus_x.append(a)
			nucleus_y.append(row)
	else:
		continue
	'''
	print("peaks = {}".format(peaks))
	print("peaks[0] = {}".format(peaks_x))
	print("peaks[1] = {}".format(peaks_y))
	'''

########################
#set up watershed seeds#
########################
'''
def find_if_close(c1,c2):
	row1,row2=c1.shape[0],c2.shape[0]
	for i in xrange(row1):
		for j in xrange(row2):
			dist=np.linalg.norm(c1[i]-c2[j])
			if abs(dist)<20:
				return True
			elif i==row1-1 and j==row2-1:
				return False
'''

#nucleus_x=list(np.array(nucleus_x).flat)
#nucleus_y=list(np.array(nucleus_y).flat)
#print(nucleus_x)
#print(nucleus_y)

peak_crd = pd.DataFrame(list(zip(nucleus_x,nucleus_y)),columns=['x','y'])
#print(peak_crd)
#clusterer = hdbscan.HDBSCAN(min_cluster_size=7,min_samples=1).fit(peak_crd)
clusterer = DBSCAN(eps=20,min_samples=10).fit(peak_crd)
#print(clusterer)
#print(clusterer.labels_)
peak_crd['labels'] = clusterer.labels_
#print(peak_crd)

def ellbound(dataframe,label,r_index):
	boundr=int(dataframe[str(label)][r_index])
	return boundr

mid_val = peak_crd.groupby(['labels']).median() #find median (in y) per cluster (nucleus) to draw for watershed seed
#print(mid_val)
#print(len(mid_val))

seed_kernel=np.ones((3,3),np.uint8)
i = Image.new("1",(1920,1920),"black")
wshed_seed = ImageDraw.Draw(i)
#print(int(mid_val['y'][13]))
for a in range(0,len(mid_val)-1):
	#print(nucleus_x[a],nucleus_y[a])
	markerdilate=3
	#print(int(mid_val['x'][a]),int(mid_val['y'][a]))
	y_b=ellbound(mid_val,'y',a)
	x_b=ellbound(mid_val,'x',a)
	#print(x_b,y_b)
	#wshed_seed.ellipse((int(mid_val['x'][a])-markerdilate,int(mid_val['y'][a])-markerdilate,int(mid_val['x'][a])+markerdilate,int(mid_val['y'][a])+markerdilate),"white")
	wshed_seed.ellipse((x_b-markerdilate,y_b-markerdilate,x_b+markerdilate,y_b+markerdilate),'white')
	#wshed_seed.point((ellbound(mid_val,'x',a),ellbound(mid_val,'y',a)),'white')
#group by cluster label
#find median y per goup and rest of coordinate with it
#put these coords in new list and use these coordinates to plot watershed seeds

#markerdilate = 1
#wshed_seed.ellipse((mid_val['x']-markerdilate,mid_val['y']-markerdilate,mid_val['x']+markerdilate,mid_val['y']+markerdilate),'white')



wshed_seed = np.array(i)
wshed_seed_th = np.array(wshed_seed)
wshed_seed_th = np.multiply(wshed_seed_th,1).astype(np.uint8)

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
for a in uniq_wshed:
	occur=np.count_nonzero(wshed_show==a)
	print("{} frequency = {}".format(a,occur))
	if occur > 5000:
		wshed_show = np.where(wshed_show==a,-1,wshed_show)
print(uniq_wshed)
#wshed_show_size = measure.regionprops(wshed_show)
#print(wshed_show)
#print(wshed_show_size[1])
wshed_bound=cv.inRange(wshed_show,-1,-1)
#wshed_show_gray = cv.cvtColor(wshed_show,cv.COLOR_RGB2GRAY)
#print(wshed_show.dtype)
#print(wshed_show)
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
contour_draw=cv.drawContours(eightB_img2,contours,-1,(255,255,255),3)
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
spec_G=gridspec.GridSpec(ncols=2,nrows=2,figure=fig)
ax1 = fig.add_subplot(spec_G[0,0])
ax2 = fig.add_subplot(spec_G[0,1])
ax1.plot(list(range(0,1920)),xVal)
ax1.plot(peaks_x,peaks_y,"x")
ax1.vlines(x=peaks_x,ymin=contour_heights,ymax=xVal[peaks_x])
ax1.hlines(y=peaks[1].get("width_heights"),xmin=peaks[1].get("left_ips"),xmax=peaks[1].get("right_ips"))

x = list(range(0,1920))
y = [row]*len(x)
y_plot = [row]*len(peaks[0])
ax2.plot(x,y)
ax2.plot(peaks[0],y_plot,"x")
ax2.scatter(peak_crd['x'],peak_crd['y'],marker='x',c=peak_crd['labels'])
ax2.imshow(img2)

#ani = FuncAnimation(fig,animate,frames=1920,fargs=(x,img2),interval=150)
ax3 = fig.add_subplot(spec_G[1,0])
ax3.imshow(wshed_seed)
ax4 = fig.add_subplot(spec_G[1,1])
#ax4.imshow(wshed_show[1])
#ax4.imshow(wshed_bound)
#ax4.imshow(wshed_show_og)
ax4.imshow(contour_draw)

plt.show()
plt.clf
