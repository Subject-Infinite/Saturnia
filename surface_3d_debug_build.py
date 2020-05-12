import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import itertools # import permutations, product
from matplotlib.animation import FuncAnimation
from scipy.signal import find_peaks, peak_prominences
from PIL import Image, ImageDraw

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

im = "nm_t011_z140_c002.tif"
#im = "green_.tif"
row = 948
#col = 631
img2 = np.array(cv.imread(im,-1)) #cv.IMREAD_UNCHANGED))
#img2 = (img2/256).astype(np.uint8)
#print(img2.dtype)
img2 = cv.GaussianBlur(img2, (31,31),0)
#print(img2.dtype)
xVal = img2[row]

peaks = find_peaks(xVal,height=500,width=(20,100),distance=75,prominence=150)
print("peaks = {}".format(peaks))
peaks_x = peaks[0]
peaks_y = peaks[1].get('peak_heights')

prominences = peak_prominences(xVal,peaks[0])
print("prominences = {}".format(prominences))
contour_heights = xVal[peaks_x] - prominences[0]

'''
print("peaks = {}".format(peaks))
print("peaks[0] = {}".format(peaks_x))
print("peaks[1] = {}".format(peaks_y))
'''

fig = plt.figure()
ax1 = fig.add_subplot(1,3,1)
ax2 = fig.add_subplot(1,3,2)
#ax2 = ax1b.add_subplot(1,4,1)

ax1.plot(list(range(0,1920)),xVal)
ax1.plot(peaks_x,peaks_y,"x")
ax1.vlines(x=peaks_x,ymin=contour_heights,ymax=xVal[peaks_x])
ax1.hlines(y=peaks[1].get("width_heights"),xmin=peaks[1].get("left_ips"),xmax=peaks[1].get("right_ips"))
x = list(range(0,1920))
y = [row]*len(x)
y_plot = [row]*len(peaks[0])
ax2.plot(x,y)
ax2.plot(peaks[0],y_plot,"x")
ax2.imshow(img2)
#####
y_line=[row]*len(peaks_x)
i = Image.new("1",(1920,1920),"black")
wshed_seed = ImageDraw.Draw(i)
for a in range(0,len(peaks[0])):
	wshed_seed.ellipse((peaks_x[a]-3,y_line[a]-3,peaks_x[a]+3,y_line[a]+3),"white")
wshed_seed = np.array(i)
print(wshed_seed)
#ax3 = fig.add_subplot(1,3,3)
#ax3.imshow(wshed_seed)
#####

#ani = FuncAnimation(fig,animate,frames=1920,fargs=(x,img2),interval=150)

#wshed_seed.dtype='int32'
wshed_seed=np.multiply(wshed_seed,1).astype(np.uint8)
print(wshed_seed)
print("img2 dtype = {}, wshed_seed dtype = {}".format(img2.dtype,wshed_seed.dtype))

back_img2=(img2/256).astype(np.uint8)
back_img2=cv.threshold(back_img2,1,255,cv.THRESH_BINARY)
kernel = np.ones((3,3),np.uint8)

print("img2={}".format(back_img2))

back_img2=cv.dilate(back_img2[1],kernel,iterations=10)

#print("back_img2.shape={}, wshed_seed.shape={}".format(back_img2.shape,wshed_seed.shape))
#print("back_img2 dtype = {}, wshed_seed dtype = {}".format(back_img2.dtype,wshed_seed.dtype))


unknown=cv.subtract(back_img2,wshed_seed)

_, markers = cv.connectedComponents(wshed_seed)
markers = markers+10

markers[unknown==255]=0
img2=(img2/255).astype("uint8")
print(img2.dtype)
print(markers.dtype)
print(img2.shape)
print(markers)
img2=cv.cvtColor(img2,cv.COLOR_GRAY2RGB) #my images are single channel but opencv watershed requires 3 channel input
wshed_show = cv.watershed(img2,markers)



ax3 = fig.add_subplot(1,3,3)
ax3.imshow(wshed_show)

#wshed = cv.watershed(img2,wshed_seed)


plt.show()
plt.clf
