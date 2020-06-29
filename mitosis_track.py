#script to track where mitosis occurs per generations. Construct 3D graph (in 2 dimensions, 3rd dimension depicted as pointcolour. plot position of mitosis events in 2D space, colour code with time. Using shades of grey for ease of viewing.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
from matplotlib import cm

mitosisData = pd.read_csv("tracks_.csv")

mitosisSUB=mitosisData[mitosisData["GEN"].str.match("\[1")] #take your generation of interest. reduce dataframe to contain only nuclei of yoru genreation of interest, and those that are registered as dividing (i.e the ones which have a generation + 1 encoded in their track.
mitosisSUB=mitosisSUB[mitosisSUB['GEN'].str.contains('2')] #find if it is tracked to mitosis. this is just going to be generation of interest+1 at the end of the string/list.  these 2 values are important as they are modifiable parameters which decide which generation you are finding the time point for mitosos for.

mitTimeList=[]
for a in range(0,len(mitosisSUB)):
	valueInList=mitosisSUB["GEN"].iloc[a]
	genToList=ast.literal_eval(valueInList)
	xLOC=mitosisSUB["X"].iloc[a]
	xVAL=ast.literal_eval(xLOC)
	yLOC=mitosisSUB["Y"].iloc[a]
	yVAL=ast.literal_eval(yLOC)
	zLOC=mitosisSUB["Z"].iloc[a]
	zVAL=ast.literal_eval(zLOC)
	tLOC=mitosisSUB["T"].iloc[a]
	tVAL=ast.literal_eval(tLOC)
	genLEN=len(genToList)
	print("{},{},{},{}".format(xVAL[-1],yVAL[-1],zVAL[-1],tVAL[-1],genLEN))
	
	plotcrd=(xVAL[-1],yVAL[-1],zVAL[-1],genLEN) #create tuple with all the coordinates we could need, x,y,z,T
	mitTimeList.append(plotcrd)
print(mitTimeList)
for b in mitTimeList:
	print("b= ", b)
	plt.scatter(b[1],b[2],c=cm.binary(b[3]*15),s=100) #need to multiply the mitosis time value by 15 as the colour pallette doesn't register the small time differences. scaling up puts distance between time points and these register better on colour scale. make points nice and big to emphasise colours.
	
plt.show()
