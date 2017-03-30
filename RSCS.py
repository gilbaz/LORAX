# Academic Paper - CVPR 2017 "3D Point Cloud Registration for Localization using a Deep Neural Network Auto-Encoder"
# Authors - Gil Elbaz, Tamar Avraham, Anath Fischer
# --------------------------------------------------
# Key Stage of LORAX Point Cloud Registration Algorithm
# By utilizing the RSCS method the point clouds can be analyzed as SuperPoints--local subsets of points---
# useful for many applications, including PC registration when dealing with large point cloud datasets
# --------------------------------------------------
# Simple implementation of RSCS function in python3.5
import numpy as np
import random
import math
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#--- RSCS Functions ---
def createRandomSphereCoverSet(pntCloud,coverageLim = 0.95,coverSphereRad = 1):
    # Main RSCS function - Runs Random Sphere Cover Set (RSCS) Algorithm
    # INPUT: pntCloud is a list of point tuples Nx3 [(x1,y1,z1),(x2,y2,z2),...(xn,yn,zn)]
    # coverageLim is the percent of all points covered, coverSphereRad is the radius of the cover sphere
    # OUTPUT: superPointCloud is a list of point tuples with a list of labels included,
    # output size -> nx4 (n<=N) with list of k super-point labels
    # output example ->  [(x1,y1,z1,[l1,..lk]),(x2,y2,z2,[l1,..lk]),...(xn,yn,zn,[l1,..lk])]
    pntCloudFull = getLabeledFormat(pntCloud)
    spLabel = 0
    while getCoveragePercent(pntCloudFull) < coverageLim:
        centerPnt = chooseRandNonCoveredPnt(pntCloudFull)
        pntCloudFull = labelPointsInRad(centerPnt, coverSphereRad, pntCloudFull, spLabel)
        spLabel +=1
    return pntCloudFull, spLabel

def createRandomSphereCoverSetFixedNum(pntCloud,superPointNum=20, pointsInSP=10, coverSphereRad=1):
    # Main RSCS function - Runs Random Sphere Cover Set (RSCS) Algorithm - Augmented for other extra use cases
    # INPUT: pntCloud is a list of point tuples Nx3 [(x1,y1,z1),(x2,y2,z2),...(xn,yn,zn)]
    # superPointNum is the number of SP, pointsInSP is the number of points in each SP,
    # coverSphereRad is the radius of the CoverSphere
    # OUTPUT: superPointCloud is a list of point tuples with a list of labels included,
    # output size -> nx4 (n<=N) with list of k super-point labels
    # output example ->  [(x1,y1,z1,[l1,..lk]),(x2,y2,z2,[l1,..lk]),...(xn,yn,zn,[l1,..lk])]
    pntCloudFull = getLabeledFormat(pntCloud)
    spLabel = 0
    checkLoop = 0
    while spLabel < superPointNum:
        centerPnt = chooseRandNonCoveredPnt(pntCloudFull)
        pntCloudFull, hasChanged = labelNumPointsInRad(centerPnt, coverSphereRad, pntCloudFull, spLabel, pointsInSP)
        if hasChanged:
            spLabel += 1
            checkLoop = 0
        checkLoop +=1
        if checkLoop > superPointNum*5:
            print('Error: Enter Valid Input Parameters')
            return pntCloudFull, spLabel
    return pntCloudFull, spLabel

def getLabeledFormat(pntCloud):
    # Add Super Point Label List to standard input format
    # INPUT: Nx3 [(x1,y1,z1),(x2,y2,z2),...(xn,yn,zn)]
    # OUTPUT: Nx4 [(x1,y1,z1,[]),(x2,y2,z2,[]),...(xn,yn,zn,[])]
    labelPntCloud = []
    for pnt in pntCloud:
        newPnt = (*pnt, [])
        labelPntCloud.append(newPnt)
    return labelPntCloud

def chooseRandNonCoveredPnt(pntCloud):
    # Choose a random non-covered point (with an empty label list)
    # INPUT: pntCloud [(x1,y1,z1,[l1,..lk]),(x2,y2,z2,[l1,..lk]),...(xn,yn,zn,[l1,..lk])]
    # OUTPUT: point (x,y,z,[])
    nonCoveredList = [pnt for pnt in pntCloud if len(pnt[3])==0]
    newCoverPoint = random.choice(nonCoveredList)
    return newCoverPoint

def labelPointsInRad(centerPnt, coverSphereRad, pntCloud, labelNum):
    # Label Points in radius as uniform Super Point
    # INPUT: centerPnt (x,y,z), coverSphereRad is scalar, pntCloud is a list of point tuples Nx4, labelNum - scalar (int) Superpoint label
    # OUTPUT: list of points within radius (n2<=N)  [(x1,y1,z1,[l1,..lk]),(x2,y2,z2,[l1,..lk]),...(xn2,yn2,zn2,[l1,..lk])]
    for pnt in pntCloud:
        dist = math.sqrt((pnt[0]-centerPnt[0])**2 + (pnt[1]-centerPnt[1])**2 + (pnt[2]-centerPnt[2])**2)
        if dist <= coverSphereRad:
            pnt[3].append(labelNum)
    return pntCloud

def labelNumPointsInRad(centerPnt, coverSphereRad, originPntCloud, labelNum, numLabel):
    # Label Points in radius as uniform Super Point
    # INPUT: centerPnt (x,y,z), coverSphereRad is scalar, pntCloud is a list of point tuples Nx4, labelNum - scalar (int) Superpoint label
    # OUTPUT: list of points within radius (n2<=N)  [(x1,y1,z1,[l1,..lk]),(x2,y2,z2,[l1,..lk]),...(xn2,yn2,zn2,[l1,..lk])]
    pntCloud = copy.deepcopy(originPntCloud)
    numPointsInSP = 0
    for pnt in pntCloud:
        dist = math.sqrt((pnt[0]-centerPnt[0])**2 + (pnt[1]-centerPnt[1])**2 + (pnt[2]-centerPnt[2])**2)
        if numPointsInSP < numLabel:
            if dist <= coverSphereRad:
                pnt[3].append(labelNum)
                numPointsInSP +=1
    hasChanged = True
    if numPointsInSP<numLabel:
        print('Error: Raise Radius - Not enough points ('+str(numPointsInSP)+') to meet numLabel requirement of '+str(numLabel))
        return originPntCloud, False
    return pntCloud, hasChanged

def getCoveragePercent(pntCloud):
    # Get percent of points covered
    # INPUT: pntCloud [(x1,y1,z1,[l1,..lk]),(x2,y2,z2,[l1,..lk]),...(xn,yn,zn,[l1,..lk])]
    # OUTPUT: scalar (float) - percent of points covered
    numPntNonCovered = len([pnt for pnt in pntCloud if len(pnt[3]) == 0])
    totalPnt = len(pntCloud)
    return (totalPnt-numPntNonCovered)/totalPnt

#--- Display ---
def displaySP(pntCloud,curAxis):
    # Display Super Point Cloud
    # INPUT: Super Point Point Cloud Nspx4 [(x1,y1,z1,[]),(x2,y2,z2,[]),...(xn,yn,zn,[])]
    # OUTPUT: Display
    pntX = [pnt[0] for pnt in pntCloud if len(pnt[3]) > 0]
    pntY = [pnt[1] for pnt in pntCloud if len(pnt[3]) > 0]
    pntZ = [pnt[2] for pnt in pntCloud if len(pnt[3]) > 0]
    col = random.sample(range(1, 100), 3)
    curAxis.scatter(pntX, pntY, pntZ, color=[float(col[0])/100.0, float(col[1])/100.0, float(col[2])/100.0], marker='o')

def displayAllSP(superPntList):
    # Visualize Results
    fig = plt.figure()  # create figure
    ax = fig.add_subplot(111, projection='3d')  # initialize 3D axis
    m1 = max([max(pnt[3]) for pnt in superPntList if len(pnt[3]) > 0])
    for i in range(m1):  # for each SP label
        pntList = [pnt for pnt in superPntList if i in pnt[3]]
        displaySP(pntList, ax)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.waitforbuttonpress()

#--- Example Run ---
def createRandomCloud(n = 10000):
    # Create Random Point Cloud with n points
    # INPUT: scalar n - number of points
    # OUTPUT: point cloud list of point tuples, Nx3 [(x1,y1,z1),(x2,y2,z2),...(xn,yn,zn)]
    pntCloud = []
    for i in range(n):
        pntX = random.randrange(0, 100)
        pntY = random.randrange(0, 100)
        pntZ = random.randrange(0, 100)
        pntCloud.append((pntX, pntY, pntZ))
    return pntCloud

def runExample():
    # Run example SP division
    pntCloud = createRandomCloud(10000)
    superPntList, numSP = createRandomSphereCoverSet(pntCloud, coverSphereRad=30)
    #superPntList, numSP = createRandomSphereCoverSetFixedNum(pntCloud,superPointNum=50,pointsInSP=100,coverSphereRad=30)
    print('total points:' + str(len(superPntList)))
    print('total super points:' + str(numSP))
    displayAllSP(superPntList)

if __name__=='__main__':
    print('Start')
    runExample()