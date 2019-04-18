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
from scipy.spatial import KDTree

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
    tree = KDTree(pntCloud)
    while getCoveragePercent(pntCloudFull) < coverageLim:
        centerPnt = chooseRandNonCoveredPnt(pntCloudFull)
        pntCloudFull = labelPointsInRad(centerPnt, coverSphereRad, pntCloudFull, spLabel, tree)
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
    prevCenterPntList = []
    spLabel = 0
    checkLoop = 0
    tree = KDTree(pntCloud)
    while spLabel < superPointNum:
        centerPnt = chooseRandMinCoveredPnt(pntCloudFull,prevCenterPntList)
        prevCenterPntList.append(centerPnt) #to not select the same point twice
        pntCloudFull, hasChanged = labelNumPointsInRad(centerPnt, coverSphereRad, pntCloudFull, spLabel, pointsInSP, tree)
        if hasChanged:
            spLabel += 1
            checkLoop = 0
        checkLoop +=1
        if checkLoop > superPointNum*3:
            print('Error: Enter Valid Input Parameters')
            return pntCloudFull
    return pntCloudFull

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

def chooseRandMinCoveredPnt(pntCloud,prevCenterPntList):
    # Choose a random minimally-covered point (with an minimal sized label list)
    # Useful when partial coverage assumption isn't valid
    # INPUT: pntCloud [(x1,y1,z1,[l1,..lk]),(x2,y2,z2,[l1,..lk]),...(xn,yn,zn,[l1,..lk])]
    # OUTPUT: point (x,y,z,[])
    minCov = 0
    newCoverPoint = []
    #Choose a random point with a minimum coverage
    while not newCoverPoint:
        nonCoveredList = [pnt for pnt in pntCloud if len(pnt[3])==minCov and pnt[0:3] not in prevCenterPntList]
        if len(nonCoveredList)>0:
            newCoverPoint = random.choice(nonCoveredList)
        else:
            minCov +=1

    return newCoverPoint

def labelPointsInRad(centerPnt, coverSphereRad, pntCloud, labelNum, tree):
    # Label Points in radius as uniform Super Point
    # INPUT: centerPnt (x,y,z), coverSphereRad is scalar, pntCloud is a list of point tuples Nx4, labelNum - scalar (int) Superpoint label
    # OUTPUT: list of points within radius (n2<=N)  [(x1,y1,z1,[l1,..lk]),(x2,y2,z2,[l1,..lk]),...(xn2,yn2,zn2,[l1,..lk])]
    centerPnt = list(centerPnt[0:3])
    pointsInRad = tree.query_ball_point(centerPnt, coverSphereRad)
    for idx in pointsInRad:
        pnt = pntCloud[idx]
        pnt[3].append(labelNum)
    return pntCloud

def labelNumPointsInRad(centerPnt, coverSphereRad, pntCloud, labelNum, numLabel, tree):
    # Label Points in radius as uniform Super Point
    # INPUT: centerPnt (x,y,z), coverSphereRad is scalar, pntCloud is a list of point tuples Nx4, labelNum - scalar (int) Superpoint label
    # OUTPUT: list of points within radius (n2<=N)  [(x1,y1,z1,[l1,..lk]),(x2,y2,z2,[l1,..lk]),...(xn2,yn2,zn2,[l1,..lk])]
    centerPnt = list(centerPnt[0:3])
    pointsInRad = tree.query_ball_point(centerPnt, coverSphereRad)
    if len(pointsInRad) < numLabel:
        print('Error: Raise Radius - Not enough points ('+str(numPointsInSP)+') to meet numLabel requirement of '+str(numLabel))
        return pntCloud, False
    random.shuffle(pointsInRad)
    numPointsInSP = 0
    for idx in pointsInRad:
        if numPointsInSP >= numLabel:
            break
        pnt = pntCloud[idx]
        pnt[3].append(labelNum)
        numPointsInSP +=1
    hasChanged = True
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

def displayAllSP(superPntList,titleName):
    # Visualize Results
    fig = plt.figure()  # create figure
    ax = fig.add_subplot(111, projection='3d')  # initialize 3D axis
    m1 = max([max(pnt[3]) for pnt in superPntList if len(pnt[3]) > 0])+1
    for i in range(m1):  # for each SP label
        pntList = [pnt for pnt in superPntList if i in pnt[3]]
        displaySP(pntList, ax)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_title(titleName)

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
        pntCloud.append([pntX, pntY, pntZ])
    return pntCloud

def runExample():
    # Run example RSCS Superpoint creation
    # Defining output by coverage percentage
    pntCloud = createRandomCloud(1000)
    csRad = 30
    superPntList, numSP = createRandomSphereCoverSet(pntCloud, coverSphereRad=csRad)
    print('You set ' + str(95) + ' Point Coverage, while defining the radius as ' + str(csRad))
    print('-----')
    print('original points: '+str(len(pntCloud)))
    print('labeled points: ' + str(len([pnt for pnt in superPntList if len(pnt[3]) > 0])))
    print('super points found:' + str(max([max(pnt[3]) for pnt in superPntList if len(pnt[3]) > 0])+1))
    displayAllSP(superPntList, 'Original RSCS')
    print('-----')

def runExampleFixed():
    # Run example RSCS Superpoint creation
    # Using a fixed output definition number of SP and Points in each SP
    pntCloud = createRandomCloud(4000)
    numSP = 50
    spSize = 60
    csRad = 25
    superPntList  = createRandomSphereCoverSetFixedNum(pntCloud,superPointNum=numSP,pointsInSP=spSize,coverSphereRad=csRad)
    print('You set '+str(numSP)+' SuperPoints with '+str(spSize)+' in each SP, while defining the radius as '+str(csRad))
    print('-----')
    print('original points: '+str(len(pntCloud)))
    labeledPntNum = len([pnt for pnt in superPntList if len(pnt[3]) > 0])
    print('labeled points: ' + str(labeledPntNum))
    if labeledPntNum>0:
        print('super points found:' + str(max([max(pnt[3]) for pnt in superPntList if len(pnt[3]) > 0])+1))
        displayAllSP(superPntList, 'Fixed Size RSCS')
        print('~Successful fixed-RSCS application~')
    else:
        print('~Failed Attempt - Change the input parameters to recieve your requested results. This is code not magic.~')
    print('-----')

if __name__=='__main__':
    print('Start')
    runExample()
    runExampleFixed()
    plt.show()