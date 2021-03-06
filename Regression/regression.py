#regression
from numpy import *

def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

def standRegres(xArr, yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T * xMat
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws

def lwlr(testPoint, xArr, yArr, k = 1.0):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))
    for j in range(m):
        diffMat = testPoint - xMat[j, :]
        weights[j, j] = exp(diffMat * diffMat.T / (-2.0 * k ** 2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * weights * yMat)
    return testPoint * ws

def lwlrTest(testArr, xArr, yArr, k = 1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat

def rssError(yArr, yHatArr):
    return ((yArr - yHatArr) ** 2).sum()

def ridgeRegres(xMat, yMat, lam = 0.2):
    xTx = xMat.T * xMat
    denom = xTx + eye(shape(xMat)[1]) * lam
    print(denom)
    if linalg.det(denom) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = denom.I * (xMat.T * yMat)
    return ws

def ridgeTest(xArr, yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yMean = mean(yMat, 0)
    yMat = yMat - yMean
    xMeans = mean(xMat, 0)
    xVar = var(xMat, 0)
    xMat = (xMat - xMeans) / xVar
    numTestPts = 30
    wMat = zeros((numTestPts, shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, exp(i - 10))
        wMat[i, :] = ws.T
    return wMat

def stageWise(xArr, yArr, eps = 0.01, numIt = 100):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yMean = mean(yMat, 0)
    yMat = yMat - yMean
    xMat = regularize(xMat)
    m, n = shape(xMat)
    returnMat = zeros((numIt, n))
    ws = zeros((n, 1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):
        print(ws.T)
        lowestError = inf
        for j in range(n):
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                rssE = rssError(yMat.A, yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i, :] = ws.T
    return returnMat

def regularize(xMat):
    inMat = xMat.copy()
    inMeans = mean(inMat,0)
    inVar = var(inMat,0)  
    inMat = (inMat - inMeans)/inVar
    return inMat

# need to apply google search API
# from time import sleep
# import json
# import urllib2
# def searchForSet(retX, retY, setNum, yr, numPce, origPrc):
#     sleep(10)
#     myAPIstr = 'get from code.google.com'
#     searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json' % (myAPIstr, setNum)
#     pg = urllib2.urlopen(searchURL)
#     print(pg)
#     retDict = json.loads(pg.read())
#     for i in range(len(retDict['items'])):
#         try:
#             currItem = retDict['items'][i]
#             if currItem['product']['condition'] == 'new':
#                 newFlag = 1.0
#             else:
#                 newFlag = 0
#             listOfInv = currItem['product']['inventories']
#             for item in listOfInv:
#                 sellingPrices = item['price']
#                 if sellingPrices > origPrc * 0.5:
#                     print("%d\t%d\t%d\t%f\t%f" % (yr, numPce, newFlag, origPrc, sellingPrices))
#                     retX.append([yr, numPce, newFlag, origPrc])
#                     retY.append(sellingPrices)
#         except:
#             print("problem with item %d" % i)

# def setDataCollect(retX, retY):
#     searchForSet(retX, retY, 8288, 2006, 800, 49.99)
#     searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
#     searchForSet(retX, retY, 10197, 2007, 5195, 499.99)
#     searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
#     searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
#     searchForSet(retX, retY, 10196, 2006, 800, 249.99)



if __name__ == '__main__':
    # lgX = []
    # lgY = []
    # setDataCollect(lgX, lgY)
    # print(lgX)
    # print(lgY)
    # xArr, yArr = loadDataSet('ex0.txt')
    # ws = standRegres(xArr, yArr)
    # xMat = mat(xArr)
    # yMat = mat(yArr)
    # yHat = xMat * ws
    # print(yArr[0])
    # print(lwlr(xArr[0], xArr, yArr, 1.0))
    # print(lwlr(xArr[0], xArr, yArr, 0.001))
    # yHat = lwlrTest(xArr, xArr, yArr, 0.003)
    # print(yHat)
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0], s = 2, c = 'g')
    # xCopy = xMat.copy()
    # xCopy.sort(0)
    # yHat = xMat * ws
    # srtInd = xMat[:, 1].argsort(0)
    # xSort = xMat[srtInd][:, 0, :]
    # ax.plot(xSort[:, 1], yHat[srtInd])
    # ax.plot(xCopy[:, 1], yHat)
    # plt.show()
    # print(corrcoef(yHat.T, yMat))
    # abX, abY = loadDataSet('abalone.txt')
    # yHat01 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 0.1)
    # yHat1 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1)
    # yHat10 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)
    # print(rssError(abY[0:99], yHat01.T))
    # print(rssError(abY[0:99], yHat1.T))
    # print(rssError(abY[0:99], yHat10.T))
    # yHat01 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 0.1)
    # print(rssError(abY[100:199], yHat01.T))
    # yHat1 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 1)
    # print(rssError(abY[100:199], yHat1.T))
    # yHat10 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 10)
    # print(rssError(abY[100:199], yHat10.T))
    # ws = standRegres(abX[0:99], abY[0:99])
    # yHat = mat(abX[100:199]) * ws
    # print(rssError(abY[100:199], yHat.T.A))
    # ridgeWeights = ridgeTest(abX, abY)
    # print(ridgeWeights)
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(ridgeWeights)
    # plt.show()
    # xArr, yArr = loadDataSet('abalone.txt')
    # print(stageWise(xArr, yArr, 0.01, 200))
    # print(stageWise(xArr, yArr, 0.001, 5000))














