#svdRec.py
#divide m * n into m * m, m * n, n * n
from numpy import *
from numpy import linalg as la

def loadExData():
    return [[1, 1, 1, 0, 0],
            [2, 2, 2, 0, 0],
            [1, 1, 1, 0, 0],
            [5, 5, 5, 0, 0],
            [1, 1, 0, 2, 2],
            [0, 0, 0, 3, 3],
            [0, 0, 0, 1, 1]]

def loadExData3():
    return [[4, 4, 0, 2, 2],
            [4, 0, 0, 3, 3],
            [4, 0, 0, 1, 1],
            [1, 1, 1, 2, 0],
            [2, 2, 2, 0, 0],
            [1, 1, 1, 0, 0],
            [5, 5, 5, 0, 0]]

def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]

def ecluidSim(inA, inB):
    return 1.0 / (1.0 + la.norm(inA - inB))

def pearsSim(inA, inB):
    if len(inA) < 3:
        return 1.0
    return 0.5 + 0.5 * corrcoef(inA, inB, rowvar = 0)[0][1]

def cosSim(inA, inB):
    num = float(inA.T * inB)
    denom = la.norm(inA) * la.norm(inB)
    return 0.5 + 0.5 * (num / denom)

def standEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0:
            continue
        overlap = nonzero(logical_and(dataMat[:, item].A > 0, dataMat[:, j].A > 0))[0]
        if len(overlap) == 0:
            similarity = 0
        else:
            similarity = simMeas(dataMat[overlap, item], dataMat[overlap, j])
        print('the %d and %d similarity is: %f' % (item, j, similarity))            
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal

def recommend(dataMat, user, N = 3, simMeas = cosSim, estMethod = standEst):
    unratedItems = nonzero(dataMat[user, :].A == 0)[1]
    if len(unratedItems) == 0:
        return 'you rated everything'
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))
    return sorted(itemScores, key = lambda jj: jj[1], reverse = True)[:N]

def svdEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    U, Sigma, VT = la.svd(dataMat)
    #top 3 feature include 90%+ power for exdata2
    Sig4 = mat(eye(4) * Sigma[: 4])
    xformedItems = dataMat.T * U[:, :4] * Sig4.I
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0 or j == item:
            continue
        similarity = simMeas(xformedItems[item, :].T, xformedItems[j, :].T)
        print('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal


if __name__ == '__main__':
    # data = loadExData()
    # U, Sigma, VT = linalg.svd(data)
    # print(U)
    # print(Sigma)
    # print(VT)
    # Sig3 = mat([[Sigma[0], 0, 0], [0, Sigma[1], 0], [0, 0, Sigma[2]]])
    # print(U[:, :3] * Sig3 * VT[:3, :])
    # myMat = mat(loadExData3())
    # print(ecluidSim(myMat[:, 0], myMat[:, 4]))
    # print(ecluidSim(myMat[:, 0], myMat[:, 0]))
    # print(pearsSim(myMat[:, 0], myMat[:, 4]))
    # print(pearsSim(myMat[:, 0], myMat[:, 0]))
    # print(cosSim(myMat[:, 0], myMat[:, 4]))
    # print(cosSim(myMat[:, 0], myMat[:, 0]))
    # myMat[0, 1] = myMat[0, 0] = myMat[1, 0] = myMat[2, 0] = 4
    # myMat[3, 3] = 2
    # print(myMat)
    # print(recommend(myMat, 2))
    # print(recommend(myMat, 2, simMeas = ecluidSim))
    # print(recommend(myMat, 2, simMeas = pearsSim))
    myMat = mat(loadExData2())
    print(recommend(myMat, 1, estMethod = svdEst))
    # print(recommend(myMat, 1, estMethod = svdEst, simMeas = pearsSim))








