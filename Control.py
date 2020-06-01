import random
import numpy
import jnius_config
jnius_config.set_classpath('C:\\Users\\conta\\eclipse-workspace/BehaviourGraphs.jar')
from jnius import autoclass

def main():
    D = getActionSequences()  # Data set to cluster
    following = getFollowing()
    A = {"Start", "Pre-Instruction", "Concurrent Instruction (Positive)", "Concurrent Instruciton (Negative)",
         "Post-Instruction (Positive)", "Post-Instruction (Negative)", "Manual Manipulation", "Questioning",
         "Positive Modelling", "Negative Modelling", "First Name", "Hustle", "Praise", "Scold", "Console",
         "End"}  # Action space
    sizeA = len(A)  # Size of the action space
    k = 8  # Number of clusters
    clusters = cluster(k, D, sizeA, following)  # The set of clusters


def getActionSequences():
    # Do something with Java here
    print("getting action sequences")
    PythonHelper = autoclass('behaviour_graphs.PythonHelper')
    ph = PythonHelper()
    '''testSequences = ph.getSequences()
    print(testSequences)
    print(testSequences[0][0])'''
    return ph.getSequences()


def getFollowing():
    # Do something with Java here
    print("getting following behaviours")
    PythonHelper = autoclass('behaviour_graphs.PythonHelper')
    ph = PythonHelper()
    testFollowing = ph.getFollowing()
    print(testFollowing)
    print(testFollowing[0])
    print(testFollowing[0][0])
    print(testFollowing[0][0][1])
    return ph.getFollowing()


def cluster(k, D, sizeA, following):
    rand = random.seed()
    returnClusters = []

    # print("Test (4,40): ", constrained_sum_sample_pos(4,40))
    # print("Test (7, 100): ", constrained_sum_sample_pos(7,100))


    # Initialise the set of optimal clusters by randomising each transition matrix
    rangeGap = 0.0001
    sum_sample = constrainedSumSampleNonneg(16, 1.0, rangeGap)
    print("Test (16, 1.0): ", sum_sample)
    sumOfTest = 0.0
    for c in sum_sample:
        sumOfTest = sumOfTest + c
    print("Sum of test: ", sumOfTest)

    '''PythonHelper = autoclass('behaviour_graphs.PythonHelper')
    ph = PythonHelper()
    javaMatrix = ph.getSquashMatrix()
    matrix = [
        [javaMatrix[0][0], javaMatrix[0][1], javaMatrix[0][2], javaMatrix[0][3], javaMatrix[0][4], javaMatrix[0][5], javaMatrix[0][6], javaMatrix[0][7], javaMatrix[0][8], javaMatrix[0][9], javaMatrix[0][10], javaMatrix[0][11], javaMatrix[0][12], javaMatrix[0][13], javaMatrix[0][14], javaMatrix[0][15]],
        [javaMatrix[1][0], javaMatrix[1][1], javaMatrix[1][2], javaMatrix[1][3], javaMatrix[1][4], javaMatrix[1][5], javaMatrix[1][6], javaMatrix[1][7], javaMatrix[1][8], javaMatrix[1][9], javaMatrix[1][10], javaMatrix[1][11], javaMatrix[1][12], javaMatrix[1][13], javaMatrix[1][14], javaMatrix[1][15]],
        [javaMatrix[2][0], javaMatrix[2][1], javaMatrix[2][2], javaMatrix[2][3], javaMatrix[2][4], javaMatrix[2][5], javaMatrix[2][6],
         javaMatrix[2][7], javaMatrix[2][8], javaMatrix[2][9], javaMatrix[2][10], javaMatrix[2][11], javaMatrix[2][12],
         javaMatrix[2][13], javaMatrix[2][14], javaMatrix[2][15]],
        [javaMatrix[3][0], javaMatrix[3][1], javaMatrix[3][2], javaMatrix[3][3], javaMatrix[3][4], javaMatrix[3][5], javaMatrix[3][6],
         javaMatrix[3][7], javaMatrix[3][8], javaMatrix[3][9], javaMatrix[3][10], javaMatrix[3][11], javaMatrix[3][12],
         javaMatrix[3][13], javaMatrix[3][14], javaMatrix[3][15]],
        [javaMatrix[4][0], javaMatrix[4][1], javaMatrix[4][2], javaMatrix[4][3], javaMatrix[4][4], javaMatrix[4][5], javaMatrix[4][6],
         javaMatrix[4][7], javaMatrix[4][8], javaMatrix[4][9], javaMatrix[4][10], javaMatrix[4][11], javaMatrix[4][12],
         javaMatrix[4][13], javaMatrix[4][14], javaMatrix[4][15]],
        [javaMatrix[5][0], javaMatrix[5][1], javaMatrix[5][2], javaMatrix[5][3], javaMatrix[5][4], javaMatrix[5][5], javaMatrix[5][6],
         javaMatrix[5][7], javaMatrix[5][8], javaMatrix[5][9], javaMatrix[5][10], javaMatrix[5][11], javaMatrix[5][12],
         javaMatrix[5][13], javaMatrix[5][14], javaMatrix[5][15]],
        [javaMatrix[6][0], javaMatrix[6][1], javaMatrix[6][2], javaMatrix[6][3], javaMatrix[6][4], javaMatrix[6][5], javaMatrix[6][6],
         javaMatrix[6][7], javaMatrix[6][8], javaMatrix[6][9], javaMatrix[6][10], javaMatrix[6][11], javaMatrix[6][12],
         javaMatrix[6][13], javaMatrix[6][14], javaMatrix[6][15]],
        [javaMatrix[7][0], javaMatrix[7][1], javaMatrix[7][2], javaMatrix[7][3], javaMatrix[7][4], javaMatrix[7][5], javaMatrix[7][6],
         javaMatrix[7][7], javaMatrix[7][8], javaMatrix[7][9], javaMatrix[7][10], javaMatrix[7][11], javaMatrix[7][12],
         javaMatrix[7][13], javaMatrix[7][14], javaMatrix[7][15]],
        [javaMatrix[8][0], javaMatrix[8][1], javaMatrix[8][2], javaMatrix[8][3], javaMatrix[8][4], javaMatrix[8][5], javaMatrix[8][6],
         javaMatrix[8][7], javaMatrix[8][8], javaMatrix[8][9], javaMatrix[8][10], javaMatrix[8][11], javaMatrix[8][12],
         javaMatrix[8][13], javaMatrix[8][14], javaMatrix[8][15]],
        [javaMatrix[9][0], javaMatrix[9][1], javaMatrix[9][2], javaMatrix[9][3], javaMatrix[9][4], javaMatrix[9][5], javaMatrix[9][6],
         javaMatrix[9][7], javaMatrix[9][8], javaMatrix[9][9], javaMatrix[9][10], javaMatrix[9][11], javaMatrix[9][12],
         javaMatrix[9][13], javaMatrix[9][14], javaMatrix[9][15]],
        [javaMatrix[10][0], javaMatrix[10][1], javaMatrix[10][2], javaMatrix[10][3], javaMatrix[10][4], javaMatrix[10][5], javaMatrix[10][6],
         javaMatrix[10][7], javaMatrix[10][8], javaMatrix[10][9], javaMatrix[10][10], javaMatrix[10][11], javaMatrix[10][12],
         javaMatrix[10][13], javaMatrix[10][14], javaMatrix[10][15]],
        [javaMatrix[11][0], javaMatrix[11][1], javaMatrix[11][2], javaMatrix[11][3], javaMatrix[11][4], javaMatrix[11][5], javaMatrix[11][6],
         javaMatrix[11][7], javaMatrix[11][8], javaMatrix[11][9], javaMatrix[11][10], javaMatrix[11][11], javaMatrix[11][12],
         javaMatrix[11][13], javaMatrix[11][14], javaMatrix[11][15]],
        [javaMatrix[12][0], javaMatrix[12][1], javaMatrix[12][2], javaMatrix[12][3], javaMatrix[12][4], javaMatrix[12][5], javaMatrix[12][6],
         javaMatrix[12][7], javaMatrix[12][8], javaMatrix[12][9], javaMatrix[12][10], javaMatrix[12][11], javaMatrix[12][12],
         javaMatrix[12][13], javaMatrix[12][14], javaMatrix[12][15]],
        [javaMatrix[13][0], javaMatrix[13][1], javaMatrix[13][2], javaMatrix[13][3], javaMatrix[13][4], javaMatrix[13][5], javaMatrix[13][6],
         javaMatrix[13][7], javaMatrix[13][8], javaMatrix[13][9], javaMatrix[13][10], javaMatrix[13][11], javaMatrix[13][12],
         javaMatrix[13][13], javaMatrix[13][14], javaMatrix[13][15]],
        [javaMatrix[14][0], javaMatrix[14][1], javaMatrix[14][2], javaMatrix[14][3], javaMatrix[14][4], javaMatrix[14][5], javaMatrix[14][6],
         javaMatrix[14][7], javaMatrix[14][8], javaMatrix[14][9], javaMatrix[14][10], javaMatrix[14][11], javaMatrix[14][12],
         javaMatrix[14][13], javaMatrix[14][14], javaMatrix[14][15]],
        [javaMatrix[15][0], javaMatrix[15][1], javaMatrix[15][2], javaMatrix[15][3], javaMatrix[15][4], javaMatrix[15][5], javaMatrix[15][6],
         javaMatrix[15][7], javaMatrix[15][8], javaMatrix[15][9], javaMatrix[15][10], javaMatrix[15][11], javaMatrix[15][12],
         javaMatrix[15][13], javaMatrix[15][14], javaMatrix[15][15]]
    ]
    returnClusters.append(matrix)

    javaMatrix = ph.getPhysioMatrix()
    matrix = [
        [javaMatrix[0][0], javaMatrix[0][1], javaMatrix[0][2], javaMatrix[0][3], javaMatrix[0][4], javaMatrix[0][5],
         javaMatrix[0][6], javaMatrix[0][7], javaMatrix[0][8], javaMatrix[0][9], javaMatrix[0][10], javaMatrix[0][11],
         javaMatrix[0][12], javaMatrix[0][13], javaMatrix[0][14], javaMatrix[0][15]],
        [javaMatrix[1][0], javaMatrix[1][1], javaMatrix[1][2], javaMatrix[1][3], javaMatrix[1][4], javaMatrix[1][5],
         javaMatrix[1][6], javaMatrix[1][7], javaMatrix[1][8], javaMatrix[1][9], javaMatrix[1][10], javaMatrix[1][11],
         javaMatrix[1][12], javaMatrix[1][13], javaMatrix[1][14], javaMatrix[1][15]],
        [javaMatrix[2][0], javaMatrix[2][1], javaMatrix[2][2], javaMatrix[2][3], javaMatrix[2][4], javaMatrix[2][5],
         javaMatrix[2][6],
         javaMatrix[2][7], javaMatrix[2][8], javaMatrix[2][9], javaMatrix[2][10], javaMatrix[2][11], javaMatrix[2][12],
         javaMatrix[2][13], javaMatrix[2][14], javaMatrix[2][15]],
        [javaMatrix[3][0], javaMatrix[3][1], javaMatrix[3][2], javaMatrix[3][3], javaMatrix[3][4], javaMatrix[3][5],
         javaMatrix[3][6],
         javaMatrix[3][7], javaMatrix[3][8], javaMatrix[3][9], javaMatrix[3][10], javaMatrix[3][11], javaMatrix[3][12],
         javaMatrix[3][13], javaMatrix[3][14], javaMatrix[3][15]],
        [javaMatrix[4][0], javaMatrix[4][1], javaMatrix[4][2], javaMatrix[4][3], javaMatrix[4][4], javaMatrix[4][5],
         javaMatrix[4][6],
         javaMatrix[4][7], javaMatrix[4][8], javaMatrix[4][9], javaMatrix[4][10], javaMatrix[4][11], javaMatrix[4][12],
         javaMatrix[4][13], javaMatrix[4][14], javaMatrix[4][15]],
        [javaMatrix[5][0], javaMatrix[5][1], javaMatrix[5][2], javaMatrix[5][3], javaMatrix[5][4], javaMatrix[5][5],
         javaMatrix[5][6],
         javaMatrix[5][7], javaMatrix[5][8], javaMatrix[5][9], javaMatrix[5][10], javaMatrix[5][11], javaMatrix[5][12],
         javaMatrix[5][13], javaMatrix[5][14], javaMatrix[5][15]],
        [javaMatrix[6][0], javaMatrix[6][1], javaMatrix[6][2], javaMatrix[6][3], javaMatrix[6][4], javaMatrix[6][5],
         javaMatrix[6][6],
         javaMatrix[6][7], javaMatrix[6][8], javaMatrix[6][9], javaMatrix[6][10], javaMatrix[6][11], javaMatrix[6][12],
         javaMatrix[6][13], javaMatrix[6][14], javaMatrix[6][15]],
        [javaMatrix[7][0], javaMatrix[7][1], javaMatrix[7][2], javaMatrix[7][3], javaMatrix[7][4], javaMatrix[7][5],
         javaMatrix[7][6],
         javaMatrix[7][7], javaMatrix[7][8], javaMatrix[7][9], javaMatrix[7][10], javaMatrix[7][11], javaMatrix[7][12],
         javaMatrix[7][13], javaMatrix[7][14], javaMatrix[7][15]],
        [javaMatrix[8][0], javaMatrix[8][1], javaMatrix[8][2], javaMatrix[8][3], javaMatrix[8][4], javaMatrix[8][5],
         javaMatrix[8][6],
         javaMatrix[8][7], javaMatrix[8][8], javaMatrix[8][9], javaMatrix[8][10], javaMatrix[8][11], javaMatrix[8][12],
         javaMatrix[8][13], javaMatrix[8][14], javaMatrix[8][15]],
        [javaMatrix[9][0], javaMatrix[9][1], javaMatrix[9][2], javaMatrix[9][3], javaMatrix[9][4], javaMatrix[9][5],
         javaMatrix[9][6],
         javaMatrix[9][7], javaMatrix[9][8], javaMatrix[9][9], javaMatrix[9][10], javaMatrix[9][11], javaMatrix[9][12],
         javaMatrix[9][13], javaMatrix[9][14], javaMatrix[9][15]],
        [javaMatrix[10][0], javaMatrix[10][1], javaMatrix[10][2], javaMatrix[10][3], javaMatrix[10][4],
         javaMatrix[10][5], javaMatrix[10][6],
         javaMatrix[10][7], javaMatrix[10][8], javaMatrix[10][9], javaMatrix[10][10], javaMatrix[10][11],
         javaMatrix[10][12],
         javaMatrix[10][13], javaMatrix[10][14], javaMatrix[10][15]],
        [javaMatrix[11][0], javaMatrix[11][1], javaMatrix[11][2], javaMatrix[11][3], javaMatrix[11][4],
         javaMatrix[11][5], javaMatrix[11][6],
         javaMatrix[11][7], javaMatrix[11][8], javaMatrix[11][9], javaMatrix[11][10], javaMatrix[11][11],
         javaMatrix[11][12],
         javaMatrix[11][13], javaMatrix[11][14], javaMatrix[11][15]],
        [javaMatrix[12][0], javaMatrix[12][1], javaMatrix[12][2], javaMatrix[12][3], javaMatrix[12][4],
         javaMatrix[12][5], javaMatrix[12][6],
         javaMatrix[12][7], javaMatrix[12][8], javaMatrix[12][9], javaMatrix[12][10], javaMatrix[12][11],
         javaMatrix[12][12],
         javaMatrix[12][13], javaMatrix[12][14], javaMatrix[12][15]],
        [javaMatrix[13][0], javaMatrix[13][1], javaMatrix[13][2], javaMatrix[13][3], javaMatrix[13][4],
         javaMatrix[13][5], javaMatrix[13][6],
         javaMatrix[13][7], javaMatrix[13][8], javaMatrix[13][9], javaMatrix[13][10], javaMatrix[13][11],
         javaMatrix[13][12],
         javaMatrix[13][13], javaMatrix[13][14], javaMatrix[13][15]],
        [javaMatrix[14][0], javaMatrix[14][1], javaMatrix[14][2], javaMatrix[14][3], javaMatrix[14][4],
         javaMatrix[14][5], javaMatrix[14][6],
         javaMatrix[14][7], javaMatrix[14][8], javaMatrix[14][9], javaMatrix[14][10], javaMatrix[14][11],
         javaMatrix[14][12],
         javaMatrix[14][13], javaMatrix[14][14], javaMatrix[14][15]],
        [javaMatrix[15][0], javaMatrix[15][1], javaMatrix[15][2], javaMatrix[15][3], javaMatrix[15][4],
         javaMatrix[15][5], javaMatrix[15][6],
         javaMatrix[15][7], javaMatrix[15][8], javaMatrix[15][9], javaMatrix[15][10], javaMatrix[15][11],
         javaMatrix[15][12],
         javaMatrix[15][13], javaMatrix[15][14], javaMatrix[15][15]]
    ]
    returnClusters.append(matrix)'''

    for i in range(0, k):
        matrix = [
            constrainedSumSampleNonneg(16, 1.0, rangeGap),
            constrainedSumSampleNonneg(16, 1.0, rangeGap),
            constrainedSumSampleNonneg(16, 1.0, rangeGap),
            constrainedSumSampleNonneg(16, 1.0, rangeGap),
            constrainedSumSampleNonneg(16, 1.0, rangeGap),
            constrainedSumSampleNonneg(16, 1.0, rangeGap),
            constrainedSumSampleNonneg(16, 1.0, rangeGap),
            constrainedSumSampleNonneg(16, 1.0, rangeGap),
            constrainedSumSampleNonneg(16, 1.0, rangeGap),
            constrainedSumSampleNonneg(16, 1.0, rangeGap),
            constrainedSumSampleNonneg(16, 1.0, rangeGap),
            constrainedSumSampleNonneg(16, 1.0, rangeGap),
            constrainedSumSampleNonneg(16, 1.0, rangeGap),
            constrainedSumSampleNonneg(16, 1.0, rangeGap),
            constrainedSumSampleNonneg(16, 1.0, rangeGap),
            constrainedSumSampleNonneg(16, 1.0, rangeGap)]

        returnClusters.append(matrix)

    '''print(returnClusters)
    print(len(returnClusters))
    print(len(returnClusters[0]))
    print(len(returnClusters[0][0]))'''
    for i in range(0,k):
        print("cluster", i)
        print("[", end = '')
        for j in range(0,16):
            print("[", returnClusters[i][j], "]")
        print("]\n\n\n\n")

    # Initialise sequence assignments Z = z1 ... zn
    Z = []
    for i in range(0, len(D)):
        Z.append(1)

    change = 1
    count = 1
    # Repeat until Z converges to stable assignments
    while change == 1:
        change = 0
        # E step - compute assignments for each sequence zi
        print("E step ", count)
        for i in range(0, len(Z)):
            initial = Z[i]
            l = len(D[i])
            products = []

            for zi in range(0, k):
                product = 1
                for j in range(2, l):
                    '''print("i: ", i)
                    print("zi: ", zi)
                    print("j: ", j)
                    print("D[i][j]: ", D[i][j])
                    print("D[i][j-1]: ", D[i][j-1])
                    print("Z[zi]: ", Z[zi])
                    print("returnClusters[Z[zi]]: ", returnClusters[Z[zi]])
                    print("size: ", len(returnClusters[Z[zi]]))
                    print("returnClusters[Z[zi]][D[i][j]]: ", returnClusters[Z[zi]][D[i][j]])
                    print("returnClusters")'''
                    product = product * returnClusters[zi][D[i][j]][D[i][j - 1]]
                products.append(product)
            #print(products)
            Z[i] = numpy.argmax(products)
            #print(Z[i])
            new = Z[i]
            if initial != new:
                change = 1

        print("Z: ", Z)

        # Make a list of all sequences by cluster
        sequences = []
        for z in range(0, k):
            sequences.append([])
        for z in range(0, k):
            for i in range(0, len(Z)):
                if Z[i] == z:
                    sequences[z].append(i)

        # M step - Update each transition matrix
        print("M step ", count)
        for z in range(0, k):
            for i in range(0, sizeA):
                for j in range(0, sizeA):
                    initial = returnClusters([z][i][j])
                    divisor = sumCountTransitions(sequences[z], sizeA, j, following)
                    if divisor == 0 and len(sequences[z]) > 2 and returnClusters[z][i][j] != 0.0:
                        #print("Setting returnClusters[", z, "][", i, "][", j, "] to 0.")
                        returnClusters[z][i][j] = 0.0
                        '''returnClusters[z][i][j] - 0.01
                        if returnClusters[z][i][j] < 0.0:
                            returnClusters[z][i][j] = 0.0'''
                    elif divisor != 0:
                        returnClusters[z][i][j] = countTransitions(sequences[z], i, j, following) / divisor
                    '''sum = 0
                    for x in range(1, sizeA):
                        sum = sum + following[x][j]
                    returnClusters[z][i][j] = following[i][j] / sum'''

        count = count + 1

    for i in range(0,k):
        print("cluster", i)
        print("[", end = '')
        for j in range(0,16):
            print("[", returnClusters[i][j], "]")
        print("]\n\n\n\n")


def constrainedSumSamplePos(n, total, rangeGap):
    """Return a randomly chosen list of n positive integers summing to total.
    Each such list is equally likely to occur."""
    numpyRange = numpy.arange(0.0, total, rangeGap)
    range = numpy.ndarray.tolist(numpyRange)
    dividers = sorted(random.sample(range, n - 1))
    return [a - b for a, b in zip(dividers + [total], [0.0] + dividers)]

def constrainedSumSampleNonneg(n, total, rangeGap):
    """Return a randomly chosen list of n nonnegative integers summing to total.
    Each such list is equally likely to occur."""

    return [x - rangeGap for x in constrainedSumSamplePos(n, total + (n * rangeGap), rangeGap)]

def countTransitions(sequences, i, j, following):
    # sequences is a list of all sequences assigned to the current cluster.
    sum = 0
    for sequenceNo in sequences:
        '''print("sequenceNo: ", sequenceNo)
        print("following[sequenceNo][i][j]: ", following[sequenceNo][i][j])'''
        sum = sum + following[sequenceNo][i][j]
    # print("Returning ", sum)
    return sum


def sumCountTransitions(sequences, size, j, following):
    sum = 0
    for x in range(0, size):
        sum = sum + countTransitions(sequences, x, j, following)
    return sum


if __name__ == '__main__':
    main()
