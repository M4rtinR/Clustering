import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import jnius_config
import copy
jnius_config.set_classpath('C:\\Users\\conta\\eclipse-workspace/BehaviourGraphsRunnable.jar')
from jnius import autoclass,PythonJavaClass,cast,java_method

Integer = autoclass('java.lang.Integer')

def main():
    '''
    data values:
    0 = TEST
    1 = SQUASH COACHES
    2 = STROKE PHYSIOTHERAPISTS
    3 = ALL

    compound values:
    0 = ONLY PARENT BEHAVIOURS
    1 = INCLUDE CONCURRENT BEHAVIOURS
    '''
    data = 3
    compound = 1
    '''centroids = [[0.2, 0.3], [0.4, 0.8]]
    clusters = [[[0.21, 0.25], [0.19, 0.35]], [[0.58, 0.67], [0.22, 0.93]]]
    print(inertia(centroids, clusters))'''

    PythonHelper = autoclass('behaviour_graphs.PythonHelper')
    ph = PythonHelper(data)

    DFromJava = ph.getSequences(data, compound)  # Observed behaviour seqeunces from the Java program which reads in the data
    # following = createTransitionMatrices(D)

    sizeA = ph.uniqueElements(DFromJava) + 2  # Size of the action space
    D = [[behaviour for behaviour in sequence] for sequence in DFromJava]  # Convert behaviour sequences to a more python-friendly form

    # following = getFollowing()
    '''A = {"Start", "Pre-Instruction", "Concurrent Instruction (Positive)", "Concurrent Instruciton (Negative)",
         "Post-Instruction (Positive)", "Post-Instruction (Negative)", "Manual Manipulation", "Questioning",
         "Positive Modelling", "Negative Modelling", "First Name", "Hustle", "Praise", "Scold", "Console",
         "End"}  # Action space'''
    #sizeA = len(A)  # Size of the action space


    # Test data:
    '''D2 = [[random.randint(1, 4) for j in range(0, random.randint(1, 20))] for i in range(0, 10)]
    print(D1)
    print(type(D1))
    print(D2)
    print(type(D2))'''

    '''D = [[1, 1, 1, 1, 2, 1], [1, 1, 2, 1, 1, 1], [3, 4, 3, 3, 3, 3], [3, 3, 3, 3, 3, 4]]  #, [1,1,1,2,1,1,1,2,1], [2,1,1,1,2,1,1,1,1,1], [1,1,1,2,1], [1,1,1,1,1,1,1,2,2,1,1,1], [1,1,1,1,2,2,1,1,1,2,1], [3,3,3,4,3,3,3,4,3], [3,3,3,4,3], [4,3,3,3,3,3,4,3,3], [3,3,3,3,4,4,3,3,3,4,3]]
    sizeA = 6
    following = createTransitionMatrices(D, sizeA)
    print(following)'''

    k, gapdf = optimalK(D, sizeA, nrefs=3, maxClusters=len(D))  # Number of clusters
    print('optimal k = ', k)
    plt.plot(gapdf.clusterCount, gapdf.gap, linewidth=3)
    plt.scatter(gapdf[gapdf.clusterCount == k].clusterCount, gapdf[gapdf.clusterCount == k].gap, s=250, c='r')
    plt.grid(True)
    plt.xlabel('Cluster Count')
    plt.ylabel('Gap Value')
    plt.title('Gap Values by Cluster Count')
    plt.show()
    '''dispdf = sumofsquarederrors(D, sizeA, maxClusters=15)
    plt.plot(dispdf.clusterCount, dispdf.errorsWk, linewidth=3, color='red')
    #plt.plot(dispdf.clusterCount, dispdf.errorsInertia, linewidth=3, color='blue')
    plt.plot(dispdf.clusterCount, dispdf.expectedLogMean, linewidth=3, color='green')
    plt.plot(dispdf.clusterCount, dispdf.expectedMeanLog, linewidth=3, color='yellow')
    plt.grid(True)
    plt.xlabel('Cluster Count')
    plt.ylabel('Sum of Squared Error')
    plt.title('Comparison of Sum of Squared Errors by Cluster Count')
    plt.show()
    # k = 2'''
    Z, clusterCentroids, following = cluster2(k, D, sizeA)  # The set of clusters
    print('Z:', Z)
    print('Centroids:', clusterCentroids)
    ph.createGraph(clusterCentroids)
    input("Press Enter to continue...")

def getActionSequences():
    # Get the observed behaviour sequences from the Java program which reads in the data.
    print("getting action sequences")
    PythonHelper = autoclass('behaviour_graphs.PythonHelper')
    ph = PythonHelper()
    '''testSequences = ph.getSequences()
    print(testSequences)
    print(testSequences[0][0])'''
    return ph.getSequences()


'''def getActionSpaceSize(D, ph):
    # Calculate the size of the action space based on the behaviour sequences given.
    PythonHelper = autoclass('behaviour_graphs.PythonHelper')
    # ph = PythonHelper()
    return PythonHelper(data).uniqueElements(D) + 2'''


def createTransitionMatrices(D, sizeA):
    # The dimensions of each transition matrix will be nxn where n = number of unique behaviours in data.

    # First count the number of transitions between each behaviour.
    countedMatrix = [[[0 for k in range(0, sizeA)] for j in range(0, sizeA)] for i in range(0, len(D))]
    totalOccs = [[0 for j in range(0, sizeA)] for i in range(0, len(D))]
    newD = [[] for i in range(0, len(D))]  # Add 0 to start and sizeA to end of each sequence

    for i in range(0, len(D)):
        previous = 0
        totalOccs[i][previous] += 1
        newD[i].append(0)
        for current in D[i]:
            countedMatrix[i][previous][current] += 1
            totalOccs[i][current] += 1
            newD[i].append(current)
            previous = current
        countedMatrix[i][previous][sizeA-1] += 1  # Deal with last behaviour
        totalOccs[i][sizeA-1] += 1
        newD[i].append(sizeA-1)

    # print("totalOccs: ", totalOccs)

    # Then calculate it as a percentage of total behaviours.
    transitionMatrices = [[[float(0) if totalOccs[i][row] == 0 else count/totalOccs[i][row] for count in countedMatrix[i][row]] for row in range(0, len(countedMatrix[i]))] for i in range(0, len(D))]
    # print(transitionMatrices)

    totals = [len(sequence) + 2 for sequence in D]  # +2 for start and end
    # print(totals)

    for matrix in range(0, len(transitionMatrices)):
        for row in range(0, len(transitionMatrices[matrix])):
            transitionMatrices[matrix][row][0] = totalOccs[matrix][row]/totals[matrix]
    # print(transitionMatrices)

    return newD, transitionMatrices


def sumofsquarederrors(data, sizeA, nrefs=3, maxClusters=15):
    resultsdf = pd.DataFrame({'clusterCount': [], 'errorsWk': [], 'errorsInertia': [], 'expectedLogMean': [], 'expectedMeanLog': []})

    for gap_index, k in enumerate(range(1, maxClusters)):
        refDisps = np.zeros(nrefs)

        for i in range(nrefs):
            randomReference = [np.random.randint(1, high=sizeA - 1,
                                                 size=np.random.randint(1, max([len(sequence) for sequence in data])))
                               for i in range(0, len(data))]
            Z, clusterCentroids, transitionMatrices = cluster2(k, randomReference, sizeA)
            clusters = createClusterMatrices(Z, transitionMatrices, k)

            refDisp = Wk(clusterCentroids, clusters)

            refDisps[i] = refDisp

        Z, clusterCentroids, transitionMatrices = cluster2(k, data, sizeA)
        clusters = createClusterMatrices(Z, transitionMatrices, k)

        dispWk = Wk(clusterCentroids, clusters)
        dispInertia = inertia(clusterCentroids, clusters)

        resultsdf = resultsdf.append({'clusterCount': k, 'errorsWk': np.log(dispWk), 'errorsInertia': np.log(dispInertia), 'expectedLogMean': np.log(np.mean(refDisps)), 'expectedMeanLog': np.mean([np.log(rd) for rd in refDisps])}, ignore_index=True)

    return resultsdf


def optimalK(data, sizeA, nrefs=3, maxClusters=15):
    """
    Calculates the optimal K value using Gap Statistic from Tibshirani, Walther, Hastie
    Params:
        data: ndarry of behaviour sequences
        sizeA: size of the action space
        nrefs: number of sample reference datasets to create
        maxClusters: Maximum number of clusters to test for
    Returns: (gaps, optimalK)
    """
    gaps = np.zeros((len(range(1, maxClusters)),))
    resultsdf = pd.DataFrame({'clusterCount': [], 'gap': []})

    for gap_index, k in enumerate(range(1, maxClusters)):
        print('k:', k)
        # Holder for reference dispersion results
        refDisps = np.zeros(nrefs)

        # For n references, generate random sample and perform EM clustering getting resulting dispersion of each loop
        for i in range(nrefs):
            # Create new random reference set by generating i random sequences of length m
            # where i = no. of sequences in data and m = random int between 1 and max length of any sequence in the data.
            randomReference = [np.random.randint(1, high=sizeA - 1,
                                                 size=np.random.randint(1, max([len(sequence) for sequence in data])))
                               for i in range(0, len(data))]
            # print('randomReference', i, ', length =', len(randomReference), ':', randomReference)

            # Fit to it using defined clustering algorithm
            Z, clusterCentroids, transitionMatrices = cluster2(k, randomReference, sizeA)
            clusters = createClusterMatrices(Z, transitionMatrices, k)
            # print('clusters for randomReference ', i, ': ', clusters)
            # print('centroids for randomReference', i, ': ', clusterCentroids)
            # print('elements in each cluster: ', [len(c) for c in clusters])

            refDisp = inertia(clusterCentroids, clusters)
            # print('refDisp for randomReference ', i, ': ', refDisp)
            refDisps[i] = refDisp

        # Fit cluster to original data and create dispersion
        Z, clusterCentroids, transitionMatrices = cluster2(k, data, sizeA)
        clusters = createClusterMatrices(Z, transitionMatrices, k)
        # print('clusters for data, k =', k, ': ', clusters)
        # print('centroids for data, k =', k, ': ', clusterCentroids)
        # print('elements in each cluster:', [len(c) for c in clusters])

        origDisp = inertia(clusterCentroids, clusters)
        # print('origDisp for data, k =', k, ': ', origDisp)

        # Calculate gap statistic
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)
        # gap = np.mean([np.log(rd) for rd in refDisps]) - np.log(origDisp)
        # print('gap, k =', k, ': ', gap)

        # Assign this loop's gap statistic to gaps
        gaps[gap_index] = gap

        resultsdf = resultsdf.append({'clusterCount': k, 'gap': gap}, ignore_index=True)

    return gaps.argmax() + 1, resultsdf  # Plus 1 because index of 0 means 1 cluster is optimal, index 2 = 3 clusters are optimal


def createClusterMatrices(Z, transitionMatrices, k):
    # Creates an array of size k containing lists of the transition matrices currently grouped into together.

    clusterMatrices = [[] for i in range(0, k)]
    # for each assignment in Z
    for i in range(0, len(Z)):
        # add corresponding transition matrix to appropriate element of return list
        clusterMatrices[Z[i]].append(transitionMatrices[i])

    return clusterMatrices


def inertia(centroids, clusters):
    # Calculate sum of squared distance for each cluster.

    return sum(  # for each cluster
        [sum(  # for each point in the cluster
            [math.pow(np.linalg.norm(np.array(clusters[i][j]) - np.array(centroids[i])), 2) for j in
             range(0, len(clusters[i]))]  # square the euclidean distance between point and centroid
        ) for i in range(0, len(centroids))]
    )


def Wk(centroids, clusters):
    # Calculate sum of squared distance for each cluster.

    return sum(  # for each cluster
        [(1 / 2*len(clusters[i])) * sum(  # for each point in the cluster
            [math.pow(np.linalg.norm(np.array(clusters[i][j]) - np.array(centroids[i])), 2) for j in
             range(0, len(clusters[i]))]  # square the euclidean distance between point and centroid
        ) for i in range(0, len(centroids))]
    )


def cluster2(k, origD, sizeA):
    # Generate transition matrices from the data
    D, transitionMatrices = createTransitionMatrices(origD, sizeA)

    # Initialise centroids by randomising centroid1 ... centroidk
    rangeGap = 0.0001
    '''randomReference = [
        np.random.randint(1, high=sizeA - 1, size=np.random.randint(1, max([len(sequence) for sequence in origD]))) for i
        in range(0, len(origD))]
    initialisationSequences = [
        np.random.randint(1, high=sizeA - 1, size=np.random.randint(1, max([len(sequence) for sequence in origD]))) for
        i in range(0, k)]
    initialisationD, centroids = createTransitionMatrices(initialisationSequences, sizeA)'''
    centroids = [[constrainedSumSamplePos(sizeA, 1.0, rangeGap) for j in range(0, sizeA)] for i in range(0, k)]

    # Initialise sequence assignments Z = z1 ... zn
    Z = [0 for i in range(0, len(D))]

    change = 1
    while change == 1:
        # E-step
        change = 0
        initial = Z
        Z = E_step(D, centroids, transitionMatrices)
        if initial != Z:
            change = 1

        # M-step
            centroids = M_step(Z, k, sizeA, transitionMatrices, centroids, 0)  # Set 0s to 0.005 so that product does not = 0
        else:
            centroids = M_step(Z, k, sizeA, transitionMatrices, centroids, 1)  # Set 0s to 0 because we don't need product again

    return Z, centroids, transitionMatrices


def E_step(D, centroids, transitionMatrices):

    newZ = [np.argmax([np.product([centroids[zi][D[i][j-1]][D[i][j]] for j in range(1, len(D[i]))]) for zi in range(0, len(centroids))]) for i in range(0, len(D))]

    '''newZ = [np.argmin(  # Assign minimum of
        [np.linalg.norm(np.array(transitionMatrices[i]) - np.array(centroids[j])) for j in range(0, len(centroids))])
            for i in range(0, len(transitionMatrices))]  # Distance between the current transition matrix and each centroid.'''

    return newZ


def M_step(Z, k, sizeA, transitionMatrices, oldCentroids, final):
    groupedMatrices = groupMatrices(Z, transitionMatrices, k)

    # Each centroid becomes the average transition matrix of each matrix assigned to that cluster.
    centroids = [oldCentroids[z] if len(groupedMatrices[z]) == 0 else averageMatrix(groupedMatrices[z], sizeA, final) for z in
                 range(0, k)]

    return centroids


def groupMatrices(Z, transitionMatrices, k):
    # Group the matrices into their repective assigned clusters.
    groupedMatrices = [[] for i in range(0, k)]

    for assignment in range(0, len(Z)):
        groupedMatrices[Z[assignment]].append(transitionMatrices[assignment])

    return groupedMatrices


def averageMatrix(matrixList, sizeA, final):
    # Calculate the average for each point in the given matrix list.
    averageMatrix = [[np.mean([matrix[row][col] for matrix in matrixList]) if  # Use mean unless mean==0 or it's the final M step
                      np.mean([matrix[row][col] for matrix in matrixList]) != 0 or final else 0.005
                      for col in range(0, sizeA)] for row in range(0, sizeA)]

    return averageMatrix


def constrainedSumSamplePos(n, total, rangeGap):
    """Return a randomly chosen list of n positive integers summing to total.
    Each such list is equally likely to occur."""
    numpyRange = np.arange(0.0, total, rangeGap)
    range = np.ndarray.tolist(numpyRange)
    dividers = sorted(random.sample(range, n - 1))
    return [a - b for a, b in zip(dividers + [total], [0.0] + dividers)]


def displayGraphs(clusters):
    PythonHelper = autoclass('behaviour_graphs.PythonHelper')
    ph = PythonHelper()
    ph.createGraph(clusters)


if __name__ == '__main__':
    main()




def countTransitions2(D, Z, z, i, j):
    total = 0
    for d in range(0, len(Z)):
        if Z[d] == z:
            total += sum(
                [1 if (D[d][current] == i and D[d][current + 1] == j) else 0 for current in range(0, len(D[d]) - 1)])

    return total


def getFollowing():
    # Do something with Java here
    print("getting following behaviours")
    PythonHelper = autoclass('behaviour_graphs.PythonHelper')
    ph = PythonHelper()
    '''testFollowing = ph.getFollowing()
    print(testFollowing)
    print(testFollowing[0])
    print(testFollowing[0][0])
    print(testFollowing[0][0][1])'''
    return ph.getFollowing()


def cluster(k, origD, sizeA):
    D, following = createTransitionMatrices(origD, sizeA)
    #sizeA = getActionSpaceSize(D)
    #sizeA = 6

    rand = random.seed()
    returnClusters = []

    # print("Test (4,40): ", constrained_sum_sample_pos(4,40))
    # print("Test (7, 100): ", constrained_sum_sample_pos(7,100))


    # Initialise the set of optimal clusters by randomising each transition matrix
    rangeGap = 0.0001
    sum_sample = constrainedSumSampleNonneg(sizeA, 1.0, rangeGap)
    # print("Test (16, 1.0): ", sum_sample)
    sumOfTest = 0.0
    for c in sum_sample:
        sumOfTest = sumOfTest + c
    # print("Sum of test: ", sumOfTest)

    for i in range(0, k):
        matrix = []
        for j in range(0, sizeA):
            matrix.append(constrainedSumSampleNonneg(sizeA, 1.0, rangeGap))

        returnClusters.append(matrix)

    '''print(returnClusters)
    print(len(returnClusters))
    print(len(returnClusters[0]))
    print(len(returnClusters[0][0]))
    for i in range(0,k):
        print("cluster", i)
        print("[", end = '')
        for j in range(0,16):
            print("[", returnClusters[i][j], "]")
        print("]\n\n\n\n")'''

    # Initialise sequence assignments Z = z1 ... zn
    Z = []
    for i in range(0, len(D)):
        Z.append(1)

    change = 1
    count = 1
    Echange = 0

    previousE = Z.copy()
    previousM = copy.deepcopy(returnClusters)

    # Repeat until Z converges to stable assignments
    while change == 1:
        change = 0
        echange = 0
        mchange = 0

        if Echange == 0:
            # E step - compute assignments for each sequence zi
            tempE = Z.copy()
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
                Z[i] = np.argmax(products)
                #print(Z[i])
                new = Z[i]
                previous = previousE[i]

                # Check for convergence
                if count == 1:
                    if initial != new:
                        # print("1st round E step changes happening")
                        echange = 1
                else:
                    if initial != new and new != previous:
                        # print("E step changes happening")
                        echange = 1

            previousE = tempE
            print("Z: ", Z)

        if echange == 0:
            Echange = 1

        # Make a list of all sequences by cluster
        sequences = []
        for z in range(0, k):
            sequences.append([])
        for z in range(0, k):
            for i in range(0, len(Z)):
                if Z[i] == z:
                    sequences[z].append(i)

        tempM = copy.deepcopy(returnClusters)
        # M step - Update each transition matrix
        print("M step ", count)
        for z in range(0, k):
            for i in range(0, sizeA):
                for j in range(0, sizeA):
                    initial = returnClusters[z][i][j]
                    # print("returnClusters[", z, "][", i, "][", j, "] = ", returnClusters[z][i][j], ", previousM[", z, "][", i, "][", j, "] = ", previousM[z][i][j])
                    divisor = sumCountTransitions(sequences[z], sizeA, j, following)
                    if divisor == 0 and len(sequences[z]) > 2 and returnClusters[z][i][j] != 0.0:
                        # print("Setting returnClusters[", z, "][", i, "][", j, "] to 0.")
                        returnClusters[z][i][j] = 0.0
                        '''returnClusters[z][i][j] - 0.01
                        if returnClusters[z][i][j] < 0.0:
                            returnClusters[z][i][j] = 0.0'''
                    elif divisor != 0:
                        # print("Change")
                        returnClusters[z][i][j] = countTransitions(sequences[z], i, j, following) / divisor
                    '''sum = 0
                    for x in range(1, sizeA):
                        sum = sum + following[x][j]
                    returnClusters[z][i][j] = following[i][j] / sum'''
                    # print("returnClusters[", z, "][", i, "][", j, "] = ", returnClusters[z][i][j], ", previousM[", z,
                          # "][", i, "][", j, "] = ", previousM[z][i][j])
                    final = returnClusters[z][i][j]
                    difference = initial - final
                    previous = previousM[z][i][j]

                    # Check for convergence
                    if count == 1:
                        if difference < -0.01 or difference > 0.01:
                            # print("1st round M step changes happening in returnClusters[", z, "][", i, "][", j, "]: changed from ", initial, " to ", final)
                            mchange = 1
                    else:
                        # print("in else, difference = ", difference, ", final = ", final, ", previous = ", previous)
                        if (difference < -0.01 or difference > 0.01) and final != previous:
                            # print("M step changes happening in returnClusters[", z, "][", i, "][", j, "]: changed from ", initial, " to ", final)
                            mchange = 1
        previousM = tempM

        if not((Echange == 1 and mchange == 0) or (mchange == 0 and count > 50)):
            change = 1

        count = count + 1

    # Display results
    '''print("Z: ", Z)
    for i in range(0,k):
        print("cluster", i)
        print("[", end = '')
        for j in range(0,sizeA):
            print("[", returnClusters[i][j], "]")
        print("]\n\n\n\n")'''

    return Z, returnClusters, following


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

