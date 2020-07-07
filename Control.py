import random
import numpy as np
import pandas as pd
import jnius_config
import copy
jnius_config.set_classpath('C:\\Users\\conta\\eclipse-workspace/BehaviourGraphsRunnable.jar')
from jnius import autoclass

def main():
    D = getActionSequences()
    #following = createTransitionMatrices(D)

   #sizeA = getActionSpaceSize(D)

    #following = getFollowing()
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

    k, gapdf = optimalK(D, nrefs=3, maxClusters=10)  # Number of clusters
    print('optimal k = ', k)
    Z, clusterCentroids, following = cluster(k, D)  # The set of clusters
    displayGraphs(clusterCentroids)
    input("Press Enter to continue...")

def getActionSequences():
    # Do something with Java here
    print("getting action sequences")
    PythonHelper = autoclass('behaviour_graphs.PythonHelper')
    ph = PythonHelper()
    '''testSequences = ph.getSequences()
    print(testSequences)
    print(testSequences[0][0])'''
    return ph.getSquashSequences()


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


def getActionSpaceSize(D):
    PythonHelper = autoclass('behaviour_graphs.PythonHelper')
    ph = PythonHelper()
    return ph.uniqueElements(D) + 2


def cluster(k, D):
    following = createTransitionMatrices(D)
    #sizeA = getActionSpaceSize(D)
    sizeA = 6

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
    print(len(returnClusters[0][0]))''''''
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


def createTransitionMatrices(D):
    # The dimensions of each transition matrix will be nxn wher n = number of unique behaviours in data.
    PythonHelper = autoclass('behaviour_graphs.PythonHelper')
    ph = PythonHelper()
    #sizeA = ph.uniqueElements(D) + 2  #Plus 2 to include "start" and "end" behaviours
    sizeA = 6

    # First count the number of transitions between each behaviour.
    countedMatrix = [[[0 for k in range(0, sizeA)] for j in range(0, sizeA)] for i in range(0, len(D))]
    totalOccs = [[0 for j in range(0, sizeA)] for i in range(0, len(D))]

    for i in range(0, len(D)):
        previous = 0
        totalOccs[i][previous] += 1
        for current in D[i]:
            # print('i: ', i, ', previous: ', previous, ', current: ', current)
            countedMatrix[i][previous][current] += 1
            totalOccs[i][current] += 1
            previous = current
        countedMatrix[i][previous][sizeA-1] += 1  # Deal with last behaviour
        totalOccs[i][sizeA-1] += 1

    print("totalOccs: ", totalOccs)

    # Then calculate it as a percentage of total behaviours.
    transitionMatrices = [[[float(0) if totalOccs[i][row] == 0 else count/totalOccs[i][row] for count in countedMatrix[i][row]] for row in range(0, len(countedMatrix[i]))] for i in range(0, len(D))]
    print(transitionMatrices)

    totals = [len(sequence) for sequence in D]
    print(totals)

    for matrix in range(0, len(transitionMatrices)):
        for row in range(0, len(transitionMatrices[matrix])):
            transitionMatrices[matrix][row][0] = totalOccs[matrix][row]/totals[matrix]
    print(transitionMatrices)

    return transitionMatrices


def constrainedSumSamplePos(n, total, rangeGap):
    """Return a randomly chosen list of n positive integers summing to total.
    Each such list is equally likely to occur."""
    numpyRange = np.arange(0.0, total, rangeGap)
    range = np.ndarray.tolist(numpyRange)
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


def optimalK(data, nrefs=3, maxClusters=15):
    """
    Calculates KMeans optimal K using Gap Statistic from Tibshirani, Walther, Hastie
    Params:
        data: ndarry of shape (n_samples, n_features)
        nrefs: number of sample reference datasets to create
        maxClusters: Maximum number of clusters to test for
    Returns: (gaps, optimalK)
    """
    gaps = np.zeros((len(range(1, maxClusters)),))
    resultsdf = pd.DataFrame({'clusterCount': [], 'gap': []})
    #sizeA = getActionSpaceSize(data)
    sizeA = 6
    for gap_index, k in enumerate(range(1, maxClusters)):

        # Holder for reference dispersion results
        refDisps = np.zeros(nrefs)

        # For n references, generate random sample and perform EM clustering getting resulting dispersion of each loop
        for i in range(nrefs):
            # Create new random reference set
            randomReference = [np.random.randint(sizeA, size=np.random.randint(1, max([len(sequence) for sequence in data]))) for i in range(0, len(data))]
            print('randomReference', i, ', length =', len(randomReference), ':', randomReference)
            
            # Fit to it
            Z, clusterCentroids, transitionMatrices = cluster(k, randomReference)
            clusters = createClusterMatrices(Z, transitionMatrices, k)
            print('clusters for randomReference ', i, ': ', clusters)
            print('elements in each cluster: ', [len(c) for c in clusters])

            refDisp = inertia(clusterCentroids, clusters)
            print('refDisp for randomReference ', i, ': ', refDisp)
            refDisps[i] = refDisp

        # Fit cluster to original data and create dispersion
        Z, clusterCentroids, transitionMatrices = cluster(k, data)
        clusters = createClusterMatrices(Z, transitionMatrices, k)

        origDisp = inertia(clusterCentroids, clusters)

        # Calculate gap statistic
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)

        # Assign this loop's gap statistic to gaps
        gaps[gap_index] = gap

        resultsdf = resultsdf.append({'clusterCount': k, 'gap': gap}, ignore_index=True)

    return (gaps.argmax() + 1,
            resultsdf)  # Plus 1 because index of 0 means 1 cluster is optimal, index 2 = 3 clusters are optimal


def createClusterMatrices(Z, transitionMatrices, k):
    clusterMatrices = [[] for i in range(0, k)]
    # for each assignment in Z
    for i in range(0, len(Z)):
        # add corresponding transition matrix to appropriate element of return list
        clusterMatrices[Z[i]].append(transitionMatrices[i])

    return clusterMatrices


def inertia(centroids, clusters):
    # for each cluster
    return sum(
        [sum(
            [np.linalg.norm(np.array(clusters[i][j])-np.array(centroids[i])) for j in range(0, len(clusters[i]))]
        ) for i in range(0, len(centroids))]
    )
    # for each point in the cluster
    # get euclidean distance between point and centroid
    # square it
    # add to total
    # add to total


def displayGraphs(clusters):
    PythonHelper = autoclass('behaviour_graphs.PythonHelper')
    ph = PythonHelper()
    ph.createGraph(clusters)


if __name__ == '__main__':
    main()
