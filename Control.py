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
    k = 5  # Number of clusters
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
    for i in range(0, len(D) - 1):
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
                    product = product * returnClusters[Z[zi]][D[i][j]][D[i][j - 1]]
                products.append(product)
            #print(products)
            Z[i] = numpy.argmax(products)
            new = Z[i]
            if initial != new:
                change = 1

        print("Z: ", Z)

        # Make a list of all sequences by cluster
        sequences = [[], [], [], [], []]
        for z in range(0, k - 1):
            for i in range(0, len(Z)):
                if Z[i] == z:
                    sequences[z].append(i)

        # M step - Update each transition matrix
        print("M step ", count)
        for z in range(0, k):
            for i in range(0, sizeA):
                for j in range(0, sizeA):
                    divisor = sumCountTransitions(sequences[z], sizeA, j, following)
                    '''if divisor == 0:
                        returnClusters[z][i][j] = 0.0
                    else:'''
                    if divisor != 0:
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
