import numpy.matlib
import numpy as np

"""The function that computes the entries in the log-likelihood matrix"""
def viterbiLog(logArr,transition, emission, observations):
     # base case where time t=1
    for i in range(len(logArr)):
        logArr[i][0]=np.log(pi[i])+np.log(emission[i][observations[0]]) #fill in the first column


    for j in range(len(logArr)):
        for t in range(len(logArr[j])):
            if t!=0:
                 logArr[j][t+1]=max([row[t] for row in logArr]+np.log([row[j] for row in transition]))+np.log(emission[j][observations[t+1]])


    return logArr

"""The function that fill in the entries of the most likely transition array I"""
def mostLikelytransition(I_arr,logArr,transition):
    for j in range(len(I_arr)):
        for t in range(len(I_arr[j])):
            I_arr[j][t+1]=np.argmax(logArr)




#pre-process the data from files

transition=np.genfromtxt("transition.txt")

emission=np.genfromtxt("emission.txt")

observations=np.genfromtxt("observations.txt")

pi=np.genfromtxt("initialState.txt")

row,col=27, 180000 #hard-coded the dimensions of the 2-D log array
logArr=[[0 for x in range(col)] for y in range(row)]

logArr=viterbiLog(logArr,transition,emission,observations)  #fill in every entry computed by Viterbi Algorithm

I_arr=logArr=[[0 for x in range(col)] for y in range(row)] #I array has the same size as logArr

