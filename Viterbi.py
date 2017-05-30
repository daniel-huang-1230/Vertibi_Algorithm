
import matplotlib.pyplot as plt
import numpy as np

"""The function that computes the entries in the log-likelihood matrix"""
def viterbiLog(logArr,transition, emission, observations):
     # base case where time t=1
    for i in xrange(len(logArr)):
        logArr[i][0]=np.log(pi[i])+np.log(emission[i][observations[0]]) #fill in the first column


    for j in xrange(len(logArr)):
        for t in xrange(len(logArr[j])):
            if t!=0 and t+1<len(logArr[j]):
                 logArr[j][t+1]=np.max([row[t] for row in logArr]+np.log([row[j] for row in transition]))+np.log(emission[j][observations[t+1]])

    return logArr

"""The function that fill in the entries of the most likely transition array I"""
def mostLikelytransition(I_arr,logArr,logtransition):
    for j in xrange(len(I_arr)):
        for t in xrange(len(I_arr[j])) :
            if t+1<len(I_arr[j]):
                l=np.array([row[t] for row in logArr])
                log_a=np.array([row[j] for row in logtransition])
                I_arr[j][t+1]=np.argmax(l+log_a)

    return I_arr



#pre-process the data from files

transition=np.genfromtxt("transition.txt")

emission=np.genfromtxt("emission.txt")

observations=np.genfromtxt("observations.txt")

pi=np.genfromtxt("initialState.txt")

row,col=27, 180000 #hard-coded the dimensions of the 2-D log array
logArr=[[0 for x in xrange(col)] for y in xrange(row)]

logArr=viterbiLog(logArr,transition,emission,observations)  #fill in every entry computed by Viterbi Algorithm

I_arr=[[0 for x in xrange(col)] for y in xrange(row)] #I array has the same size as logArr
logtransition=transition
for entry in logtransition:
    entry=np.log(entry)


I_arr=mostLikelytransition(I_arr,logArr,logtransition)

#Now we can compute the most likely sequence S*

s_sequence=[0 for t in xrange(10)]

s_sequence[col-1]=np.argmax([row[col-1] for row in logArr]) #at time=T

for t in xrange(col-2,-1,-1):
    s_sequence[t]=I_arr[s_sequence[t+1]][t+1]



for st in s_sequence:
    print(st)