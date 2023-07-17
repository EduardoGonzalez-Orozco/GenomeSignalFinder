#!/usr/bin/env python
# coding: utf-8

# In[51]:


########################First tool: Scan 
#Input= bed file (first column SNP position, SNP cm, summary statistic value) 
##Output=decompoisitonTable for all levels, plots for EMP and LEN distribution to select signal
############################
##################################################################
####NOTES:
#-> Elimininate a bug located in the signal size detection also some "trash" lines 
##V1.1
# -> Eliminate the use of --first in visualize and break (Working on...)


import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
import pywt
import matplotlib.cm as cm
from statistics import mode
import multiprocessing
import pickle

#Arguments:
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input file coma separated (csv) with: <position Base Pairs>,<position centi morgans>,<value>")
    parser.add_argument("--outname", required=True,help="Output name")
    parser.add_argument("--levels", default=5, help="Maximum decomposition level")
    parser.add_argument("--sd", default=10, help="Number of standard deviations to the mean to denoise")
    parser.add_argument("--tmode", default="soft", help="Threshold mode : 'soft' 'hard' " )
    parser.add_argument("--wavelet", default="db10", help="Wavelet type 'db1-43' " )
    parser.add_argument("--first", default=10, help="(First level to Maximum level) for signal scanning after 1M values levels after 10 is recommended " )
    parser.add_argument("--cores", default=1, help="Number of cores to parallelize the signal detection" )

    args = parser.parse_args()



######## Reading files 

dataI = np.genfromtxt(args.input, delimiter=',')

pos_index = 0
val_index = 2
cm_index=1

# Capturing data
val=dataI[:,val_index]
pos=dataI[:,pos_index]
centM=dataI[:,cm_index]


##Checkin pair or unpair
num_lines=len(val)
if num_lines % 2 == 0:
    flag2=1
else:
    flag2=2


############# Parte 1 transformation and denoising
# ##################--------- Denoising_using_Wavlets------####################


def filling_zeros_DC(arr_list,levelD):
    for i in range(levelD,len(arr_list)):
        fill=np.zeros(len(arr_list[i]))
        arr_list[i] = fill
    return arr_list

def denoise_data(arr_list,sdV,modeL):
    filtered_list = []
    for arr in arr_list:
        mean_value=np.median(np.abs(arr))
        std_value=np.std(np.abs(arr))
        threshold=mean_value+(sdV*std_value)
        filtered_arr =pywt.threshold(arr, threshold, mode=modeL, substitute=0)
        filtered_list.append(filtered_arr)
    return filtered_list


# Levels to decompose
levels=int(args.levels)
#Levels to retainn
numDforRecon=2 #Min must be 2
#Thresholding mehod
threshMode=args.tmode
#Number of sd  from mean to filter
sdF=int(args.sd)
#Wavelet type
waveType=args.wavelet

#######Creating results array
reconstructVals_results = np.empty((len(val),levels+3))  # Initialize an empty array to store the results
#Filling first column with positions
reconstructVals_results[:, 0] = pos
reconstructVals_results[:, 1] = centM
reconstructVals_results[:, 2] = val


###### Procesing signal
for i in range(3, levels+3):
    wavelet_coeffs = pywt.wavedec(val,waveType, level=i-1)
    wavelet_coeffs_aproxCoeff=wavelet_coeffs[0]
    wavelet_coeffs_detailxCoeff=wavelet_coeffs[1:]

     #print(wavelet_coeffs_detailxCoeff)

     #Denoising based on mean and sd 
    denoised_detailed_waveletCoeff= denoise_data(wavelet_coeffs_detailxCoeff,sdF,threshMode)
    coeflToRecTmp=list([wavelet_coeffs_aproxCoeff]+denoised_detailed_waveletCoeff)

     ### Eliminating first levels detailed coefficients 
    coeflToRec=filling_zeros_DC(coeflToRecTmp,numDforRecon)

     #Reconstructing signal
    siganlRec=pywt.waverec(coeflToRec,waveType)

    if flag2 == 1:
        siganlRec2=siganlRec
    if flag2 == 2:
        siganlRec2=siganlRec[:-1]

    #Filing the result array
    reconstructVals_results[:, i]=siganlRec2


with open(f'{args.outname}_decomposition_table.pkl', "wb") as file_recV:
    pickle.dump(reconstructVals_results, file_recV)

#################--------- Detecting peaks and estimating emp (Parallel)------####################

resultsAll =[]
firstLevel=int(args.first)

# for b in range(1,levels+1):
for p in range(firstLevel, levels):

    l=p
    data = reconstructVals_results[:,l+2] 
    posF=reconstructVals_results[:,0]
    cmPos=reconstructVals_results[:,1]

    # Calculate the mean of all values
    mean_value = np.mean(data)

    # Find local maxima
    maxima_indices = argrelextrema(data, np.greater)[0]

    # Identify start and end points of each peak
    peak_start_indices = []
    peak_end_indices = []
    cenM_start=[]
    cenM_end=[]

    ####Blank array for results

    for min_index in maxima_indices:
        # Find the left neighbor of the peak
        left_neighbor = min_index - 1

        # Keep moving to the left until the value is higher than the mean or a local maximum is encountered
        while left_neighbor >= 0 and (data[left_neighbor] >= mean_value):
            left_neighbor -= 1

        # Find the right neighbor of the peak
        right_neighbor = min_index + 1

        # Keep moving to the right until the value is higher than the mean or a local maximum is encountered
        while right_neighbor < len(data) and (data[right_neighbor] >= mean_value):
            right_neighbor += 1

        # Add the start and end indices of the peak
        peak_start_indices.append(left_neighbor + 1)
        peak_end_indices.append(right_neighbor - 1)


    ##Generating the joined table results for peaks
    firstLevelColumn= np.full(len(peak_end_indices), firstLevel)
    peakDetectionResultsT = np.column_stack((peak_start_indices, peak_end_indices, maxima_indices,firstLevelColumn))

   ###No unique filer Si quieres filro debes comentar esta sección:
    peakDetectionResults=peakDetectionResultsT


    peak_index=peakDetectionResults[:,2]


    def estimate_emp(signalV, noiseV):
        max_signal_power = np.max(signalV)
        base_lineN=np.mean(noiseV)
        noise=np.var(noiseV)

        Ht=np.absolute(max_signal_power-base_lineN)

        emp = ((2*Ht)-(0.5*noise))/noise
        return emp


    # Define a worker function for parallel processing
    def process_data(args):
        a, indexS = args 
        startI = peakDetectionResults[a, 0]
        endI = peakDetectionResults[a, 1]
        maxPos=peakDetectionResults[a, 2]
        firstL=peakDetectionResults[a, 3]

        start = reconstructVals_results[startI, 0]
        end = reconstructVals_results[endI, 0]

        startcm = reconstructVals_results[startI, 1]
        endcm = reconstructVals_results[endI, 1]

        noiseValIndex = 1

        maskSignal = (reconstructVals_results[:, 0] >= start) & (reconstructVals_results[:, 0] <= end)
        signal = reconstructVals_results[maskSignal]

        maskNoise = (reconstructVals_results[:, 0] < start) | (reconstructVals_results[:, 0] > end)
        noise = reconstructVals_results[maskNoise]

        result = estimate_emp(signal[:, l + 2], noise[:, l + 2])

        difLen=end-start
        difCm=endcm-startcm

        return np.array([start, end, difLen,startcm, endcm, difCm,maxPos,result,firstL,indexS])
    
    if __name__ == '__main__':
        
        # Number of processes to use (adjust as needed)
        num_processes =int(args.cores)

        # Create a pool of processes
        pool = multiprocessing.Pool(processes=num_processes)

        # Map the worker function to the range of values
        args_list = [(a, indexS) for indexS, a in enumerate(range(len(peak_index)))]
        results_list = pool.map(process_data, args_list)
        resultsAll.append(results_list)

    
        pool.close()
        pool.join()

with open(f'{args.outname}_peak_Ids.pkl', "wb") as file:
    pickle.dump(resultsAll, file)

length = [[subarray[5] for subarray in sublist] for sublist in resultsAll]
empower = [[subarray[7] for subarray in sublist] for sublist in resultsAll]
####### Plot variance Empower vs Length 


# range(firstLevel, levels)

resEmp = []
resLen = []

length = [[subarray[5] for subarray in sublist] for sublist in resultsAll]

colors = plt.cm.tab10(range(1,levels-firstLevel))


for p in range(1,levels-firstLevel):
    varE = np.var(empower[p])
    varL = np.var(length[p])

    resEmp.append(varE)
    resLen.append(varL)

plt.scatter(resEmp, resLen, c=colors[:len(resEmp)], marker='o')

# Add legend
for i, (varE, varL) in enumerate(zip(resEmp, resLen)):
    plt.text(varE, varL, f'Level {i+1+firstLevel}', fontsize=8)

plt.xlabel('Empower')
plt.ylabel('Length Variance')
plt.legend(loc='best')
plt.savefig(f'{args.outname}_diagnostic.png')





# In[50]:


