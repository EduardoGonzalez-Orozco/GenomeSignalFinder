
#####Notes V1.1
# Eliminating the option --first, so the code will know which was the first level in scan
#####Notes V1.2
## Adding a line to indicate significant signals based on sd to the mean 
## Option to plot hline of max or min depending on --type argument



from IPython.display import SVG, display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
import matplotlib.cm as cm
from statistics import mode
import pickle
import argparse
import csv



##### Arguments

#Arguments:
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input name should correspond as the same name used in scan or in case of breaker use the generated name before __decomposition_table.pkl or peak_Ids.pkl")
    parser.add_argument("--level", required=True, help="Selected level to visualze")
    parser.add_argument("--type",default="max", required=True, help="Select the analysis type for finding peak signals: max or min (Deafult:max)")
    parser.add_argument("--line", default="T", help="If 'T' a line indicating <sdline> standard deviations from mean")
    parser.add_argument("--sdline", default=1, help="Number of standard deviatons from mean to plot line")




   
    args = parser.parse_args()



###Decomposition table
with open(f'{args.input}_decomposition_table.pkl', "rb") as file:
    reconstructVals_results = pickle.load(file)

with open(f'{args.input}_peak_Ids.pkl', "rb") as file:
    resultsAll = pickle.load(file)




level=int(args.level)
signalType=str(args.type)


# ###Input: Table peak detection

###Input: Decompositon table
dataOTV = reconstructVals_results[:,level+2]
posFOTV=reconstructVals_results[:,0]
cmPosOTV=reconstructVals_results[:,1]
 
firstLevel=int(resultsAll[0][0][8])
levelTV=level -firstLevel## Must based on the position ofthe decomposition table (decide which level to visualize)
dataTV=resultsAll[levelTV]

## Selectin the values for peaks
startTV = [subarray[0] for subarray in dataTV]
endTV =[subarray[1] for subarray in dataTV]
cmstartTV = [subarray[3] for subarray in dataTV]
cmendTV = [subarray[4] for subarray in dataTV]
difcmTV =[subarray[5] for subarray in dataTV]
maxIndexTV=[subarray[6] for subarray in dataTV]
empowerTV = [subarray[7] for subarray in dataTV]


# print(dataTV)
fig, ax = plt.subplots()
desired_width = 15 # in inches
fig.set_size_inches(desired_width, fig.get_size_inches()[1])


# # Define a colormap based on the distances
cmap = cm.get_cmap('autumn')

#Funtion to flat the values
def flatten_list(lst):
    flattened_list = []
    for item in lst:
        if isinstance(item, list):
            flattened_list.extend(flatten_list(item))
        else:
            flattened_list.append(item)
    return flattened_list



# Example usage
one_dimensional_cmstartTV = flatten_list(cmstartTV)
one_dimensional_listEMP = flatten_list(empowerTV)
one_dimensional_listEMP = flatten_list(empowerTV)
one_dimensional_listCM = flatten_list(cmPosOTV)
one_dimensional_difcmTV = flatten_list(difcmTV)
one_dimensional_maxIndexTV_t = flatten_list(maxIndexTV)
one_dimensional_maxIndexTV = [int(element) for element in one_dimensional_maxIndexTV_t]


# # Plot the dataset
scatter=ax.plot(posFOTV,dataOTV, 'b-', label='Data')

# # Plot the data points
ax.scatter([posFOTV[i] for i in one_dimensional_maxIndexTV],
           [dataOTV[i] for i in one_dimensional_maxIndexTV],
           c=one_dimensional_listEMP, cmap=cmap)


for i, val,dist in zip(one_dimensional_maxIndexTV, one_dimensional_listEMP,one_dimensional_difcmTV):
    rounded_emp = round(val, 3)
    rounded_cmd = round(dist, 3)
    text = f"EMP:{rounded_emp}(Cm:{rounded_cmd}) "
    ax.text(posFOTV[i], dataOTV[i], text, ha='center', va='bottom')


# Plot starting and ending points


#Start
def get_index_numbers(lst, values):
    index_numbers = [index for index, item in enumerate(lst) if item in values]
    return index_numbers

index_numbers = get_index_numbers(posFOTV,list(set(startTV)))
ax.plot(posFOTV[index_numbers], dataOTV[index_numbers], 'g^', label='Signal Start')

# ##Ending

index_numbers = get_index_numbers(posFOTV,list(set(endTV)))
ax.plot(posFOTV[index_numbers], dataOTV[index_numbers], 'mv', label='Signal End')

### Adding second X axis legends for Cm
value=np.mean(dataOTV)
n=len(cmPosOTV)
arrSX = np.full(n, value)
ax2 = ax.twiny()
# print(cmPosOTV)
ax2.plot(cmPosOTV,arrSX,color='white', linewidth=0.001) # Create a dummy plot
ax.ticklabel_format(style='plain')

ax.set_xlabel('Base Pairs')
ax2.set_xlabel('Cm')
ax.legend()
##### Adding line to indicate significant signals
flagPlotLine=args.line
if flagPlotLine == "T":
    sdVal=int(args.sdline)
    sdV=np.std(dataOTV)
    cutoff1=value+sdVal*sdV
    cutoff2=value-sdVal*sdV
    if(signalType == "max"):
        ax2.axhline(y=cutoff1, color='r', linestyle='--')
    if(signalType == "min"):
        ax2.axhline(y=cutoff2, color='r', linestyle='--')



plt.savefig(f'{args.input}_Plot_DecomposeSignal_Level_{args.level}.png')

# print(dataTV)

with open(f'{args.input}_Peak_Table_Level_{args.level}.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    
    # Write each row of the table
    for row in dataTV:
        writer.writerow(row)
