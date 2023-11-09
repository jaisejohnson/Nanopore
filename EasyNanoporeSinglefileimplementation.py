#Originally written by: https://github.com/MelancholeyLemon/EasyNanopore/tree/master
#Modified by Jaise Johnson, CeNSE, IISc Bangalore, India
#Added abilities to handle multiple files together, with plotting window and recording the input parameters

import pandas as pd
from pyabf import ABF
import os
import eventDetect
from datetime import datetime
import sys
import time
import matplotlib.pyplot as plt
import scipy.signal
import numpy as np
from pathlib import Path
from multiprocessing import Pool,freeze_support


def eventDownFast(rawSignal, startCoeff, endCoeff, filterCoeff, minDuration, maxDuration, outpath, sampleRate,filefn):
    starttime = time.time()
    iterNumber = 0
    fileNumber = 0
    nowTime = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    padLen = np.int64(sampleRate)
    prepadded = np.ones(padLen) * np.mean(rawSignal[0:1000])
    signalToFilter = np.concatenate((prepadded, rawSignal))
    rawSignal = np.array(rawSignal)

    mlTemp = scipy.signal.lfilter([1 - filterCoeff, 0], [1, -filterCoeff], signalToFilter)
    vlTemp = scipy.signal.lfilter([1 - filterCoeff, 0], [1, -filterCoeff], np.square(signalToFilter - mlTemp))

    ml = np.delete(mlTemp, np.arange(padLen))
    vl = np.delete(vlTemp, np.arange(padLen))

    sl = ml - startCoeff * np.sqrt(vl)
    Ni = len(rawSignal)
    points = np.array(np.where(rawSignal <= sl)[0])
    to_pop = np.array([])
    for i in range(1, len(points)):
        if points[i] - points[i - 1] == 1:
            to_pop = np.append(to_pop, i)
    to_pop = np.int64(to_pop)
    points = np.unique(np.delete(points, to_pop))
    NumberOfEvents = 0;
    RoughEventLocations = np.zeros((10000, 3))
    event_current = np.zeros((10000, 3))
    minc = np.zeros(10000)
    print('Length of Points:',len(points))
    print('Length of Points to Pop:',len(to_pop))
    for i in points:
        event_start_mean = ml[i-100]
        if i >= Ni - 10:
            break;
        start = i
        El = ml[i] - endCoeff * np.sqrt(vl[i])
        while rawSignal[i + 1] < El and i <= Ni - 10:
            i = i + 1
        if ((minDuration * sampleRate / 1000) < (i + 1 - start)) and ((i + 1 - start) < (maxDuration * sampleRate / 1000)) and (event_start_mean - min(rawSignal[start:i + 1])) > 0:
            NumberOfEvents = NumberOfEvents + 1
            RoughEventLocations[NumberOfEvents - 1, 2] = i + 1 - start
            RoughEventLocations[NumberOfEvents - 1, 0] = start
            RoughEventLocations[NumberOfEvents - 1, 1] = i + 1
            minc[NumberOfEvents-1] = min(rawSignal[start:i+1])
            event_current[NumberOfEvents - 1, 0] = (event_start_mean - min(rawSignal[start:i + 1]))
            event_current[NumberOfEvents - 1, 1] = event_start_mean
            event_current[NumberOfEvents - 1, 2] = abs(event_current[NumberOfEvents - 1, 0] / event_current[NumberOfEvents - 1, 1]) * 10000

    event_statistic = np.zeros((NumberOfEvents, 6))
    #event_statistic[:, 0] = RoughEventLocations[0: NumberOfEvents, 2] / sampleRate * 1000
    event_statistic[:, 0] = RoughEventLocations[0: NumberOfEvents, 2]
    event_statistic[:, 1:4] = event_current[0: NumberOfEvents, 0:3]
    if iterNumber == 0:
        #event_statistic[:, 4] = (RoughEventLocations[0: NumberOfEvents, 0] + iterNumber * sampleRate * 10) * 1000 / sampleRate
        event_statistic[:, 4] = (RoughEventLocations[0: NumberOfEvents,0])
    else:
        #event_statistic[:, 4] = (RoughEventLocations[0: NumberOfEvents,0] + iterNumber * sampleRate * 10) * 1000 / sampleRate -10
        event_statistic[:, 4] = (RoughEventLocations[0: NumberOfEvents,0] )
    event_statistic[:, 5] = fileNumber
    minc = minc[:len(event_statistic)]
    with open(outpath, "a+") as fp:
        np.savetxt(fp, event_statistic, fmt='%.3f', delimiter="\t")

    eventloc = np.array(event_statistic[:,4]).astype(int)
    eventwidth = np.array(event_statistic[:,0]).astype(int)
    evnd = eventloc+eventwidth

    current_intensity = np.array(event_statistic[:,1])
    Baseline_current = np.array(event_statistic[:,2])
    ratio = np.array(event_statistic[:, 3])

    print('Eventlocs:',list(eventloc))
    print('Event ends:',list(evnd))
    print('RAWSIGNAL:',rawSignal.shape)
    savedict = {'Event Start Point':eventloc,
                'Event End Point': evnd,
                'Event Width': eventwidth,
                'Current_Intensity (nA)':current_intensity,
                'Baseline_Current (nA)':Baseline_current,
                'Ratio':ratio
                }
    print('Lengths:',len(eventloc),len(evnd),len(eventwidth),len(current_intensity),len(Baseline_current))
    df = pd.DataFrame(savedict)
    endtime = time.time()
    elapsed = endtime - starttime
    print('elapsed',elapsed)
    #df.to_csv(outpath+nowTime+'.csv')

    plt.plot(rawSignal, color='DarkBlue', linewidth='0.6')
    for i,e in enumerate(eventloc):
        plt.plot(np.arange(e,evnd[i]),rawSignal[e:evnd[i]],color='Red',linewidth = '0.6')
    plt.scatter(evnd,rawSignal[evnd],marker='*')
    plt.scatter(eventloc, rawSignal[eventloc], marker='s')
    plt.title(filefn)
    plt.show()

    return elapsed,df,outpath


def eventUpFast(rawSignal, startCoeff, endCoeff, filterCoeff, minDuration, maxDuration, outpath, sampleRate,filefn):
    starttime = time.time()
    iterNumber = 0
    fileNumber = 0
    nowTime = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    padLen = np.int64(sampleRate)
    prepadded = np.ones(padLen) * np.mean(rawSignal[0:1000])
    signalToFilter = np.concatenate((prepadded, rawSignal))
    rawSignal = np.array(rawSignal)

    mlTemp = scipy.signal.lfilter([1 - filterCoeff, 0], [1, -filterCoeff], signalToFilter)
    vlTemp = scipy.signal.lfilter([1 - filterCoeff, 0], [1, -filterCoeff], np.square(signalToFilter - mlTemp))

    ml = np.delete(mlTemp, np.arange(padLen))
    vl = np.delete(vlTemp, np.arange(padLen))

    sl = ml + startCoeff * np.sqrt(vl)
    Ni = len(rawSignal)
    points = np.array(np.where(rawSignal >= sl)[0])
    to_pop = np.array([])
    for i in range(1, len(points)):
        if points[i] - points[i - 1] == 1:
            to_pop = np.append(to_pop, i)
    to_pop = np.int64(to_pop)
    points = np.unique(np.delete(points, to_pop))
    NumberOfEvents = 0
    RoughEventLocations = np.zeros((10000, 3))
    event_current = np.zeros((10000, 3))
    minc = np.zeros(10000)
    print('Length of Points:',len(points))
    print('Length to Pop:', len(to_pop))

    for i in points:
        event_start_mean = ml[i-100]
        if i >= Ni - 10:
            break;
        start = i
        El = ml[i] + endCoeff * np.sqrt(vl[i])
        while rawSignal[i + 1] > El and i <= Ni - 10:
            i = i + 1
        if ((minDuration * sampleRate / 1000) < (i + 1 - start)) and ((i + 1 - start) < (maxDuration * sampleRate / 1000)) and (max(rawSignal[start:i + 1]) - event_start_mean) > 0:
            NumberOfEvents = NumberOfEvents + 1
            RoughEventLocations[NumberOfEvents - 1, 2] = i + 1 - start
            RoughEventLocations[NumberOfEvents - 1, 0] = start
            RoughEventLocations[NumberOfEvents - 1, 1] = i + 1
            event_current[NumberOfEvents - 1, 0] = (max(rawSignal[start:i + 1]) - event_start_mean)
            event_current[NumberOfEvents - 1, 1] = event_start_mean
            event_current[NumberOfEvents - 1, 2] = abs(event_current[NumberOfEvents - 1, 0] / event_current[NumberOfEvents - 1, 1]) * 10000


    event_statistic = np.zeros((NumberOfEvents, 6))
    event_statistic[:, 0] = RoughEventLocations[0: NumberOfEvents, 2] / sampleRate * 1000
    event_statistic[:, 1:4] = event_current[0: NumberOfEvents, 0:3]
    if iterNumber == 0:
        event_statistic[:, 4] = (RoughEventLocations[0: NumberOfEvents, 0] + iterNumber * sampleRate * 10) * 1000 / sampleRate
    else:
        event_statistic[:, 4] = (RoughEventLocations[0: NumberOfEvents,
                                 0] + iterNumber * sampleRate * 10) * 1000 / sampleRate -10
    event_statistic[:, 5] = fileNumber
    minc = minc[:len(event_statistic)]


    eventloc = np.array(event_statistic[:, 4]).astype(int)
    eventwidth = np.array(event_statistic[:, 0]).astype(int)
    evnd = eventloc + eventwidth
    current_intensity = np.array(event_statistic[:, 1])
    Baseline_current = np.array(event_statistic[:, 2])
    ratio = np.array(event_statistic[:, 3])


    print('Eventlocs:', list(eventloc))
    print('Event ends:', list(evnd))
    print('RAWSIGNAL:', rawSignal.shape)
    savedict = {'Event Start Point': eventloc,
                'Event End Point': evnd,
                'Event Width': eventwidth,
                'Current_Intensity (nA)': current_intensity,
                'Baseline_Current (nA)': Baseline_current,
                'Ratio': ratio
                }
    print('Lengths of Events:',len(eventloc), len(evnd), len(eventwidth), len(current_intensity), len(Baseline_current))
    df = pd.DataFrame(savedict)
    #df.to_csv(outpath + nowTime + '.csv')
    elapsed = (time.time() - starttime)
    fig,axs = plt.subplots()
    axs.plot(rawSignal, color='DarkBlue', linewidth='0.6')
    for i, e in enumerate(eventloc):
        axs.plot(np.arange(e, evnd[i]), rawSignal[e:evnd[i]], color='Red', linewidth='0.6')
    axs.scatter(evnd, rawSignal[evnd], marker='*')
    axs.scatter(eventloc, rawSignal[eventloc], marker='s')
    axs.set_title(filefn)
    plt.show()

    return elapsed,df,outpath
    '''with open(fileName, "a+") as fp:
        np.savetxt(fp, event_statistic, fmt='%.3f', delimiter="\t")'''
def detectMain(filepath,pattern, startCoeff, endCoeff, filterCoeff, minDuration, maxDuration):


    filefn = Path(filepath).stem
    print(filefn)
    dirname = os.path.dirname(filepath)
    if not os.path.exists(dirname+'\\esynpr'):
        os.makedirs(dirname+'\\esynpr')
    resultName = os.path.join(dirname +'\\esynpr' + '\\'+str(filefn))
    print(resultName)
    abf = ABF(filepath)
    current = abf.data[0]

    if pattern == "down":
        elapsed,df,outpath = eventDownFast(current, startCoeff, endCoeff, filterCoeff, minDuration, maxDuration, resultName,abf.dataRate,filefn)
    elif pattern == "up":
       elapsed,df,outpath = eventUpFast(current, startCoeff, endCoeff, filterCoeff, minDuration, maxDuration, resultName,abf.dataRate,filefn)
    else:
        sys.exit(1)

    print('Elapsed Time:',elapsed)
    print('finished')
    return elapsed,df,outpath


pattern = 'down'
startCoeff = 3.71
endCoeff = 0.12
filterCoeff = 0.978
minDuration = 0.001
maxDuration = 10


fplis = [] # List of folders containing .abf files

#out = detectMain(filepath,pattern, startCoeff, endCoeff, filterCoeff, minDuration, maxDuration)
proceed = 'y'
for fp in fplis:
    fnames = os.listdir(fp)
    fnames = [fname for fname in fnames if fname.endswith('.abf')]
    print(len(fnames),fnames)
    abffilenames, processing_time,stcoeflist,endcoeflist,filtercoefflist = [], [],[],[],[]
    i = 0
    while i <= len(fnames)-1:
        try:
            if proceed == 'y':
                fname = fnames[i]
                print(i,fname)
            elif proceed == 'n':
                fname = fnames[i]
                print(i,fname)
                startCoeff = float(input('startCoeff:'))
                endCoeff = float(input('endCoeff:'))
                filterCoeff = float(input('filterCoeff:'))

            if os.path.splitext(os.path.join(fp,fname))[1] == '.abf':

                filepath = os.path.join(fp,fname)
                elapsed,df,outpath = detectMain(filepath,pattern, startCoeff, endCoeff, filterCoeff, minDuration, maxDuration)
                print('ELAPSED TIME:',elapsed)
                nowTime = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
                proceed = input('Proceed? :')

                if proceed == 'y':
                    i = i + 1
                    stcoeflist.append(startCoeff)
                    endcoeflist.append(endCoeff)
                    filtercoefflist.append(filterCoeff)
                    abffilenames.append(fname)
                    processing_time.append(elapsed)
                    plt.savefig(fname+'esynpr.png')
                    df.to_csv(outpath +'esynpr.csv')
                elif proceed == 'n':
                    i = i
                else:
                    print('YOU PRESSED THE WRONG BUTTON')
                    print(len(abffilenames),'abffilenames so far:',abffilenames)
                    i = i
        except:
            print(f"Error analyzing {fname}")
            print('NO EVENTS DETECTED CHANGE PARAMETERS')
            proceed = 'n'
            i = i


    timingdict = {'Filenames':abffilenames,
                  'Processing Time (s)':processing_time,
                  'startCoeff':stcoeflist,
                  'endCoeff':endcoeflist,
                  'fiterCoeff':filtercoefflist}
    timedf = pd.DataFrame(timingdict)
    timedf.to_csv(os.path.join(fp +'\\esynpr' + '\\'+'Processing_times'+nowTime+'.csv'))