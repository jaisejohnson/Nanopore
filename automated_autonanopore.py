#Original Code Written by https://github.com/bellstwohearted/AutoNanopore
#Modified with plotting and facility to visualize the detected events and record the parameters used to find events
#Jaise Johnson, CeNSE, IISc Bangalore, INDIA
import os
import subprocess
import argparse
import numpy as np
import pyabf
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import time
from pathlib import Path
style.use('ggplot')


# Path to the folder containing signal files


fplis = [] #list out the folder paths containi

signal_dir,theta,window_size = '1','1.5','30'

for fp in fplis:
    signal_folder = fp
    outpath = os.path.join(fp,'anpr')
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    print(outpath)

    # List all signal files in the folder
    signal_files = [os.path.join(signal_folder, file) for file in os.listdir(signal_folder) if file.endswith('.abf')]
    print(len(signal_files),signal_files)
    # Path to the Python script you want to run
    script_to_run = r"C:\Users\Smart\PycharmProjects\programcomp\AutoNanopore.py"

    # Loop through each signal file and analyze it
    abffilenames, processing_time, thetalist, wslist, = [], [], [], []
    proceed = 'y'
    i = 0
    while i < len(signal_files)-1:
        if proceed == 'y':
            signal_file = signal_files[i]
            print(i,signal_file)
        elif proceed == 'n':
            signal_file = signal_files[i]
            print(i,signal_file)
            signal_file = signal_files[i]
            theta = input('theta:')
            window_size = input('window_size:')

        current = pyabf.ABF(signal_file).data[0]
        abffilenm = Path(signal_file).stem
        print(abffilenm)


        fig,axs = plt.subplots()
        axs.plot(current,linewidth = '0.4',color = 'DarkBlue')
        axs.set_title(abffilenm)
        # Construct the command to run the Python script with arguments
        cmd = [
            "python",  # or "python3" depending on your environment
            script_to_run,
            "--file_path", signal_file,
            "--output_path", outpath,  # Replace with your desired output directory
            "--signal_direction", signal_dir,  # Specify your desired values for these arguments
            "--theta", theta,
            "--window_size", window_size
        ]

        # Run the command in the command prompt
        try:
            st = time.time()
            subprocess.run(cmd, check=True)
            pt = time.time() - st
            print(f"Analysis completed for {signal_file}")
            outsdf = pd.read_csv(os.path.join(outpath,abffilenm+'.csv'))
            evsp =outsdf['Event start index']
            evep = outsdf['Event end index']
            for j in range(len(evsp)):
                axs.plot(np.arange(evsp[j], evep[j]), current[evsp[j]:evep[j]], linewidth=0.3, color='Red')
                axs.scatter(evsp[j], current[evsp[j]])

            plt.show()
            proceed = input('Proceed:')

            if proceed == 'y':
                abffilenames.append(abffilenm)
                processing_time.append(pt)
                thetalist.append(theta)
                wslist.append(window_size)
                i = i + 1
            elif proceed == 'n':
                i = i
            else:
                print("YOU PRESSED THE WRONG BUTTON")
                i = i
        except subprocess.CalledProcessError:
            print(f"Error analyzing {signal_file}")
            print('NO EVENTS DETECTED CHANGE PARAMETERS')
            proceed ='n'
            i = i

    timingdict = {'Filenames':abffilenames,
                  'Processing Time(s)':processing_time,
                  'Theta':thetalist,
                  'Window Size (ms)':wslist}
    timedf = pd.DataFrame(timingdict)
    timedf.to_csv(os.path.join(fp,'anpr/Processing_times.csv'))


