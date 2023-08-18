# CardiOnco_TFE

To install the environment and dependencies :
First install Anaconda, then open the Anaconda prompt, write:
`conda env create -f environment.yaml`

Then choos the `envTFE` environment in Anaconda navigator, install spyder, launch it and open the `basic_window.py` file.
Launch the file with the "play" button in Spyder and have fun !

------

## In case Spyder does not launch
- Uninstall `envTFE` from the Anaconda **navigator** ("environments" page in the left-side of the interface)
- Create a new virtual environment (optional)
- In Anaconda **prompt**, activate the new environment (if necessary) with `conda activate newEnvName` (with "newEnvName" the name of your environment ; if you did not create a new environment, skip this step)
- In the anaconda prompt, write : `conda uninstall spyder-kernels` and once it is done, write `conda install spyder-kernels=0.*`
- Finally, in the anaconda prompt, write : `pip install numpy scipy matplotlib pyQt6 pyqtgraph neurokit2 scikit-learn` and once done, close anaconda prompt.
- Open Anaconda navigator, choose the new environment (or stay in "base" if you have not created a new one) then launch spyder and launch `basic_window.py`
