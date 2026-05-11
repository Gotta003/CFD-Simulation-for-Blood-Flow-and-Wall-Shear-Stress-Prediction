# CFD-Simulation-for-Blood-Flow-and-Wall-Shear-Stress-Prediction
AI-based predictive model that, using pre-operative CT scans, estimates the risk of adverse events after EVAR, such as endoleak, graft migration, or reintervention. The output consists in a patient-level risk prediction.  

# Setup
To be enable to runt the code in the following repository you need to:
```bash
chmod +x setup.sh
./setup.sh
```
This will create a virtual environment located in the folder and this will enable to run the scripts and the complete pipeline and install all the libraries located in "requirements.txt". To run the python scripts that follows in the description activate the virtual environment:
```bash
source .evar_env/bin/activate
```

## Structure of Patient Folder
The patients where located into a simulation_db in the same path in which this repository is cloned. Each patient other than the number should be structured as pz{number with 3 digits}. Each patient training has a CT scan pre-EVAR, saved in folder '../cta/pz{id}', a list of the reports related to that patient and the associated complications.

# Training Dataset Dashboard
To monitor the flow of each patient for the dataset extraction and processing we have created a dedicated dashboard. It gives a live view of where every patient stands across all five pipeline stages, and lets you annotate outcomes directly without editing CSV files by hand.
| Step | Detection Logic |
|------|-----------------|
| **SEGMENTATION**  | Checks whether a folder 'pz{id}' exists under any of the configured simulation_db base paths. |
| **REPORT ANALYSIS**  | Checks '../report/pz{id}' for files and on the right panel in patient checlist there is a way to keep track of the exploring files. To be signed to complete all the checkboxs must be checked. |
| **CFD Simulation**  | Looks inside '{sim_db}/pz{id}/Simulations/pz{id}/' for '*-procs' sub-folders. If '.vtp' files are found after the final train, they are automatically copied to 'data/vtp_files/pz{id}/' 
| **Image Processing** |
| **Labeling**  | 

## Segmentation - NNIterative

## CFD Simulation

First of all you have to process the data:


for running PINN model you also have to extract the wall data running extract_wall.py script.

After you processed the data you can run the experiments via one of these command:

python main.py --config "[config name]"

name of configurations implemented are
- xgboost
    run xgboost benchmark with optimal founded parameters on 5-fold
- pointnet
    run pointnet++ SOTA benchmark model with optimal founded parameters on 5-fold
- pinn
    run our implemented PINN model with optimal founded parameters on 5-fold

output of mains are saved in output directory, in particular:
- checkpoint
    contains all models obtained from each run
- results
    contains all models results obtained from each split

after you trained the model you can obtain the specific pathology prediction running the script (located in src/models/ directory)

python multilabel_classifier.py --modelpath [path to trained model]

to use the interface:
...
