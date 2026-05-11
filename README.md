# CFD-Simulation-for-Blood-Flow-and-Wall-Shear-Stress-Prediction
AI-based predictive model that, using pre-operative CT scans, estimates the risk of adverse events after EVAR, such as endoleak, graft migration, or reintervention. The output consists in a patient-level risk prediction.  


First of all you have to process the data:
...

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