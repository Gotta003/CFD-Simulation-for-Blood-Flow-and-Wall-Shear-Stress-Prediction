import numpy as np
import torch 
import pandas as pd 
from datasets.evar_dataset import EVARDataset
import matplotlib.pyplot as plt
from typing import List
import pickle
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.ticker as ticker
 

def plot_3d_errors_comparison(gt_matrix, pred_matrix, title, labels=['pressure', 'wss', 'velocity']):
    gt_matrix = np.array(gt_matrix)
    pred_matrix = np.array(pred_matrix)
    
    # Se le matrici hanno shape [N, 6] (X,Y,Z, P,W,V)
    gt_vals = gt_matrix[:, 3:]      
    pred_vals = pred_matrix[:, 3:]  
    coords = gt_matrix[:, :3]  # Coordinate XYZ

    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{'type': 'scene'}]*3],
        subplot_titles=[f"SE Reale: {l}" for l in labels] # Cambiato titolo
    )

    custom_colorscale = [[0, 'rgb(0, 255, 0)'], [1, 'rgb(255, 0, 0)']]

    for i in range(3):
        # L'errore ora è espresso nelle unità di misura reali (es. Pa^2)
        se = (gt_vals[:, i] - pred_vals[:, i])**2
        
        # Il clipping al 98° percentile è ancora più utile con dati non normalizzati
        # perché gli outlier possono avere valori molto alti
        v_max = np.percentile(se, 98)
        if v_max == 0: v_max = 1e-7

        fig.add_trace(
            go.Scatter3d(
                x=coords[:, 0], 
                y=coords[:, 1], 
                z=coords[:, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color=se,
                    cmax=v_max, # Forza il limite superiore per il contrasto
                    colorscale=custom_colorscale,
                    colorbar=dict(
                        title=f"{labels[i]} SE",
                        x=0.31 + (i * 0.34),
                        thickness=10
                    ),
                    showscale=True
                ),
                # Formattazione scientifica se i valori sono molto piccoli o grandi
                text=[f"Err: {val:.2e}" for val in se], 
                hoverinfo='text'
            ),
            row=1, col=i+1
        )
    fig.update_layout(
        title_text=title,
        title_x=0.5,
        height=700,
        width=1600,
        # Rimuoviamo eventuali luci che creano ombre sui colori
        scene=dict(aspectmode='data'),
        scene2=dict(aspectmode='data'),
        scene3=dict(aspectmode='data')
    )
    
     # Sincronizziamo le telecamere (opzionale, utile per ruotarli tutti insieme)
    # Nota: Plotly non supporta nativamente il sync perfetto al mouse in questo modo, 
    # ma possiamo impostare una vista iniziale comune.
    fig.update_scenes(xaxis_title='X', yaxis_title='Y', zaxis_title='Z')
    fig.show()

def extract_altman_plot(gts: List[np.ndarray], preds: List[np.ndarray], title = "all pointclouds plot",  labels=['pressure', 'wss', 'velocity']):

    # Assicurati che gts e preds siano [N, 6] e già denormalizzati
    gt_features = gts[:, 3:]
    pred_features = preds[:, 3:]

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    plt.suptitle(title, fontsize=18, fontweight='bold', y=0.98)

    for i in range(3):
        gt = gt_features[:, i]
        pred = pred_features[:, i]
        
        mean = (gt + pred) / 2
        diff = pred - gt # Errore in unità reali
        md = np.mean(diff)
        sd = np.std(diff)

        ax = axes[i]
        ax.scatter(mean, diff, alpha=0.15, s=1, color='royalblue') # Alpha più basso per densità
        
        ax.axhline(0, color='black', linestyle='-', linewidth=1)
        ax.axhline(md, color='red', linestyle='--')
        ax.axhline(md + 1.96*sd, color='gray', linestyle='--')
        ax.axhline(md - 1.96*sd, color='gray', linestyle='--')

        ax.set_title(f'Bland-Altman: {labels[i]}', fontsize=14)
        ax.set_xlabel(f'Media GT/Pred ({labels[i]})', fontsize=11)
        ax.set_ylabel(f'Differenza (Pred - GT)', fontsize=11)
        
        # Aumentata precisione decimale o scientifica per le label
        ax.text(1.01, md, f'Bias: {md:.4f}', transform=ax.get_yaxis_transform(), color='red', fontsize=9)
        ax.text(1.01, md + 1.96*sd, f'+1.96SD: {1.96*sd:.4f}', transform=ax.get_yaxis_transform(), color='gray', fontsize=8)

        ax.xaxis.set_major_locator(ticker.MaxNLocator(6))
        ax.tick_params(axis='x', rotation=30)

    plt.subplots_adjust(wspace=0.4, bottom=0.15, top=0.85, left=0.07, right=0.92)
    plt.show()

def extract_pinn_out(prediction_list: List[torch.tensor] = []):
    dataset = EVARDataset(normalize = True, 
                          split_ids=np.load("/home/group4/Challenge3/vtp_analysis/outputs/splits/test_ids.npy"))
    ids_list = []
    ground_truth_list = []
    for i in range(len(dataset)):
        point, _, _  = dataset[i]
        ids_list.append(dataset.df["patient_id"].iloc[i])
        ground_truth_list.append(np.transpose(point, (1, 0)))
    prediction_list = None   
    with open('./prediction_list.pkl', 'rb') as f:
        prediction_list = pickle.load(f)
    denormalized_predictions = []
    denormalized_ground_truths = []
    for i in range(len(prediction_list)):
        # Denormalize predictions and ground truths if needed
        pred = prediction_list[i]
        gt = ground_truth_list[i]
        denormalized_pred = dataset.denormalize(np.transpose(pred, (1, 0)))
        denormalized_gt = dataset.denormalize(np.transpose(gt, (1, 0)))
        denormalized_predictions.append(denormalized_pred)
        denormalized_ground_truths.append(denormalized_gt)
    # extract global altman plot
    gt_all = np.vstack(denormalized_ground_truths)
    pred_all = np.vstack(denormalized_predictions)
    extract_altman_plot(gt_all, pred_all)
    # extract altman plot for each sample
    for i in range(len(denormalized_ground_truths)):
        print(gt.shape, pred.shape)
        gt = denormalized_ground_truths[i]
        pred = denormalized_predictions[i]
        extract_altman_plot(gt, pred, title = "plot for subject " + str(ids_list[i]))
    for i in range(len(ids_list)):
        plot_3d_errors_comparison(denormalized_ground_truths[i], denormalized_predictions[i], title = "3D error comparison for subject " + str(ids_list[i]))
    return None
  

if __name__ == '__main__':    
    extract_pinn_out()