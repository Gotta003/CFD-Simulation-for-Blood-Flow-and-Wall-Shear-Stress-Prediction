import argparse

def main():
    parser=argparse.ArgumentParser(description="Train EVAR risk predictor")
    parser.add_argument("--model", default="gnn_pinn", choices=["gnn_pinn"])
    parser.add_argument("--features", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--vtp-dir", default="data/vtp_files/", help="Spatial Features Directory")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--out", default="outputs/predictions/model.pkl")
    parser.add_argument("--device", default="auto")
    args=parser.parse_args()
    
    #GNN-PINN MODEL
    if args.model=="gnn_pinn":
        from src.models.gnn_pinn import (GNNPINNConfig, graph_dataset_composition, training_gnnpinn)
        cfg=GNNPINNConfig()
        print("Graph Dataset Composition...")
        dataset=graph_dataset_composition()
        if not dataset:
            print("No valid patients.")
            return 
        out=args.out.replace(".pkl", ".pt")
        training_gnnpinn()
        return
    #OTHER MODELS DOWN BELOW
    
if __name__=="__main__":
    main()