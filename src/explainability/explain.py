import argparse
import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap

def main():
    parser=argparse.ArgumentParser(description="SHAP explainability for EVAR model")
    parser.add_argument("--model", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out", default="outputs/shap_plots/")
    parser.add_argument("--top_n", type=int, default=3, help="Number of individual patient waterfall plots")
    args=parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    
    #Loading
    composition=joblib.load(args.model)
    model=composition["model"]
    feat_names=composition["features"]
    X_df=pd.read_csv(args.features)
    y_df=pd.read_csv(args.labels)
    df=X_df.merge(y_df, on="patient_id", how="inner")
    label_col="outcome" if "outcome" in df.columns else "label"
    X=df[feat_names].values
    y=df[label_col].values
    
    # SHAP
    clf=model.named_steps["clf"]
    pipe_transform=model[:-1]
    X_transformed=pipe_transform.transform(X)
    explainer=shap.TreeExplainer(clf)
    shap_values=explainer.shap_values(X_transformed)
    if isinstance(shap_values, list):
        sv=shap_values[1]
    else:
        sv=shap_values
    mean_abs=np.abs(sv).mean(axis=0)
    order=np.argsort(mean_abs)[::-1][:20]
    fig, ax=plt.subplots(figsize=(10,6))
    ax.barh(range(len(order)), mean_abs[order][::-1], color="#2E86AB")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([feat_names[i] for i in order[::-1]], fontsize=9)
    ax.set_xlabel("Mean SHAP value")
    ax.set_title("Global Feature Importance (SHAP)")
    plt.tight_layout()
    fig.savefig(os.path.join(args.out, "global_importance.png"), dpi=150)
    plt.close(fig)
    print("Saved: global_importance.png")
    
    #Beeswarm
    shap.summary_plot(sv, X_transformed, feature_names=feat_names, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "shap_beeswarm.png"), dpi=150)
    plt.close()
    print("Saved: shap_beeswarm.png")
    
    #Waterfall Per-Patient (Highest Predicted Risk)
    probs=model.predict_proba(X)[:,1]
    top_idx=np.argsort(probs)[::-1][:args.top_n]
    for rank, idx in enumerate(top_idx):
        patient_id=df["patient_id"].iloc[idx]
        shap.waterfall_plot(
            shap.Explanation(
                values=sv[idx],
                base_values=explainer.expected_value if not isinstance(explainer.expected_value, list) else explainer.expected_value[1]
                data=X_transformed[idx],
                feature_names=feat_names
            ),
            show=False,
            max_display=15
        )
        fname=f"waterfall_rank{rank+1}_{patient_id}.png"
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, fname), dpi=150)
        plt.close()
        print(f"Saved: {fname} (pred_risk={probs[idx]:.3f}, true={y[idx]})")
    print(f"\nAll SHAP plots saved to: {args.out}")
    
if __name__=="__main__":
    main()