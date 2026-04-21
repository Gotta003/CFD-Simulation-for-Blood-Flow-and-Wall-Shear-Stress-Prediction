import os 
import glob
import vtk
import SimpleITK as sitk
import pandas as pd

def get_mesh_info(vtp_path):
    if not os.path.exists(vtp_path):
        return None
    reader=vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtp_path)
    reader.Update()
    pd_data=reader.GetOutput()
    b=pd_data.GetBounds()
    c=pd_data.GetCenter()
    return {
        "Mesh_X_Min": b[0], "Mesh_X_Max": b[1],
        "Mesh_Y_Min": b[2], "Mesh_Y_Max": b[3],
        "Mesh_Z_Min": b[4], "Mesh_Z_Max": b[5],
        "Mesh_CX": c[0], "Mesh_CY": c[1], "Mesh_CZ": c[2],
        "Points": pd_data.GetNumberOfPoints()
    }

def get_ct_info(nrrd_path):
    if not os.path.exists(nrrd_path):
        return None
    img=sitk.ReadImage(nrrd_path)
    origin=img.GetOrigin()
    size=img.GetSize()
    spacing=img.GetSpacing()
    direction=img.GetDirection()
    return {
        "CT_Origin_X": origin[0], "CT_Origin_Y": origin[1], "CT_Origin_Z": origin[2],
        "Size_X": size[0], "Size_Y": size[1], "Size_Z": size[2],
        "Spacing_X": spacing[0], "Spacing_Y": spacing[1], "Spacing_Z": spacing[2],
        "Direction": str([round(d,2) for d in direction])
    }

def main():
    db_path="../simulation_db"
    cta_path="../cta"
    results=[]
    patients=[d for d in os.listdir(db_path) if d.startswith("pz")]
    patients.sort()
    for p_folder in patients:
        p_id=p_folder.replace('pz', '')
        print(f"Auditing Patient {p_id}...")
        vtp_file=os.path.join(db_path, p_folder, "Meshes", f"{p_folder}.vtp")
        if not os.path.exists(vtp_file):
            vtp_file=os.path.join(db_path, p_folder, "Simulations", p_folder, "mesh-complete", "mesh-complete.exterior.vtp")
        nrrd_file=os.path.join(cta_path, p_folder, f"{p_id}_0CT_pre_A.nrrd")
        row={"Patient_ID": p_id}
        try:
            mesh_data=get_mesh_info(vtp_file)
            if mesh_data:
                row.update(mesh_data)
            ct_data=get_ct_info(nrrd_file)
            if ct_data:
                row.update(ct_data)
            if mesh_data and ct_data:
                dist=((row["Mesh_CX"]-row["CT_Origin_X"])**2+
                      (row["Mesh_CY"]-row["CT_Origin_Y"])**2+
                      (row["Mesh_CZ"]-row["CT_Origin_Z"])**2)**0.5
                row["Center_Origin_Dist"]=round(dist, 2)
        except Exception as e:
            row["Error"]=str(e)
            print(f"  [ERROR] Patient {p_id}: {e}")
        results.append(row)
    df=pd.DataFrame(results)
    df.to_csv("outputs/dataset/aligment_audit.csv", index=False)

if __name__ == "__main__":
    main()