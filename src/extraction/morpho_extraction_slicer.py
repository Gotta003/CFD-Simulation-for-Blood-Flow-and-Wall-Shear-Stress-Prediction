import os
import argparse
import slicer
import vtk
import sys
import ExtractCenterline
import SegmentGeometry
import SegmentStatistics

def main():
    #Parser
    launch_path=os.getcwd()
    parser=argparse.ArgumentParser(description="Batch Morphological Features Extraction for AAA in 3D Slicer")
    parser.add_argument("--patient_id", required=True, help="Patient ID")
    parser.add_argument("--db_path", default="../simulation_db", help="Path to the database")
    parser.add_argument("--out_dir", default="data/morpho", help="Output directory")
    args = parser.parse_args()
    try:
        if "--python-script" in sys.argv:
            args_to_parse=sys.argv.index("--python-script")+2
        else:
            args_to_parse=sys.argv[1:]
        args = parser.parse_args(args_to_parse)
    except Exception as e:
        print(f"[ERROR] Argument parsing failed: {e}")
        return
    
    patient_id=args.patient_id
    db_path=os.path.abspath(os.path.join(launch_path, args.db_path))
    out_dir=os.path.abspath(os.path.join(launch_path, args.out_dir))
    
    #Path Verification
    input_dir=None
    path_variants=[
        f"{db_path}/pz{patient_id}/Simulations/pz{patient_id}/mesh-complete",
        f"{db_path}/pz_{patient_id}/Simulations/pz{patient_id}/mesh-complete",
        f"{db_path}/pz{patient_id}/mesh-complete",
        f"{db_path}/pz_{patient_id}/mesh-complete",
        f"{db_path}/pz{patient_id}/pz{patient_id}-mesh-complete"
    ]
    for p in path_variants:
        print(p)
        if os.path.isdir(p):
            input_dir=p
            break
    if not input_dir:
        print(f"[ERROR] No valid input directory found for patient {patient_id}")
        return  
    out_parent_dir=os.path.join(out_dir, f"pz{patient_id}")
    if not os.path.exists(out_parent_dir):
        os.makedirs(out_parent_dir)
    print(out_parent_dir)
    print(f"--- Processing patient {patient_id} ---")
    
    #File Loading
    vtp_path=os.path.join(input_dir, "mesh-complete.exterior.vtp")
    vtu_path=os.path.join(input_dir, "mesh-complete.mesh.vtu")
    surface_node=slicer.util.loadModel(vtp_path)
    mesh_node=slicer.util.loadModel(vtu_path)
    if not surface_node or not mesh_node:
        print(f"[ERROR] Failed to load files for patient {patient_id}")
        return
    
    #Mesh -> Segmentation of surface_node
    seg_node=slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", f"pz{patient_id}-seg")
    slicer.modules.segmentations.logic().ImportModelToSegmentationNode(surface_node, seg_node)

    #Export LabelMap & Cast to Scalar
    labelmap_node=slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", f"pz{patient_id}-labelmap")
    slicer.modules.segmentations.logic().ExportVisibleSegmentsToLabelmapNode(seg_node, labelmap_node)
    scalar_vol=slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", f"pz{patient_id}-scalar")
    parameters={"InputVolume": labelmap_node.GetID(), "OutputVolume": scalar_vol.GetID()}
    slicer.cli.runSync(slicer.modules.castscalarvolume, None, parameters)

    #Extract Centerline
    vmtk=ExtractCenterline.ExtractCenterlineLogic()
    centerline_curve=slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode", "CenterlineCurve")
    centerline_table=slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", "CenterlineTable")
    endpoints_node=slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "Endpoints")
    input_polydata=surface_node.GetPolyData()
    cleaned_polydata=vmtk.preprocess(input_polydata, 5000, 4, 1)
    network_pd=vmtk.extractNetwork(cleaned_polydata, endpoints_node)
    endpoint_positions=vmtk.getEndPoints(network_pd, startPointPosition=None)
    endpoints_node.RemoveAllControlPoints()
    for p in endpoint_positions:
        endpoints_node.AddControlPoint(vtk.vtkVector3d(p))
    if endpoints_node.GetNumberOfControlPoints()>0:
        endpoints_node.SetNthControlPointSelected(0, False)
    centerline_pd, _=vmtk.extractCenterline(cleaned_polydata, endpoints_node, 1.0)
    vmtk.createCurveTreeFromCenterline(centerline_pd, centerline_curve, centerline_table, 1.0)
    slicer.util.saveNode(centerline_table, os.path.join(out_parent_dir, "centerline_data.csv"))

    #Segment Geometry
    sg_logic=SegmentGeometry.SegmentGeometryLogic()
    view_map={"S": 0, "A": 1, "L": 2}
    for label, v_idx in view_map.items():
        res_table=slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", f"parameters_{label}")
        sg_logic.run(
            seg_node, scalar_vol, res_table, None, v_idx, 1,
            True, True, True, True, True, True, True, True, True, 
            0, True, True, True, True, True, None, "", True, True, "Results"
        )
        slicer.util.saveNode(res_table, os.path.join(out_parent_dir, f"parameters_{label}.csv"))

    #Segment Statistics
    ss_logic=SegmentStatistics.SegmentStatisticsLogic()
    ss_table=slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", "volume")
    ss_logic.getParameterNode().SetNodeReferenceID("Segmentation", seg_node.GetID())
    ss_logic.getParameterNode().SetNodeReferenceID("ScalarVolume", scalar_vol.GetID())
    ss_logic.getParameterNode().SetNodeReferenceID("DestinationTable", ss_table.GetID())
    ss_logic.computeStatistics()
    slicer.util.saveNode(ss_table, os.path.join(out_parent_dir, "volume.csv"))

    #Screenshots
    lm=slicer.app.layoutManager()
    if lm:
        threeDview=lm.threeDWidget(0).threeDView()
        threeDview.resetFocalPoint()
        surface_node.SetDisplayVisibility(True)
        centerline_curve.SetDisplayVisibility(True)
        slicer.util.forceRenderAllViews()
        slicer.util.screenshotCurrentWidget(threeDview, os.path.join(out_parent_dir, "aorta_surface_centerline.png"))
        surface_node.SetDisplayVisibility(False)
        slicer.util.forceRenderAllViews()
        slicer.util.screenshotCurrentWidget(threeDview, os.path.join(out_parent_dir, "aorta_centerline_only.png"))
        print(f"--- Finished pz{patient_id} ---\n")
        slicer.mrmlScene.Clear(0)
    else:
        print("[WARN] Layout manager not found, skipping screenshots")
    print(f"--- Finished processing patient {patient_id} ---\n")
    slicer.mrmlScene.Clear(0)

if __name__=="__main__":
    main()
    sys.exit()