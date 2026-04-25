import os
import argparse
import slicer
import vtk
import sys

MESH_UNIT_SCALE={
    "001": 1.0, #mm all others in cm
}
MESH_UNIT_SCALE_DEFAULT=10.0
PATIENTS_NO_MESH={"011", "113"}

def get_mesh_scale(patient_id):
    return MESH_UNIT_SCALE.get(patient_id, MESH_UNIT_SCALE_DEFAULT)

def apply_transform(mesh_node, matrix):
    t_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", "TempTransform")
    t_node.SetMatrixTransformToParent(matrix)
    mesh_node.SetAndObserveTransformNodeID(t_node.GetID())
    mesh_node.HardenTransform()
    slicer.mrmlScene.RemoveNode(t_node)

def normalize_mesh_to_Ct(mesh_node, ct_node, patient_id):
    def get_bounds(node):
        b=[0]*6
        node.GetBounds(b)
        return b
    
    def bounds_overlap(b1, b2):
        return (b1[0]<b2[1] and b1[1]>b2[0] and b1[2]<b2[3] and b1[3]>b2[2] and b1[4]<b2[5] and b1[5]>b2[4])
    
    def centroid(b):
        return [(b[i*2]+b[i*2+1])/2 for i in range(3)]
    ct_bounds=get_bounds(ct_node)
    #Scale
    scale=get_mesh_scale(patient_id)
    if scale!=1.0:
        print(f"[INFO] Scaling mesh by factor {scale} for patient {patient_id}")
        m=vtk.vtkMatrix4x4()
        m.Identity()
        m.SetElement(0, 0, scale)
        m.SetElement(1, 1, scale)
        m.SetElement(2, 2, scale)
        apply_transform(mesh_node, m)
    else:
        print(f"[INFO] No scaling applied for patient {patient_id}")
    mesh_bounds=get_bounds(mesh_node)
    print(f"[DEBUG] After scale - mesh bounds:"
          f"X[{mesh_bounds[0]:.1f}, {mesh_bounds[1]:.1f}] "
          f"Y[{mesh_bounds[2]:.1f}, {mesh_bounds[3]:.1f}] "
          f"Z[{mesh_bounds[4]:.1f}, {mesh_bounds[5]:.1f}]"
    )
    #Translate
    mc=centroid(mesh_bounds)
    cc=centroid(ct_bounds)
    tx, ty, tz=cc[0]-mc[0], cc[1]-mc[1], cc[2]-mc[2]
    print(f"[INFO] pz{patient_id}: translating [{tx:.1f}, {ty:.1f}, {tz:.1f}] mm")
    m=vtk.vtkMatrix4x4()
    m.Identity()
    m.SetElement(0, 3, tx)
    m.SetElement(1, 3, ty)
    m.SetElement(2, 3, tz)
    apply_transform(mesh_node, m)
    #Verification
    mesh_bounds=get_bounds(mesh_node)
    if bounds_overlap(mesh_bounds, ct_bounds):
        print(f"[INFO] pz{patient_id}: mesh aligned to CT"
              f"X[{mesh_bounds[0]:.1f}, {mesh_bounds[1]:.1f}] "
              f"Y[{mesh_bounds[2]:.1f}, {mesh_bounds[3]:.1f}] "
              f"Z[{mesh_bounds[4]:.1f}, {mesh_bounds[5]:.1f}]")
        return True
    else:
        print(f"[ERROR] pz{patient_id}: no CT overlap after scale+translate"
              f"Mesh: X[{mesh_bounds[0]:.1f}, {mesh_bounds[1]:.1f}]"
              f"Y[{mesh_bounds[2]:.1f}, {mesh_bounds[3]:.1f}] "
              f"Z[{mesh_bounds[4]:.1f}, {mesh_bounds[5]:.1f}] | "
              f"CT: X[{ct_bounds[0]:.1f}, {ct_bounds[1]:.1f}] "
              f"Y[{ct_bounds[2]:.1f}, {ct_bounds[3]:.1f}] "
              f"Z[{ct_bounds[4]:.1f}, {ct_bounds[5]:.1f}]")
        return False
    
def main():
    slicer.app.processEvents()
    #Parser
    launch_path=os.getcwd()
    parser=argparse.ArgumentParser(description="Batch Morphological Features Extraction for AAA in 3D Slicer")
    parser.add_argument("--patient_id", required=True, help="Patient ID")
    parser.add_argument("--db_path", default="/data/simulation_db", help="Path to the database")
    parser.add_argument("--cta_path", default="../cta", help="Path to CTA image folder (not users)")
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
    cta_path=os.path.abspath(os.path.join(launch_path, args.cta_path))
    db_path=os.path.abspath(os.path.join(launch_path, args.db_path))
    out_dir=os.path.abspath(os.path.join(launch_path, args.out_dir))

    if patient_id in PATIENTS_NO_MESH:
        print(f"[ERROR] Patient {patient_id} is known to have no mesh available. Skipping.")
        return

    #Temp partition
    custom_temp=os.path.join(out_dir, "slicer_temp")
    os.makedirs(custom_temp, exist_ok=True)
    slicer.app.temporaryPath = custom_temp
    print(f"[INFO] Temporary directory set to: {custom_temp}")

    #NEw SegmentGeometry
    script_dir=os.path.dirname(os.path.abspath(__file__))
    patched_sg=os.path.join(script_dir, "SegmentGeometryPatched.py")
    if os.path.isfile(patched_sg):
        import importlib, importlib.machinery, types
        loader=importlib.machinery.SourceFileLoader("SegmentGeometry", patched_sg)
        sg_module=types.ModuleType("SegmentGeometry")
        loader.exec_module(sg_module)
        sys.modules["SegmentGeometry"]=sg_module
        print(f"[INFO] Loaded patched SegmentGeometry from {patched_sg}")
    else:
        print(f"[WARN] Patched SegmentGeometry not found at {patched_sg}. Using default module.")

    import ExtractCenterline
    import SegmentGeometry

    #Path Verification
    input_dir=None
    path_variants = [
        f"{db_path}/pz{patient_id}/Meshes",
        f"{db_path}/pz_{patient_id}/Meshes",
        f"{db_path}/pz{patient_id}/Simulations/pz{patient_id}/mesh-complete",
        f"{db_path}/pz_{patient_id}/Simulations/pz{patient_id}/mesh-complete",
        f"{db_path}/pz{patient_id}/mesh-complete",
    ]
    for p in path_variants:
        print(p)
        if os.path.isdir(p):
            input_dir=p
            print(f"[INFO] Using input dir: {p}")
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
    ct_file = None
    for phase in ["pre_A", "pre_V"]:
        candidate = os.path.join(
            cta_path, f"pz{patient_id}", f"{patient_id}_0CT_{phase}.nrrd"
        )
        if os.path.isfile(candidate):
            ct_file = candidate
            print(f"[INFO] Using CT phase: {phase}")
            break
    if not ct_file:
        print(f"[ERROR] No CT volume found for pz{patient_id}")
        return
    ct_node = slicer.util.loadVolume(ct_file)
    if not ct_node:
        print(f"[ERROR] Failed to load CT for pz{patient_id}")
        return
    vtp_candidates = [
        os.path.join(input_dir, f"pz{patient_id}.vtp"),
        os.path.join(input_dir, "walls_combined.vtp"),
        os.path.join(input_dir, "mesh-complete.exterior.vtp"),
    ]
    vtp_path = None
    for candidate in vtp_candidates:
        if os.path.isfile(candidate):
            vtp_path = candidate
            print(f"[INFO] Using mesh: {os.path.basename(vtp_path)}")
            break
    if not vtp_path:
        print(f"[ERROR] No .vtp mesh found for pz{patient_id}")
        return

    surface_node = slicer.util.loadModel(vtp_path)
    if not surface_node:
        print(f"[ERROR] Failed to load mesh for pz{patient_id}")
        return
    
    slicer.util.setSliceViewerLayers(background=ct_node, fit=True)
    
    #display_node=ct_node.GetDisplayNode()
    #if display_node:
    #    display_node.AutoWindowLevelOn()
    #surface_node.HardenTransform()

    storage_node=surface_node.GetStorageNode()
    if storage_node:
        storage_node.SetCoordinateSystem(slicer.vtkMRMLStorageNode.LPS)
    surface_node.HardenTransform()

    #Normalize Mesh to CT
    if not normalize_mesh_to_Ct(surface_node, ct_node, patient_id):
        print(f"[ERROR] Failed to align mesh to CT for pz{patient_id}")
        return

    #Mesh -> Segmentation of surface_node
    seg_node=slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", f"pz{patient_id}-seg")
    if not seg_node:
        print("[ERROR] Impossible to create segmentation node")
        return
    seg_node.SetReferenceImageGeometryParameterFromVolumeNode(ct_node)
    slicer.modules.segmentations.logic().ImportModelToSegmentationNode(surface_node, seg_node)
    segmentation=seg_node.GetSegmentation()
    segmentation.SetSourceRepresentationName(slicer.vtkSegmentationConverter.GetSegmentationClosedSurfaceRepresentationName())
    conversion_ok=segmentation.CreateRepresentation(slicer.vtkSegmentationConverter.GetSegmentationBinaryLabelmapRepresentationName())
    if not conversion_ok:
        print("[ERROR] Closed surface -> binary labelmap conversion failed")
        return
    print("[INFO] Binary labelmap representation created successfully")
    
    #Export LabelMap & Cast to Scalar
    labelmap_node=slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", f"pz{patient_id}-labelmap")
    segment_ids=vtk.vtkStringArray()
    for i in range(segmentation.GetNumberOfSegments()):
        segment_ids.InsertNextValue(segmentation.GetNthSegmentID(i))
    success=slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(seg_node, segment_ids, labelmap_node, ct_node)
    if not success:
        print(f"[ERROR] Failed to export segmentation to labelmap for pz{patient_id}")
        return
    print("[INFO] Labelmap exported successfully")

    #scalar_vol=slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", f"pz{patient_id}-scalar")
    #parameters={"InputVolume": labelmap_node.GetID(), "OutputVolume": scalar_vol.GetID(), "type": "UnsignedChar"}
    #cli_node=slicer.cli.runSync(slicer.modules.castscalarvolume, None, parameters)
    #cast_ok=(cli_node is not None and scalar_vol.GetImageData() is not None and scalar_vol.GetImageData().GetNumberOfPoints()>0)
    #if not cast_ok:
    #    print("[WARN] Cast Scalar VOlume failed")
    #    scalar_vol=labelmap_node
   
    #Extract Centerline
    vmtk=ExtractCenterline.ExtractCenterlineLogic()
    centerline_curve=slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode", "CenterlineCurve")
    centerline_table=slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", "CenterlineTable")
    endpoints_node=slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "Endpoints")
    input_polydata=surface_node.GetPolyData()
    if input_polydata.GetNumberOfCells()>50000:
        cleaned_polydata=vmtk.preprocess(input_polydata, 50000, 4, 1)
    else:
        cleaned_polydata=input_polydata
    network_pd=vmtk.extractNetwork(cleaned_polydata, endpoints_node)
    endpoint_positions = vmtk.getEndPoints(network_pd, startPointPosition=None)
    endpoints_node.RemoveAllControlPoints()
    sorted_points=sorted(endpoint_positions, key=lambda p: p[2], reverse=True)
    if len(sorted_points) < 2:
        print("[WARN] Not enough endpoints found for centerline extraction")
        return
    source_point=sorted_points[0]
    target_points=sorted_points[1:]
    endpoints_node.AddControlPoint(vtk.vtkVector3d(source_point))
    for p in target_points:
        if p[2]<(source_point[2]-20):
            endpoints_node.AddControlPoint(vtk.vtkVector3d(p))
    centerline_pd, _=vmtk.extractCenterline(cleaned_polydata, endpoints_node, 1.0)
    vmtk.createCurveTreeFromCenterline(centerline_pd, centerline_curve, centerline_table, 1.0)
    slicer.util.saveNode(centerline_table, os.path.join(out_parent_dir, "centerline_data.csv"))

    #Segment Geometry
    seg_node.CreateClosedSurfaceRepresentation()
    slicer.app.processEvents()
    sg_logic=SegmentGeometry.SegmentGeometryLogic()
    if seg_node.GetSegmentation().GetNumberOfSegments()==0:
        print(f"[ERROR] No segments found in segmentation for pz{patient_id}")
        return
    segment_id=seg_node.GetSegmentation().GetNthSegmentID(0)
    axes = {
        "S (Red)": "S",
        "A (Green)": "A",
        "R (Yellow)": "R"    
    }
    
    for axis, label in axes.items():
        res_table=slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", f"parameters_{label}")
        plotChartNode=slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotChartNode", f"plot_{label}")
        print(f"--- Computing Geometry for axis: {label} ---")
        saved=False
        for interval in [1, 2, 5]:
            try:
                sg_logic.run(
                    seg_node, segment_id, ct_node, axis, interval, res_table, plotChartNode,
                    True, True, True, True, True, True, True, True, True, 
                    0, True, True, False, False, False, None, "", True, True, ""
                )
                slicer.util.saveNode(res_table, os.path.join(out_parent_dir, f"parameters_{label}.csv"))
                print(f"[SUCCESS] Saved parameters_{label}.csv")
                saved=True
                break
            except Exception as e:
                if res_table.GetNumberOfRows() > 0:
                    slicer.util.saveNode(res_table, os.path.join(out_parent_dir, f"parameters_{label}.csv"))
                    print(f"[OK] Dati estratti nonostante errore grafico per {label}")
                    saved=True
                    break
                else:
                    print(f"[WARN] Failed geometry for {label}: {e}")
                    print(f"[RETRY] interval={interval} for {label} failed")
        if not saved:
            print(f"[WARN] No data could be extracted for axis {label}")
    #Segment Statistics
    if seg_node and ct_node:
        try:
            import SegmentStatistics
    
            print("--- Computation Volume Statics ---")
            seg_node.SetAndObserveTransformNodeID(None)
            ss_logic=SegmentStatistics.SegmentStatisticsLogic()
            ss_logic.getParameterNode().SetParameter("Segmentation", seg_node.GetID())
            ss_logic.getParameterNode().SetParameter("ScalarVolume", ct_node.GetID())
            ss_logic.getParameterNode().SetParameter("visibleSegmentsOnly", "False")
            ss_logic.getParameterNode().SetParameter("LabelmapSegmentStatisticsPlugin.enabled", "True")
            ss_logic.getParameterNode().SetParameter("ScalarVolumeSegmentStatisticsPlugin.enabled", "True")
            ss_logic.getParameterNode().SetParameter("ClosedSurfaceSegmentStatisticsPlugin.enabled", "True")
            slicer.app.processEvents()
            ss_logic.computeStatistics()
            csv_path=os.path.join(out_parent_dir, "volume.csv")
            ss_logic.exportToCSVFile(csv_path)
            print("[SUCCESS] Saved volume.csv")
        except Exception as e:
            print(f"[ERROR] Failed to compute segment statistics: {e}")
    else:
        print(f"[ERROR] Seg_node or ct_node not loaded correctly")
    
    #Screenshots
    print("--- Generating Screenshots ---")
    try:
        lm = slicer.app.layoutManager()
        lm.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView)
        threeDView = lm.threeDWidget(0).threeDView()
        threeDView.resetFocalPoint()
        slicer.app.processEvents()
        renderWindow = threeDView.renderWindow()
        wti = vtk.vtkWindowToImageFilter()
        wti.SetInput(renderWindow)
        wti.Update()
        writer = vtk.vtkPNGWriter()
        shot_path = os.path.join(out_parent_dir, "aorta_3d_final.png")
        writer.SetFileName(shot_path)
        writer.SetInputConnection(wti.GetOutputPort())
        writer.Write()
        print(f"[SUCCESS] Screenshot saved in: {shot_path}")
    except Exception as e:
        print(f"[WARN] Screenshot failed: {e}")
    slicer.mrmlScene.Clear(0)

if __name__=="__main__":
    main()
    sys.exit()