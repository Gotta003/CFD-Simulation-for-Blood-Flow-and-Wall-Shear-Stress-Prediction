import os
import argparse
import slicer
import vtk
import sys

def main():
    slicer.app.processEvents()
    #Parser
    launch_path=os.getcwd()
    parser=argparse.ArgumentParser(description="Batch Morphological Features Extraction for AAA in 3D Slicer")
    parser.add_argument("--patient_id", required=True, help="Patient ID")
    parser.add_argument("--db_path", default="../simulation_db", help="Path to the database")
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

    #Temp partition
    custom_temp=os.path.join(out_dir, "slicer_temp")
    os.makedirs(custom_temp, exist_ok=True)
    slicer.app.temporaryPath = custom_temp
    print(f"[INFO] Temporary directory set to: {custom_temp}")

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
    cta_path=os.path.join(cta_path, f"pz{patient_id}", f"{patient_id}_0CT_pre_A.nrrd")
    vtp_path=os.path.join(input_dir, "mesh-complete.exterior.vtp")
    surface_node=slicer.util.loadModel(vtp_path)
    ct_node=slicer.util.loadVolume(cta_path)

    if not surface_node or not ct_node:
        print(f"[ERROR] Failed to load files for patient {patient_id}")
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