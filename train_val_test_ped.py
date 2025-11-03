if __name__ == '__main__':
    from ultralytics import YOLO
    from pathlib import Path

    ROOT      = Path("/home/omar.elalaily/pedestrian_traffic_light")        # server base
    MY_YAML   = ROOT / "weights"    / "ped_2classes.yaml"
    WEIGHTS   = ROOT / "weights"   / "yolov8.yaml"
    RUNS      = ROOT / "output_runs"
    WEIGHT    = ROOT / "output_runs" / "train" / "weights" / "best.pt"
    RUNS_fine = ROOT / "fine_tune_output_runs"

    model = YOLO(WEIGHTS)

    #Train on my custom dataset
    model.train(
                data=MY_YAML,
                epochs=100,
                imgsz=1280,
                batch=8,
                device=[0,1,2,3],
                lr0=1e-4,
                lrf=1e-5,
                cos_lr=True,        
                patience=20,
                project=RUNS,
                name="train")

    #Evaluate on VAL
    metrics_val = model.val(
        data=MY_YAML,
        split='val',
        imgsz=1280,
        device=[0,1,2,3],
        project=RUNS,
        name='val',
        verbose=True,
        plots=True
    )
    print(metrics_val)

    #Evaluate on Test
    metrics_test = model.val(
                        data=MY_YAML,
                        split='test',
                        device=[0,1,2,3],
                        project=RUNS,
                        name='test',
                        verbose=True,
                        plots=True)
    print(metrics_test)

    #Fine Tune
    if WEIGHT.exists():
        model_fine = YOLO(WEIGHT, task='detect')
        model_fine.train(
                        data=MY_YAML,
                        epochs=20,      
                        batch=8,
                        imgsz=1280,
                        lr0=1.3e-4,       
                        lrf=1.3e-5,
                        device=[0,1,2,3],
                        cos_lr=True,
                        patience=10,
                        project=RUNS_fine,
                        name="train_fine_tune")

        #Evaluate Fine Tune model on val
        metrics_fine_tune_val = model_fine.val(
                            data=MY_YAML,
                            split='val',
                            device=[0,1,2,3],
                            project=RUNS_fine,
                            name='val_fine_tune',
                            verbose=True,
                            plots=True)
        print(metrics_fine_tune_val)

        #Evaluate Fine Tune model on Test
        metrics_fine_tune_test = model_fine.val(
                            data=MY_YAML,
                            split='test',
                            device=[0,1,2,3],
                            project=RUNS_fine,
                            name='test_fine_tune',
                            verbose=True,
                            plots=True)
        print(metrics_fine_tune_test)
    else:
        print("best.pt not found yet; skipping fine-tune this run.")
