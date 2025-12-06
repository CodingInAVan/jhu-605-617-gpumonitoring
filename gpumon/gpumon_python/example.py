import gpumon_py as gpumon
import torch
import time

# 1. Init
gpumon.init(app_name="PyTorch_Training", log_path="gpu_train.log", sample_interval_ms=100)

# 2. Use Scopes
try:
    print("Loading Model...")
    with gpumon.Scope("ModelLoad"):
        model = torch.resnet50().cuda()
        data = torch.randn(128, 3, 224, 224).cuda()

    print("Starting Training...")
    # Tagged Scope
    with gpumon.Scope("TrainingLoop", tag="epoch_1"):
        for i in range(10):
            # Nested Scope
            with gpumon.Scope("ForwardPass"):
                output = model(data)

            with gpumon.Scope("BackwardPass"):
                loss = output.sum()
                loss.backward()

            # Simulate work
            time.sleep(0.1)

finally:
    # 3. Shutdown
    gpumon.shutdown()