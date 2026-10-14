from rknn.api import RKNN

ONNX   = "/mnt/c/study/projects/EdgeDeviceGestureRecognition/gru_model.onnx"
DATASET = "/mnt/c/study/projects/EdgeDeviceGestureRecognition/dataset.txt"
OUT    = "/mnt/c/study/projects/EdgeDeviceGestureRecognition/pose_sequence_gru.rknn"

rknn = RKNN(verbose=True)

rknn.config(
    target_platform="rk3588",
    # DO NOT set mean_values / std_values:
    # temporal_features.py does no mean/std normalization.
    # The tensor is already in the model's expected range.
)

print("--> load_onnx")
assert rknn.load_onnx(model=ONNX) == 0

print("--> build (INT8)")
ret = rknn.build(do_quantization=True, dataset=DATASET)
if ret != 0:
    print("INT8 build failed; retrying FP16...")
    ret = rknn.build(do_quantization=False)
    assert ret == 0, "FP16 build also failed"

print("--> export_rknn")
assert rknn.export_rknn(OUT) == 0

rknn.release()
print("Done:", OUT)