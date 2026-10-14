from rknn.api import RKNN

rknn = RKNN(verbose=True)

rknn.config(
    target_platform="rk3588",
    # Do NOT set mean_values / std_values here.
    # Your temporal_features.py performs no mean/std normalization,
    # and these fields are intended for image inputs anyway.
)

ret = rknn.load_onnx(model="pose_sequence_gru.onnx")
assert ret == 0, "load_onnx failed"

ret = rknn.build(
    do_quantization=True,
    dataset="dataset.txt",     # real (1,15,85) .npy files
)
assert ret == 0, "build failed"

ret = rknn.export_rknn("pose_sequence_gru_int8.rknn")
assert ret == 0, "export failed"

rknn.release()