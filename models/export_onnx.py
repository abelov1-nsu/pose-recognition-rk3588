import torch
from recognize import PoseSequenceGRU

checkpoint = torch.load("pose_sequence_model.pt", map_location="cpu")
classes      = checkpoint["classes"]
input_size   = checkpoint.get("input_size", 85)   # 85, confirmed
hidden_size  = checkpoint.get("hidden_size", 64)
num_layers   = checkpoint.get("num_layers", 1)

model = PoseSequenceGRU(input_size, hidden_size, len(classes), num_layers)
model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
model.eval()

# CONFIRMED from temporal_features.py:
#   SEQUENCE_LENGTH = 15
#   FEATURES_PER_FRAME = 17 * 5 = 85
dummy_input = torch.randn(1, 15, 85)

torch.onnx.export(
    model,
    dummy_input,
    "pose_sequence_gru.onnx",
    input_names=["pose_sequence"],
    output_names=["logits"],
    opset_version=17,
    dynamo=True,     # critical: legacy exporter keeps GRU math correct
    verify=True,
).save()