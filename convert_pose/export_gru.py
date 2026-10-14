import torch
import torch.nn as nn

class PoseSequenceGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )
    def forward(self, x):
        out, _ = self.gru(x)
        return self.classifier(out[:, -1, :])

CKPT = "/mnt/c/study/projects/EdgeDeviceGestureRecognition/pose_sequence_model.pt"
OUT  = "/mnt/c/study/projects/EdgeDeviceGestureRecognition/gru_model.onnx"

ckpt = torch.load(CKPT, map_location="cpu")
model = PoseSequenceGRU(
    ckpt["input_size"],
    ckpt["hidden_size"],
    len(ckpt["classes"]),
    ckpt["num_layers"],
)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

dummy = torch.randn(1, 15, 85)   # (batch, SEQUENCE_LENGTH, FEATURES_PER_FRAME)

torch.onnx.export(
    model,
    dummy,
    OUT,
    input_names=["pose_sequence"],
    output_names=["logits"],
    opset_version=17,
    dynamo=False,   # force legacy exporter; keeps GRU math correct
    verify=True,
)

print("Exported:", OUT)
print("Classes:", ckpt["classes"])