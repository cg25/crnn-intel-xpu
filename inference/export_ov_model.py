import torch
from src.core.crnn import CRNN
import openvino as ov


def export_to_onnx():
    model = CRNN(img_h=32, num_classes=37)
    checkpoint = torch.load("../outputs/checkpoints/latest.pth", map_location="cpu")
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    dummy_input = torch.randn(1, 1, 32, 100)

    torch.onnx.export(
        model,
        dummy_input,
        "../outputs/crnn.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {3: "width"},
            "output": {0: "seq_len"}
        },
        opset_version=14
    )


def convert_to_openvino():
    core = ov.Core()

    model = core.read_model("../outputs/crnn.onnx")
    ov.save_model(model, "../outputs/crnn_ov.xml")
    core.compile_model(model, "GPU")
    print("OpenVINO model compiled successfully")


if __name__ == "__main__":
    export_to_onnx()
    convert_to_openvino()