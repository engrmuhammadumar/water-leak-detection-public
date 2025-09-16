# export_onnx.py
import torch
import config
from models import CNNLSTM, CNNLSTM_Fusion

def main():
    ckpt = torch.load("cnn_lstm_cwt.pt", map_location="cpu")

    if config.USE_PHYSICS_FEATURES:
        model = CNNLSTM_Fusion(num_classes=len(config.CLASSES), physics_dim=config.NUM_PHYSICS_FEATURES)
    else:
        model = CNNLSTM(num_classes=len(config.CLASSES))
    model.load_state_dict(ckpt["model"])
    model.eval()

    dummy_seq = torch.randn(1, config.SEQ_LEN, 1, config.IMG_SIZE, config.IMG_SIZE)
    if config.USE_PHYSICS_FEATURES:
        dummy_phys = torch.randn(1, config.NUM_PHYSICS_FEATURES)
        torch.onnx.export(
            model, (dummy_seq, dummy_phys), "cnn_lstm_cwt.onnx",
            input_names=["input_seq", "physics_feats"],
            output_names=["logits"], opset_version=13, dynamic_axes=None
        )
    else:
        torch.onnx.export(
            model, dummy_seq, "cnn_lstm_cwt.onnx",
            input_names=["input_seq"], output_names=["logits"], opset_version=13
        )
    print("Exported model to cnn_lstm_cwt.onnx")

if __name__ == "__main__":
    main()
