from optimum.intel import OVModelForFeatureExtraction
from transformers import AutoTokenizer
from pathlib import Path

# 模型选型 [cite: 144]
MODEL_ID = "BAAI/bge-small-zh-v1.5"
EXPORT_PATH = "models/bge-small-ov"


def export_model():
    print(f"🚀 开始导出 OpenVINO 模型: {MODEL_ID}...")
    print("💡 Using Intel OpenVINO Model Optimizer for hardware acceleration")  # 关键注释 [cite: 149]

    # 加载并导出模型
    # export=True 会自动调用 OpenVINO Model Optimizer
    model = OVModelForFeatureExtraction.from_pretrained(MODEL_ID, export=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # 保存到本地目录
    model.save_pretrained(EXPORT_PATH)
    tokenizer.save_pretrained(EXPORT_PATH)

    print(f"🎉 模型已量化并保存至: {EXPORT_PATH}")


if __name__ == "__main__":
    export_model()
