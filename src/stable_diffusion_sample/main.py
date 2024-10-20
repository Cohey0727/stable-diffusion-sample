import os
import uuid
from diffusers import StableDiffusionPipeline
import torch


class AvailableModels:
  stable_diffusion_v1_5 = "runwayml/stable-diffusion-v1-5"
  anything_v3 = "Linaqruf/anything-v3.0"


# モデルをロードし、MPSバックエンドに設定
pipe = StableDiffusionPipeline.from_pretrained(AvailableModels.stable_diffusion_v1_5, torch_dtype=torch.float16)
pipe = pipe.to("mps")

# 画像生成
prompt = "Landscape of a beautiful city at night, with a clear sky and a full moon, a beautiful cityscape with tall buildings and a clear sky with a full moon"
image = pipe(prompt).images[0]

# フォルダがなければ作成
os.makedirs("outputs", exist_ok=True)

# 乱数でファイル名を生成
filename = str(uuid.uuid4()) + ".png"
image.save(f"outputs/{filename}")
