from diffusers import StableDiffusionPipeline
import torch

# モデルをロードし、MPSバックエンドに設定
pipe = StableDiffusionPipeline.from_pretrained("Linaqruf/anything-v3.0", torch_dtype=torch.float16)
pipe = pipe.to("mps")

# 画像生成
prompt = "A fantasy landscape"
image = pipe(prompt).images[0]

# 画像を保存
image.save("output.png")
