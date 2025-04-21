from comet import download_model, load_from_checkpoint
import json
import huggingface_hub
import os


data = []
item1 = {
            "src": "My dog is cute.",
            "mt": "내."
        }
item2 = {
            "src": "My dog is cute.",
            "mt": "내 강아지는 귀여워."
        }
data.append(item1)
data.append(item2)

# download comet kiwi model
# log in to huggingface
with open("hf_token.txt") as f:
    token = f.readline().strip()
huggingface_hub.login(token)

model_path = download_model("Unbabel/wmt22-cometkiwi-da")
model = load_from_checkpoint(model_path)

model_output = model.predict(data, batch_size=16, gpus=1)
print(model_output)