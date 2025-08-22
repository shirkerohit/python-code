import logging
import torch
from transformers import pipeline
from transformers.utils import logging as hf_logging
import time

# Silence Transformers logging
hf_logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)


print('loading model...', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    device="cpu",
    torch_dtype=torch.float32,
)

print('model loaded', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
result = pipe(
    "harvard.wav",
    generate_kwargs={
        "language": "en",
        "task": "transcribe",
    }
)
print('result:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
print(result["text"])