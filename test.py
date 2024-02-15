#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('modelscope/Llama-2-70b-ms', cache_dir='/data_share/model_hub/llama/Llama-2-70b-hf')