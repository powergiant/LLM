import os
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
with open(project_dir + "/cache_folder.txt", "r") as file:
    cache_dir = file.read()
download_dir = os.path.join(cache_dir, "wanjuan")
if not os.path.exists(download_dir):
    os.mkdir(download_dir)

import openxlab
# go https://opendatalab.org.cn/OpenDataLab/WanJuan1_dot_0 to apply ak and sk
openxlab.login(ak="***", sk="***")
from openxlab.dataset import download
download(dataset_repo='OpenDataLab/WanJuan1_dot_0',source_path='/raw/nlp', target_path=download_dir)
