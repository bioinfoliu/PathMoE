import urllib.request
import shutil
import tarfile
import os

url = "https://cbioportal-datahub.s3.amazonaws.com/brca_metabric.tar.gz"
download_path = "brca_metabric.tar.gz"
extract_dir = "./METABRIC_Data"

print("开始下载 METABRIC 数据集 (约 130MB)...")

# 1. 构造带有“浏览器伪装”的请求头
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}
req = urllib.request.Request(url, headers=headers)

try:
    # 2. 发起请求并流式写入本地文件（防止占用过多内存）
    with urllib.request.urlopen(req) as response, open(download_path, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
    print("下载完成！开始解压...")

    # 3. 解压文件
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)
        
    with tarfile.open(download_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)

    print(f"解压完毕！数据已保存在: {extract_dir}/brca_metabric")

except Exception as e:
    print(f"发生错误: {e}")