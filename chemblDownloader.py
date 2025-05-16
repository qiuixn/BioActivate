import chembl_downloader
import os
from utils.util import getPath

# 指定保存数据库文件的路径
save_path = getPath() + 'db/chembl.db'


# 使用chembl_downloader下载并解压SQLite数据库
try:
    path = chembl_downloader.download_extract_sqlite()
    print(path)
    # 检查下载的数据库文件是否存在
    if os.path.exists(path):
        # 将数据库文件复制到指定的保存路径
        os.replace(path, save_path)
        print(f"ChEMBL database has been saved to {save_path}")
    else:
        print("Failed to download or extract the ChEMBL database.")
except Exception as e:
    print(f"An error occurred: {e}")

