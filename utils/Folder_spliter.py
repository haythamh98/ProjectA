import os

def split_folder_to_subfolders(path:str,new_folders_name:str = "dir", max_size:int = 10000):
    cur_folder_idx = 0
    cur_folder_files_count = 0
    cur_folder_name = f"{new_folders_name}_{cur_folder_idx}"
    os.mkdir(os.path.join(path,cur_folder_name))

    for filename in iter(os.scandir(path)):
        filename = filename.name
        if new_folders_name in filename:
            continue
        if cur_folder_files_count >= max_size:
            cur_folder_idx += 1
            cur_folder_files_count = 0
            cur_folder_name = f"{new_folders_name}_{cur_folder_idx}"
            os.mkdir(os.path.join(path,cur_folder_name))


        os.system(f"mv {os.path.join(path,filename)} {os.path.join(path,cur_folder_name, filename)} ")
        cur_folder_files_count += 1


