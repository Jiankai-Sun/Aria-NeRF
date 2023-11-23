# process existing videos/images for colmap
ns-process-data images --data /media/jqiu/2TB_SSD/Datasets/DA-cleanup/01/nerfstudio/images/ --output-dir /media/jqiu/2TB_SSD/Datasets/DA-cleanup/01/nerfstudio_v3
python -m colmap_converter --colmap_dir /media/jqiu/2TB_SSD/Datasets/DA-cleanup/01/nerfstudio_v3/colmap/ --scale=8  # save to data/custom/colmap/images