#########################################################
#				截图相关									#
#########################################################


resolution_for_capture_image = 0.25

im_capture__left_top_right_bottom = [0, 0, 1920, 1020]

#########################################################
#				训练相关									#
#########################################################
use_gpu = True
torch_train_epoch = 10

# 分类数目，从'{0}_xxx'开始的文件夹计数到'{n}_xxx'的分类模型
classification_count = 2

# 模型文件保存路径
train_save_path = 'model/test'
