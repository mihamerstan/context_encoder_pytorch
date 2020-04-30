#####################
#Inputs to be edited
path_to_model_chkpt='context_encoder_pytorch/model/model_celebA_100k_100iter_04212020.pth'
input_image_dir='context_encoder_pytorch/dataset/train/train/'
output_dir='celebA_output/'
#####################
import os
input_images_path_list = []
for file in os.listdir(input_image_dir):
    if file.endswith(".jpg"):
        input_images_path_list.append(file)
if os.path.isdir(output_dir):
    pass
else:
    os.mkdir(output_dir)
        
for image_file_path in input_images_path_list:
    output_prefix = image_file_path[:-4]
    command = 'python context_encoder_pytorch/test_one.py ' + \
              '--test_image='+input_image_dir+image_file_path + ' ' + \
              '--output_directory='+output_dir + ' ' +\
              '--output_name_prefix='+ output_prefix + ' ' + \
              '--netG='+path_to_model_chkpt
             
    print(command)
    os.system(command)

