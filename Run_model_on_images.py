#####################
#Inputs to be edited
con_enc_dir ='/home/awd275/context_encoder_pytorch/'
path_to_model_chkpt = con_enc_dir + 'models/model_celebA_100k_100iter_04212020.pth'
input_image_dir = '/scratch/awd275/img_align_celebA/'
output_dir = '/scratch/awd275/context_encoder_pytorch_output/20200512_output/'
#####################
import os

input_images_path_list = []
#take 10000 images starting from 120000 (this is because training set are images numbered from 000000 to 100000
for ii in range(1000):
    img_number = ii+120000
    file_name = str(img_number)+'.jpg'
    input_images_path_list.append(input_image_dir+file_name)
#for file in os.listdir(input_image_dir):
#    if file.endswith(".jpg"):
#        input_images_path_list.append(file)
        
        
if os.path.isdir(output_dir):
    pass
else:
    os.mkdir(output_dir)
        
for image_file_path in input_images_path_list:
    output_prefix = image_file_path[:-4]
    command = '/home/awd275/miniconda3/envs/Inpainting/bin/python /home/awd275/context_encoder_pytorch/test_one.py ' + \
              '--test_image='+input_image_dir+image_file_path + ' ' + \
              '--output_directory='+output_dir + ' ' +\
              '--output_name_prefix='+ output_prefix + ' ' + \
              '--netG='+path_to_model_chkpt
             
    print(command)
    os.system(command)

