expr_dir = 'output/' # where to store samples and models
PSEnet_path='/home/fatih/psenet_text_detector/weights/PSEnet_best.pth.tar.part'#path to trained detection model

# about data and net
keep_ratio = False # whether to keep ratio for image resize
manualSeed = 1234 # reproduce experiemnt
imgH = 32 # the height of the input image to network
imgW = 100 # the width of the input image to network
nh = 256 # size of the lstm hidden state
nc = 1
dealwith_lossnone = True # whether to replace all nan/inf in gradients to zero
# hardware
cuda = True # enables cuda
multi_gpu = False # whether to use multi gpu
ngpu = 1 # number of GPUs to use. Do remember to set multi_gpu to True!
workers = 0 # number of data loading workers

#---------------------------Parameters for PSEnet------------------------------------------------
arch='resnet50'#specify architecture
binary_th=1.0
kernel_num=7
scale=1
long_size=2240
min_kernel_area=5.0
min_area=800.0
min_score=0.93
