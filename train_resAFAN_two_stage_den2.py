import os
import torch
import numpy as np
import sys
from datetime import datetime
import cv2
from torch.optim.lr_scheduler import StepLR
from src import network
from src.data_loader import ImageDataLoader
from src.data_loader_resFAN import ExpoDataLoader
from src.timer import Timer
from src import utils
from src.evaluate_expo_resAFAN_two_stage import evaluate_expo_flow_model_roi_mul_mask


from src.crowd_count_resAFAN_first_stage import CrowdCounter1
from src.crowd_count_resAFAN_second_stage_flow import CrowdCounter2_flow
from src.crowd_count_resAFAN_den import CrowdCounter_den



try:
    from termcolor import cprint
except ImportError:
    cprint = None

try:
    from pycrayon import CrayonClient
except ImportError:
    CrayonClient = None


def log_print(text, color=None, on_color=None, attrs=None):
    if cprint is not None:
        cprint(text, color=color, on_color=on_color, attrs=attrs)
    else:
        print(text)
        

def readimg(input0,shape3,shape4,roi):
    read_img = cv2.imread(input0,0)
    read_img = read_img.astype(np.float32, copy=False)
    ht = read_img.shape[0]
    wd = read_img.shape[1]
    ht_1 = (ht / 4) * 4
    wd_1 = (wd / 4) * 4
    ht_1 = round(ht_1)
    wd_1 = round(wd_1)
    read_img = cv2.resize(read_img, (wd_1, ht_1))
    roi=roi.reshape((ht_1,wd_1))
    read_img=np.multiply(roi,read_img)
    read_img = cv2.resize(read_img, (shape4,shape3))
    read_img = read_img.reshape((1, 1, shape3, shape4))
    return read_img





method = 'resFAN' #method name - used for saving model file
dataset_name = 'expo' #dataset name - used for saving model file
output_dir = './saved_models_resFAN_den/' #model files are saved here

#train and validation paths
train_path = '../../demo/data/dataset_final/train_frame/'
train_gt_path = '../../demo/data/dataset_final/training/traingt'
val_path = '../../demo/data/dataset_final/train_frame/'
val_gt_path = '../../demo/data/dataset_final/training/valgt'
cflow_path='../../demo/data/dataset_final/train_cflow/'
train_gt_mask='../../demo/data/dataset_final/train_cmask_csv/'
val_gt_mask='../../demo/data/dataset_final/train_cmask_csv/'
#training configuration
start_step = 0
end_step = 2000
lr = 0.00001
momentum = 0.9
disp_interval = 500
log_interval = 250
optical_usage='single'# single combine attention augmentation
combination='concat'# concate multiply elementwise

#Tensorboard  config
use_tensorboard = False
save_exp_name = method + '_' + dataset_name + '_' + 'v1'
remove_all_log = False   # remove all historical experiments in TensorBoardO
exp_name = None # the previous experiment name in TensorBoard

#feature_net for feature extraction

rand_seed = 64678    
if rand_seed is not None:
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
    
#loadt training and validation data
#expo_loader_test=ExpoDataLoader('../../demo/data/dataset_final/train_frame/','../../demo/data/dataset_final/training/testgt',val_gt_mask,shuffle=True,gt_downsample=False,pre_load=False)
#expo_loader_fname=ExpoDataFnameLoader(train_path,train_gt_path,shuffle=True,gt_downsample=False,pre_load=False)
#expo_loader=expo_loader_test
expo_loader=ExpoDataLoader(train_path,train_gt_path,train_gt_mask,shuffle=True,gt_downsample=False,pre_load=False)
#data_loader = ImageDataLoader(train_path, train_gt_path, shuffle=True, gt_downsample=False, pre_load=True)
#class_wts = data_loader.get_classifier_weights()
expo_loader_val=ExpoDataLoader(val_path, val_gt_path,val_gt_mask, shuffle=False, gt_downsample=False, pre_load=False)
#data_loader_val = ImageDataLoader(val_path, val_gt_path, shuffle=False, gt_downsample=False, pre_load=True)

#load net and initialized on creation



frozen_net=CrowdCounter1()
frozen_net.cuda()
frozen_net.eval()

flow_net=CrowdCounter2_flow()
network.load_net('./Best_models/resFAN_flow/resFAN_flow_40.h5',flow_net)
flow_net.cuda()
flow_net.eval()


task_net=CrowdCounter_den()
task_net.cuda()
task_net.train()

#net = CrowdCounter(ce_weights=class_wts)
#network.weights_normal_init(net, dev=0.01)
#net.cuda()
#net.train()

params = list(task_net.parameters())
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, task_net.parameters()), lr=lr,weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=1, gamma=0.995)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# tensorboad
use_tensorboard = use_tensorboard and CrayonClient is not None
if use_tensorboard:
    cc = CrayonClient(hostname='127.0.0.1')
    if remove_all_log:
        cc.remove_all_experiments()
    if exp_name is None:
        exp_name = datetime.now().strftime('vgg16_%m-%d_%H-%M')
        exp_name = save_exp_name 
        exp = cc.create_experiment(exp_name)
    else:
        exp = cc.open_experiment(exp_name)

# training
train_loss = 0
step_cnt = 0
re_cnt = False
t = Timer()
t.tic()

best_mae = sys.maxsize

for epoch in range(start_step, end_step+1):
    if epoch > -1:
        scheduler.step()
    step = -1
    train_loss = 0
    for blob in expo_loader:
        step = step + 1        
        im_data = blob['data']
        gt_data = blob['gt_density']
        gt_class_label = blob['gt_class_label']
        fname=blob['fname']
        roi=blob['roi']
        gt_mask=blob['gt_mask']

        copy_of_fname = fname
        split_results_of_fname = copy_of_fname.split('_')
        possible_folder_name = split_results_of_fname[0]
        folder_name = possible_folder_name.split('-')[0]
        flow_name = os.path.splitext(fname)[0] + '_flow.png'
        flow_reading_path = os.path.join(cflow_path, folder_name, flow_name)
        flow = cv2.imread(flow_reading_path,0)
        flow = flow.astype(np.float32, copy=False)
        flow = cv2.resize(flow,(im_data.shape[3],im_data.shape[2]))
        flow = flow.reshape((1, 1, flow.shape[0], flow.shape[1]))



        #data augmentation on the fly
        if np.random.uniform() > 0.5:
            #randomly flip input image and density 
            im_data = np.flip(im_data,3).copy()
            roi=np.flip(roi,3).copy()
            gt_data = np.flip(gt_data,3).copy()
            gt_mask=np.flip(gt_mask,3).copy()
            flow=np.flip(flow,3).copy()
        if np.random.uniform() > 0.5:
            #add random noise to the input image
            im_data = im_data + np.random.uniform(-10,10,size=im_data.shape)
        ######feature_net extract features
        im_data[0][0]=np.multiply(roi[0][0],im_data[0][0])
        im_data[0][1]=np.multiply(roi[0][0],im_data[0][1])
        im_data[0][2]=np.multiply(roi[0][0],im_data[0][2])
        #feature_map=feature_net(roi_im_data)
        roi_im_data=im_data

        ######combine flow with feature directly
        #read flow
        feature_map=roi_im_data

        roi_im_data=network.np_to_variable(roi_im_data, is_cuda=False, is_training=False)
        #feature_mapy=network.np_to_variable(feature_mapy, is_cuda=True, is_training=True)
        roi_flow=np.multiply(roi,flow)
        roi_flow_map=torch.from_numpy(roi_flow)
        feature_map_flow=torch.cat((roi_im_data,roi_flow_map),1)
        feature_map_flow=feature_map_flow.cuda()

        #read combined mask
        x,f=frozen_net(feature_map_flow,gt_data,gt_mask)
        in_data=torch.cat((x,f),1)
        mask_info, mask = flow_net(in_data, gt_mask)
        #roi_mask=np.multiply(mask_info,roi)

        roi_im_data=roi_im_data.cuda()
        in_data=torch.cat((roi_im_data,mask),1)
        density_map=task_net(in_data,gt_data)
        loss = task_net.loss
        train_loss += loss.item()
        step_cnt += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        density_map=density_map/100
        if step % disp_interval == 0:            
            duration = t.toc(average=False)
            fps = step_cnt / duration
            gt_count = np.sum(gt_data)
            density_map = density_map.data.cpu().numpy()
            roi_den=np.multiply(density_map,roi)
            #et_count = np.sum(density_map)
            et_count=np.sum(roi_den)
            utils.save_results_2(im_data,gt_data,density_map, mask,output_dir,'{}_{}_{}.png'.format(method,epoch,step))
            log_text = 'epoch: %4d, step %4d, Time: %.4fs, gt_cnt: %4.1f, et_cnt: %4.1f  loss: %4.10f' % (epoch,
                step, 1./fps, gt_count,et_count,loss.item())
            log_print(log_text, color='green', attrs=['bold'])
            re_cnt = True
        if re_cnt:                                
            t.tic()
            re_cnt = False

    if (epoch % 2 == 0):
        save_name = os.path.join(output_dir, '{}_{}_{}.h5'.format(method,dataset_name,epoch))
        network.save_net(save_name, task_net)
        #calculate error on the validation dataset
        
    

