import cv2
import numpy as np
import os

def save_results(input_img, gt_data,density_map,output_dir, fname='results.png'):
    input_img = input_img[0][0]
    gt_data = 255*gt_data/np.max(gt_data)
    density_map = 255*density_map/np.max(density_map)
    gt_data = gt_data[0][0]
    density_map= density_map[0][0]
    if density_map.shape[1] != input_img.shape[1]:
        density_map = cv2.resize(density_map, (input_img.shape[1],input_img.shape[0]))
        gt_data = cv2.resize(gt_data, (input_img.shape[1],input_img.shape[0]))
    result_img = np.hstack((input_img,gt_data,density_map))
    cv2.imwrite(os.path.join(output_dir,fname),result_img)




def save_results_2(input_img, gt_data,density_map,mask,output_dir, fname='results.png'):
    input_img0 = input_img[0][0]
    input_img1 = input_img[0][1]
    input_img2 = input_img[0][2]
    gt_data = 255*gt_data/np.max(gt_data)
    density_map = 255*density_map/np.max(density_map)
    mask=mask.data.cpu().numpy()
    mask = 255*mask/np.max(mask)
    gt_data = gt_data[0][0]
    density_map= density_map[0][0]
    mask=mask[0][0]
    if density_map.shape[1] != input_img.shape[1]:
        density_map = cv2.resize(density_map, (input_img0.shape[1],input_img0.shape[0]))
        gt_data = cv2.resize(gt_data, (input_img0.shape[1],input_img0.shape[0]))
        mask=cv2.resize(mask,(input_img0.shape[1],input_img0.shape[0]))
    result_img = np.hstack((input_img0,input_img1,input_img2,gt_data,density_map,mask))
    cv2.imwrite(os.path.join(output_dir,fname),result_img)

def save_results_all(input_img, gt_data,density_map,mask,flow_x,flow_y,flow,x_base_all,f_base_all,v_base_all,c_flow,output_dir, fname):
    input_img = input_img[0][0]
    gt_data = 255*gt_data/np.max(gt_data)
    density_map = 255*density_map/np.max(density_map)
    gt_data = gt_data[0][0]
    density_map= density_map[0][0]
    mask=mask.data.cpu().numpy()
    mask=mask[0][0]
    flow = flow[0][0]

    f_base_all = f_base_all.data.cpu().numpy()
    f_base_all = 255 * f_base_all / np.max(f_base_all)
    f_base = f_base_all[0][0]
    f_base = cv2.resize(f_base, (input_img.shape[1], input_img.shape[0]))

    x_base_all = x_base_all.data.cpu().numpy()
    x_base_all = 255 * x_base_all / np.max(x_base_all)
    x_base = x_base_all[0][0]
    x_base = cv2.resize(x_base, (input_img.shape[1], input_img.shape[0]))

    c_flow = c_flow.data.cpu().numpy()
    c_flow = 255 * c_flow / np.max(c_flow)
    c_flow = c_flow[0][0]
    c_flow = cv2.resize(c_flow, (input_img.shape[1], input_img.shape[0]))

    result_img = np.hstack(
        (input_img, gt_data, density_map, mask, flow, f_base, x_base, c_flow))
    result_img = result_img.astype(np.uint8, copy=False)
    save_name = os.path.join(output_dir, '{}_{}_{}.png'.format('mff_mask', 'expo', fname))

    result_f_base_img = np.hstack((f_base_all[0][0], f_base_all[0][1], f_base_all[0][2], f_base_all[0][3],
                                   f_base_all[0][4], f_base_all[0][5], f_base_all[0][6], f_base_all[0][7],
                                   f_base_all[0][8], f_base_all[0][9]))
    save_name_f_base = os.path.join(output_dir, '{}_{}_{}_{}.png'.format('mff_mask', 'expo', fname, 'f_base'))

    result_x_base_img = x_base_all[0][0]
    for stackiter in range(1, x_base_all.shape[1]):
        result_x_base_img = np.hstack((result_x_base_img, x_base_all[0][stackiter]))
    save_name_x_base = os.path.join(output_dir, '{}_{}_{}_{}.png'.format('mff_mask', 'expo', fname, 'x_base'))

    v_base_all = v_base_all.data.cpu().numpy()
    v_base_all = 255 * v_base_all / np.max(v_base_all)
    result_v_base_img = v_base_all[0][0]
    for stackiter in range(1, v_base_all.shape[1]):
        result_v_base_img = np.hstack((result_v_base_img, v_base_all[0][stackiter]))
    save_name_v_base = os.path.join(output_dir, '{}_{}_{}_{}.png'.format('mff_mask', 'expo', fname, 'v_base'))

    cv2.imwrite(save_name_f_base, result_f_base_img)
    cv2.imwrite(save_name_x_base, result_x_base_img)
    cv2.imwrite(save_name_v_base, result_v_base_img)
    cv2.imwrite(save_name, result_img)


def save_density_map(density_map,output_dir, fname='results.png'):    
    density_map = 255*density_map/np.max(density_map)
    density_map= density_map[0][0]
    cv2.imwrite(os.path.join(output_dir,fname),density_map)
    
def display_results(input_img, gt_data,density_map):
    input_img = input_img[0][0]
    gt_data = 255*gt_data/np.max(gt_data)
    density_map = 255*density_map/np.max(density_map)
    gt_data = gt_data[0][0]
    density_map= density_map[0][0]
    if density_map.shape[1] != input_img.shape[1]:
         input_img = cv2.resize(input_img, (density_map.shape[1],density_map.shape[0]))
    result_img = np.hstack((input_img,density_map))
    result_img  = result_img.astype(np.uint8, copy=False)

    cv2.imshow('Result', result_img)
    cv2.waitKey(0)
