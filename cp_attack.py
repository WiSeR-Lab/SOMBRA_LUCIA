import argparse, os, time, torch, random
import numpy as np
import pandas as pd
import open3d as o3d
import attack_utils.inference_attack_utils as attack_utils

from tqdm import tqdm
from torch.utils.data import DataLoader
from opencood.tools import train_utils
from opencood.tools import inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.utils.common_utils import torch_tensor_to_numpy
from opencood.visualization import vis_utils
from opencood.hypes_yaml import yaml_utils
from attack_utils import get_adv_loss
from attack_utils.adv_loss import focalLoss, PALoss
from defense.defender import cp_defense_lucia
from defense.robosac import robosac

random.seed(0)

def test_parser():
    parser = argparse.ArgumentParser(description="Adversarial attacks and defenses on OPV2V")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='directory of the model')
    parser.add_argument('--show_vis', action='store_true',
                        help='specify to show image visualization result')
    parser.add_argument('--show_sequence', action='store_true',
                        help='specify to show video visualization result.'
                             'it can note be set true with show_vis together ')
    parser.add_argument('--save_vis', action='store_true',
                        help='specify to save visualization result')
    parser.add_argument('--save_npy', action='store_true',
                        help='specify to save prediction and gt result'
                             'in npy_test file')
    parser.add_argument('--save_perturb', action='store_true',
                        help='specify to save perturbed feature for each frame')
    parser.add_argument('--iter', type=int, default=10, help='number of iterations for the attack, default 10')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate for the attack, default 0.1')
    parser.add_argument('--model', type=str, default='AttentiveFusion', help='choose from AttentiveFusion, CoAlign, Where2comm, V2VAM-needed for fetching attention default: AttentiveFusion')
    parser.add_argument('--attack_mode', type=str, default='tor', help='choose from tor, mor, (targeted or mass removal) default: tor')
    parser.add_argument('--skip', type=int, help="Skip the first x number of frames, default: 0", default=0)
    parser.add_argument('--data_dir', type=str, \
                        help="overwrite the test dataset directory specified in the model hypes", default="")
    parser.add_argument('--target_id', help='Specify the ID for which the attacker attempt to suppress its bbox, \
                        choose from: <TARGET ID>, random, in, out', default=-1)
    parser.add_argument('--loss', type=str, default='sombra', help='choose from sombra, pa, bim, default: sombra')
    parser.add_argument('--defense', action='store_true', help='specify if use Lucia defense')
    parser.add_argument('--robosac', action='store_true', help='specify if use ROBOSAC as defense')
    parser.add_argument('--async_mode', action='store_true', help='specify if use asynchronous communication')
    parser.add_argument('--exclude_attn', action='store_true', help='specify if exclude attention in the SOMBRA attack, default: False')
    opt = parser.parse_args()
    return opt

def main():
    opt = test_parser()
    assert not (opt.show_vis and opt.show_sequence), 'you can only visualize ' \
                                                'the results in single ' \
                                                'image mode or video mode'
    hypes = yaml_utils.load_yaml(None, opt)
    print('Dataset Building')
    number_stat = {'pred':[], 'gt':[], 'tp':[], 'fp':[]}
    # Specify attack to be targeted or mass object removal
    if opt.attack_mode == 'tor':
        try:
            opt.target_id = int(opt.target_id) # If specified integer ID
        except:
            pass
        hypes['remove_id'] = opt.target_id
        targeted = True
        dataset_worker = 0 # Ensure each frame has a randomized target without error
        remove_success = 0
        number_stat['target_detected'] = []
    elif opt.attack_mode == 'mor':
        targeted = False
        dataset_worker = 16 # Speed up data fetching
        orr_total = []

    # Building specifided loss function
    print('Building adversarial loss function')
    attn_loss = None
    if opt.loss == 'sombra':
        loss = focalLoss(hypes['loss']['args'], targeted)
        attn_loss = get_adv_loss(opt.model) if not opt.exclude_attn else None
    elif opt.loss == 'pa':
        loss = PALoss(targeted)
    elif opt.loss == 'bim':
        from attack_utils.untargeted_loss import untargetedAttack
        loss = untargetedAttack(hypes['loss']['args'])
        opt.attack_mode = 'bim'
    else:
        print('Invalid loss function')
        exit()
    
    defender = cp_defense_lucia if opt.defense else None

    if opt.defense: # Load our defense if applicable
        number_stat['defense_time'] = []
        number_stat['trust_score'] = []
    if opt.robosac:
        number_stat['defense_time'] = []
        

    # Load dataset
    if opt.data_dir != "":
        hypes['validate_dir'] = opt.data_dir
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    data_len = len(opencood_dataset)
    print(f"{data_len} samples found.")
    data_loader = DataLoader(opencood_dataset,
                            batch_size=1,
                            num_workers=dataset_worker,
                            collate_fn=opencood_dataset.collate_batch_test,
                            shuffle=False,
                            pin_memory=False,
                            drop_last=False)
    
    # Simulate asynchronous communication
    if opt.async_mode:
        hypes.update({'wild_setting':{
            'async': True,
            'async_mode': 'sim',
            'async_overhead': 100,
            'seed': 2025,
            'backbone_delay': 10,
            'data_size': 1.06,
            'loc_err': False,
            'ryp_std': 0.0,
            'transmission_speed': 27,
            'xyz_std': 0.0,
        }})
        opencood_dataset_async = build_dataset(hypes, visualize=True, train=False)
        data_loader_async = DataLoader(opencood_dataset_async,
                                batch_size=1,
                                num_workers=dataset_worker,
                                collate_fn=opencood_dataset.collate_batch_test,
                                shuffle=False,
                                pin_memory=False,
                                drop_last=False)
        async_dat_iter = iter(data_loader_async)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    _, model = train_utils.load_saved_model(saved_path, model)
    model.eval()

    # Create the dictionary for evaluation
    result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0},
                   0.5: {'tp': [], 'fp': [], 'gt': 0},
                   0.7: {'tp': [], 'fp': [], 'gt': 0}}

    # For visualization purposes
    if opt.show_sequence:
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        vis.get_render_option().background_color = [0.05, 0.05, 0.05]
        vis.get_render_option().point_size = 1.0
        vis.get_render_option().show_coordinate_frame = True

        # used to visualize lidar points
        vis_pcd = o3d.geometry.PointCloud()
        # used to visualize object bounding box, maximum 50
        vis_aabbs_gt = []
        vis_aabbs_pred = []
        for _ in range(50):
            vis_aabbs_gt.append(o3d.geometry.LineSet())
            vis_aabbs_pred.append(o3d.geometry.LineSet())
    
    # Begin evaluation
    total = 0
    for i, batch_data in tqdm(enumerate(data_loader)):
        # Simulate asynchronous communication
        if opt.async_mode:
            async_data = next(async_dat_iter)
            
        else:
            async_data = batch_data
            async_data = train_utils.to_device(async_data, device)

        if batch_data['ego']['cav_num'] < 2 or i < opt.skip: # A very limited number of datapoints only have 1 CAV
            continue

        batch_data = train_utils.to_device(batch_data, device)
        async_data = train_utils.to_device(async_data, device)
        gt_box_tensor = opencood_dataset.post_processor.generate_gt_bbx(batch_data)

        if opt.attack_mode == 'tor':
            target_bbox_tensor = opencood_dataset.post_processor.generate_gt_bbx(batch_data, \
                                                                                 selected_id=[opencood_dataset.actual_remove_id])
            batch_data['ego']['label_dict'] = batch_data['ego']['target_label_dict']
            batch_data['ego']['object_bbx_center'] = batch_data['ego']['target_object_bbox']
            if opt.async_mode:
                async_data['ego']['label_dict'] = async_data['ego']['target_label_dict']
                async_data['ego']['object_bbx_center'] = async_data['ego']['target_object_bbox']
        elif opt.attack_mode == 'mor':
            if opt.loss == 'sombra':
                attack_utils.get_empty_target(batch_data, opencood_dataset.post_processor)
                if opt.async_mode:
                    attack_utils.get_empty_target(async_data, opencood_dataset.post_processor)

        pred_box_tensor, pred_score, _, perturbation, trust_scores, defense_time = \
            attack_utils.inference_intermediate_fusion_attack(batch_data,
                                                            model,
                                                            opencood_dataset, attn_loss, opt.iter, opt.lr, criterion=loss,
                                                            cav_max=2, defender=defender) #cav_max=2 to limit attacker knowledge to only victim and itself
        
        if opt.defense:
            number_stat['trust_score'].append(str(trust_scores.tolist()))
            number_stat['defense_time'].append(defense_time)
        
        if opt.robosac:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            start = time.time()
            pred_box_tensor, pred_score, gt_boxes_tensor = robosac(batch_data, model, opencood_dataset, perturbation, attacker_idx=1, sampling_budget=10)
            torch.cuda.synchronize()
            defense_time = time.time() - start
            number_stat['defense_time'].append(defense_time)

        eval_utils.caluclate_tp_fp(pred_box_tensor,
                                    pred_score,
                                    gt_box_tensor,
                                    result_stat,
                                    0.3)
        tp, fp, gt = eval_utils.caluclate_tp_fp(pred_box_tensor,
                                    pred_score,
                                    gt_box_tensor,
                                    result_stat,
                                    0.5)
        eval_utils.caluclate_tp_fp(pred_box_tensor,
                                    pred_score,
                                    gt_box_tensor,
                                    result_stat,
                                    0.7)
        pred_num = 0 if pred_box_tensor is None else len(pred_box_tensor)
        gt_num = 0 if gt_box_tensor is None else len(gt_box_tensor)

        total += 1
        number_stat['pred'].append(int(pred_num))
        number_stat['gt'].append(int(gt_num))
        number_stat['tp'].append(int(tp))
        number_stat['fp'].append(int(fp))

        if targeted:
            num_detected = eval_utils.caluclate_tp_fp(pred_box_tensor, pred_score, target_bbox_tensor, result_stat, 0, write=False)
            print(f'Target detected: {num_detected}, FP: {fp}')
            number_stat['target_detected'].append(num_detected)
            if num_detected < 1:
                remove_success += 1
        else:
            orr = 1 - min(pred_num, gt_num) / gt_num
            orr_total.append(orr)
            print(f"Frame {i}, ORR {orr}")

        if opt.save_npy:
            npy_save_path = os.path.join(opt.model_dir, f'npy_iter{opt.iter}_lr{opt.lr}_{opt.attack_mode}_{opt.loss}')
            if not os.path.exists(npy_save_path):
                os.makedirs(npy_save_path)
            inference_utils.save_prediction_gt(pred_box_tensor,
                                                gt_box_tensor,
                                                batch_data['ego'][
                                                    'origin_lidar'][0],
                                                i,
                                                npy_save_path,
                                                pred_score)
        if opt.save_perturb:
            pertub_save_path = os.path.join(opt.model_dir, f'perturb_iter{opt.iter}_lr{opt.lr}_{opt.attack_mode}_{opt.loss}')
            if not os.path.exists(pertub_save_path):
                os.makedirs(pertub_save_path)
            if perturbation is not None:
                perturbation_np = torch_tensor_to_numpy(perturbation)
                np.save(os.path.join(pertub_save_path, '%04d_perturb.npy' % i), perturbation_np)

        if opt.show_vis or opt.save_vis:
            vis_save_path = ''
            if opt.save_vis:
                vis_save_path = os.path.join(opt.model_dir, f'vis_iter{opt.iter}_lr{opt.lr}_{opt.attack_mode}_{opt.loss}')
                if not os.path.exists(vis_save_path):
                    os.makedirs(vis_save_path)
                vis_save_path = os.path.join(vis_save_path, '%05d.png' % i)

            if pred_box_tensor is not None and gt_box_tensor is not None:
                opencood_dataset.visualize_result(pred_box_tensor,
                                                gt_box_tensor,
                                                batch_data['ego'][
                                                    'origin_lidar'],
                                                opt.show_vis,
                                                vis_save_path,
                                                dataset=opencood_dataset,
                                                target_tensor=None)
        if opt.show_sequence:
            pcd, pred_o3d_box, gt_o3d_box = \
                vis_utils.visualize_inference_sample_dataloader(
                    pred_box_tensor,
                    gt_box_tensor,
                    batch_data['ego']['origin_lidar'],
                    vis_pcd,
                    mode='constant'
                    )
            if i == 0:
                vis.add_geometry(pcd)
                if pred_o3d_box is not None:
                    vis_utils.linset_assign_list(vis,
                                                vis_aabbs_pred,
                                                pred_o3d_box,
                                                update_mode='add')

                vis_utils.linset_assign_list(vis,
                                                vis_aabbs_gt,
                                                gt_o3d_box,
                                                update_mode='add')

            if pred_o3d_box is not None:
                vis_utils.linset_assign_list(vis,
                                            vis_aabbs_pred,
                                            pred_o3d_box)
            vis_utils.linset_assign_list(vis,
                                            vis_aabbs_gt,
                                            gt_o3d_box)
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.001)

            
        
    eval_utils.eval_final_results(result_stat,
                                opt.model_dir, opt.loss, opt.attack_mode, iter=opt.iter, lr=opt.lr)

    stat_pd = pd.DataFrame(number_stat)
    result_name = os.path.join(opt.model_dir, '{}_result_iter{}_lr{}_loss{}.txt'.format(opt.attack_mode, opt.iter, opt.lr, opt.loss))
    f = open(result_name, "w")
    total_rows = stat_pd.shape[0]
    if opt.attack_mode == 'mor':
        with pd.option_context('mode.use_inf_as_na', True):
            asr_0 = (stat_pd['pred'] <= 0).sum() / total_rows
            asr_1 = (stat_pd['pred'] <= 1).sum() / total_rows
            orr_final = sum(orr_total) / len(orr_total)
            result = f'ASR_0: {asr_0}, ASR_1: {asr_1}, ORR: {orr_final}'
            print(result)
            f.write(result)
            f.close()
    elif opt.attack_mode == 'tor':
        fp_threshold = 2 # We require the attack to not introduce above average FP (2) after attack to minimize suspicion
        asr = ((stat_pd['target_detected'] < 1) & (stat_pd['fp'] < fp_threshold)).sum() / total_rows
        print(f'ASR: {asr}')
        f.write(f'ASR: {asr}')
        f.close()

    df_filename = os.path.join(opt.model_dir, f'det_result_iter{opt.iter}_lr{opt.lr}_{opt.attack_mode}_{opt.loss}.csv')
    stat_pd.to_csv(df_filename, index=False)

    if opt.show_sequence:
        vis.destroy_window()

if __name__ == '__main__':
    main()


        
