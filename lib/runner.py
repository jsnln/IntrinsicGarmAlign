import numpy as np
import os
from os.path import join
from tqdm import tqdm
import igl
import torch
import smplx
from smplx.lbs import vertices2joints

from pytorch3d.ops import laplacian, knn_points
from pytorch3d.loss import chamfer_distance

from .mesh_utils import compute_edges, export_to_ply, compute_eigenfunctions
from .lbs_utils import lbs, inv_lbs
from .network import DeformationField

from icecream import ic

class Runner:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg

        # main data folder
        self.data_folder = cfg['data_folder']
        
        # subfolders for intermediate results
        self.input_folder = join(self.data_folder, cfg['input_folder'])
        self.posed_folder = join(self.data_folder, cfg['posed_folder'])
        self.coarse_folder = join(self.data_folder, cfg['coarse_folder'])
        self.refined_folder = join(self.data_folder, cfg['refined_folder'])

        self.eigenfuncs_folder = join(self.data_folder, cfg['eigenfuncs_folder'])
        self.eigenfuncs_vis_folder = join(self.data_folder, cfg['eigenfuncs_vis_folder'])
        self.eigenfuncs_retified_folder = join(self.data_folder, cfg['eigenfuncs_rectified_folder'])
        self.eigenfuncs_retified_vis_folder = join(self.data_folder, cfg['eigenfuncs_rectified_vis_folder'])
        self.eigenfuncs_deformed_folder = join(self.data_folder, cfg['eigenfuncs_deformed_folder'])

        if not os.path.exists(self.refined_folder):
            os.makedirs(self.refined_folder)
        if not os.path.exists(self.eigenfuncs_folder):
            os.makedirs(self.eigenfuncs_folder)
        if not os.path.exists(self.eigenfuncs_retified_folder):
            os.makedirs(self.eigenfuncs_retified_folder)
        if not os.path.exists(self.eigenfuncs_retified_vis_folder):
            os.makedirs(self.eigenfuncs_retified_vis_folder)
        if not os.path.exists(self.eigenfuncs_deformed_folder):
            os.makedirs(self.eigenfuncs_deformed_folder)
        if not os.path.exists(self.posed_folder):
            os.makedirs(self.posed_folder)
        if not os.path.exists(self.coarse_folder):
            os.makedirs(self.coarse_folder)
        if not os.path.exists(self.eigenfuncs_vis_folder):
            os.makedirs(self.eigenfuncs_vis_folder)

        # other
        self.num_steps_coarse = cfg['num_steps_coarse']
        self.num_steps_refine = cfg['num_steps_refine']
        self.num_eigens_u = cfg['num_eigens_u'] # number of eigenfunctions to use
        self.num_eigens_c = cfg['num_eigens_c'] # number of eigenfunctions to compute (larger than the above, to avoid eigenvalue switch)
        self.eigens_used = list(range(cfg['eigens_range'][0], cfg['eigens_range'][1] + 1))

        self.start_index = cfg['scan_index_range'][0]
        self.end_index = cfg['scan_index_range'][1]
        self.subject_name = cfg['subject_name']

        self.template_id = cfg['template_id']
        self.template_scan_name = f'{self.subject_name}_{self.template_id:05d}'
        self.template_mesh_fn = cfg['template_mesh_fn']
        self.template_lbsw_fn = cfg['template_lbsw_fn']

        leg_angle = cfg['leg_angle']
        self.cpose_param = torch.zeros(1, 72).cuda()
        self.cpose_param[:, 5] =  leg_angle / 180 * np.pi
        self.cpose_param[:, 8] = -leg_angle / 180 * np.pi

        
        smpl_model = smplx.create(cfg['smpl_folder'], 'smpl', gender=cfg['gender']).cuda()
        self.smpl_parents = smpl_model.parents.clone()
        
        smpl_tpose_fn = join(self.data_folder, f'smpl_template.ply')
        smpl_tpose_verts, _ = igl.read_triangle_mesh(smpl_tpose_fn)
        tpose_verts = torch.from_numpy(smpl_tpose_verts).float()[None].cuda()
        tpose_joints = vertices2joints(smpl_model.J_regressor, tpose_verts).cuda()
        self.tpose_joints = tpose_joints
        self.smpl_tpose_verts = tpose_verts

        self.deform_scale_coarse = cfg['deform_scale_coarse']
        self.lr_coarse = cfg['lr_coarse']
        self.color_scale_coarse = cfg['color_scale_coarse']
        
        self.deform_scale_refine = cfg['deform_scale_refine']
        self.lr_refine = cfg['lr_refine']
        self.color_scale_refine = cfg['color_scale_refine']
        self.eigen_scale_refine = cfg['eigen_scale_refine']

        self.lap_scale = cfg['lap_scale']


    def compute_eigenfuncs(self):
        for scan_id in tqdm(range(self.start_index, self.end_index + 1)):
            scan_name = f"{self.subject_name}_{scan_id:05d}"
            fn_in = join(self.input_folder, f'{scan_name}.npz')
            fn_out = join(self.eigenfuncs_folder, f'{scan_name}_eigenfuncs.npz')

            try:
                mesh_data = np.load(fn_in)
            except:
                print(f'[ERROR] loading file {scan_name} failed. skipping it.')
                continue

            verts = np.array(mesh_data['scan_v'])
            faces = np.array(mesh_data['scan_f'])

            eigenvecs, eigenvals = compute_eigenfunctions(verts, faces, self.num_eigens_c)

            np.savez(fn_out, verts=verts, faces=faces, eigenvals=eigenvals, eigenfuncs=eigenvecs)

            for i in range(self.num_eigens_c):
                fn_out_vis = join(self.eigenfuncs_vis_folder, f'{scan_name}_{i:03d}.ply')
                export_to_ply(fn_out_vis, verts, faces, eigenvecs[:, i])



    def set_template(self):
        
        fn_template_data = join(self.input_folder, f'{self.template_scan_name}.npz')
        template_data = np.load(fn_template_data)
        
        verts, faces = igl.read_triangle_mesh(join(self.data_folder, self.template_mesh_fn))
        v_boundary_mask = template_data['v_boundary_mask']
        vert_colors = template_data['v_colors']


        self.template_verts = torch.from_numpy(verts).float().cuda()
        self.template_faces = torch.from_numpy(faces).long().cuda()
        self.template_edges = compute_edges(self.template_verts, self.template_faces)
        self.template_v_boundary_mask = torch.from_numpy(v_boundary_mask).cuda()
        self.template_v_colors = torch.from_numpy(vert_colors[:, :3].astype(np.float32)).float().cuda()
        verts_ori = torch.from_numpy(template_data['scan_v']).float().cuda()
        self.template_e_lengths = torch.norm(verts_ori[self.template_edges[:, 0]] - \
                                             verts_ori[self.template_edges[:, 1]], dim=-1).cuda()


        fn_template_eigenfuncs = join(self.eigenfuncs_folder, f'{self.template_scan_name}_eigenfuncs.npz')
        template_eigenfuncs_data = np.load(fn_template_eigenfuncs)

        self.template_lbsw = torch.from_numpy(np.load(join(self.data_folder, self.template_lbsw_fn))).float().cuda()
        self.template_eigenfuncs = torch.from_numpy(template_eigenfuncs_data['eigenfuncs']).float().cuda()
        self.template_eigenvals = torch.from_numpy(template_eigenfuncs_data['eigenvals']).float().cuda()

    
    def _load_packed_data(self, scan_id):
        
        scan_name = f'{self.subject_name}_{scan_id:05d}'

        fn_data = join(self.input_folder, f'{scan_name}.npz')
        data = np.load(fn_data)
        
        verts = torch.from_numpy(data['scan_v']).float().cuda()
        v_boundary_mask = torch.from_numpy(data['v_boundary_mask']).cuda()
        v_colors = torch.from_numpy(data['v_colors'][:, :3]).float().cuda()
        
        pose = torch.from_numpy(data['pose']).float().cuda()
        transl = torch.from_numpy(data['transl']).float().cuda()
        
        return verts, v_boundary_mask, pose, transl, v_colors
    
    def pose_by_lbs(self):
        for scan_id in tqdm(range(self.start_index, self.end_index + 1)):
            scan_name = f"{self.subject_name}_{scan_id:05d}"
            
            target_verts, target_v_boundary_mask, target_pose, target_transl, v_colors = self._load_packed_data(scan_id)

            template_posed = inv_lbs(self.template_verts[None].cuda(), self.tpose_joints, self.cpose_param, self.smpl_parents, self.template_lbsw[None].cuda())
            template_posed = lbs(template_posed['v_unposed'], self.tpose_joints, target_pose[None], self.smpl_parents, self.template_lbsw[None].cuda(), return_tfs=True)
            template_posed_verts = (template_posed['v_posed'] + target_transl)[0].detach().cpu().numpy()

            assert True == igl.write_triangle_mesh(join(self.posed_folder, f'{scan_name}_posed.ply'), template_posed_verts, self.template_faces.cpu().numpy())
            
    def deform_coarse_stage(self):
        for scan_id in tqdm(range(self.start_index, self.end_index + 1)):
            scan_name = f"{self.subject_name}_{scan_id:05d}"
            
            target_verts, target_v_boundary_mask, target_pose, target_transl, target_v_colors = self._load_packed_data(scan_id)
            
            
            template_posed_verts, _ = igl.read_triangle_mesh(join(self.posed_folder, f'{scan_name}_posed.ply'))
            template_posed_verts_base = torch.from_numpy(template_posed_verts).float().cuda()
            template_eigenfuncs_to_use = self.template_eigenfuncs[None][..., self.eigens_used]
            target_boundary_verts = target_verts[target_v_boundary_mask]
            
            
            net = DeformationField(len(self.eigens_used), 3).cuda()
            optimizer = torch.optim.Adam(net.parameters(), lr=self.lr_coarse)



            loss_prev = float('inf')
            loss_now = 0.0
            for i in range(self.num_steps_coarse):
                forward_def = net.forward(template_eigenfuncs_to_use)[0] * self.deform_scale_coarse
                template_posed_verts = template_posed_verts_base + forward_def


                target_posed_verts_product = torch.cat([target_verts, target_v_colors * self.color_scale_coarse], -1)
                template_posed_verts_product = torch.cat([template_posed_verts, self.template_v_colors * self.color_scale_coarse], -1)

                l_data, _ = chamfer_distance(target_posed_verts_product[None], template_posed_verts_product[None])
                l_bound, _ = chamfer_distance(target_boundary_verts[None], template_posed_verts[None, self.template_v_boundary_mask])
                
                template_e_now = torch.norm(template_posed_verts[self.template_edges[:, 0]] - \
                                            template_posed_verts[self.template_edges[:, 1]], dim=-1)
                
                loss_e = ((template_e_now - self.template_e_lengths).clip(min=0.0) ** 2).mean()
                loss = l_data + l_bound * 0.05 + loss_e * 1e2


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_now += loss.item()

                if (i+1) % 100 == 0:
                    print(f'[LOG] step {i+1:04d}, d_l: {loss_now - loss_prev:.4e}, loss: {loss.item():.4e}, l_data: {l_data:.4e}, l_bound: {l_bound:.4e}, l_e: {loss_e:.4e}')

                    loss_prev = loss_now
                    loss_now = 0.0
            
            assert True == igl.write_triangle_mesh(join(self.coarse_folder, f'optimized_template_{scan_name}.ply'),
                                                   template_posed_verts.detach().cpu().numpy(),
                                                   self.template_faces.cpu().numpy())
            
            del optimizer
            del net

    def rectify_embeddings_by_FM(self):
        for scan_id in tqdm(range(self.start_index, self.end_index + 1)):
            scan_name = f"{self.subject_name}_{scan_id:05d}"

            fn_fitted_coarse = join(self.coarse_folder, f'optimized_template_{scan_name}.ply')
            fn_target_eigenfuncs = join(self.eigenfuncs_folder, f'{scan_name}_eigenfuncs.npz')


            try:
                print(f'[LOG] loading {scan_name}')
                template_fitted_verts, _ = igl.read_triangle_mesh(fn_fitted_coarse)
            except:
                print(f'[LOG] loading failed. skipping {scan_name}')
                continue

            template_fitted_verts = torch.from_numpy(template_fitted_verts).float().cuda()


            target_eigendata = np.load(fn_target_eigenfuncs)
            target_verts = torch.from_numpy(target_eigendata['verts']).float().cuda()
            target_eigenfuncs = torch.from_numpy(target_eigendata['eigenfuncs']).float().cuda()

            _, knn_idx_tem2tar, _ = knn_points(template_fitted_verts[None], target_verts[None])
            knn_idx_tem2tar = knn_idx_tem2tar[0, :, 0]
            eigenfunc_pullback_to_template = target_eigenfuncs[knn_idx_tem2tar]
            

            A = eigenfunc_pullback_to_template[:, 1:]
            B = self.template_eigenfuncs[:, 1:]
            FM = torch.linalg.lstsq(A, B).solution

            target_eigen_corrected = target_eigenfuncs[:, 1:] @ FM

            # concatenate the constant back
            target_eigen_corrected = torch.cat([target_eigenfuncs[:, :1], target_eigen_corrected], -1)

            np.savez(join(self.eigenfuncs_retified_folder, scan_name + '_rectified.npz'),
                eigenfuncs=target_eigen_corrected.cpu().numpy(),
                eigenvals=target_eigendata['eigenvals'],
                verts=target_eigendata['verts'],
                faces=target_eigendata['faces'],
            )

            validate = True

            if validate:
                for i in range(target_eigen_corrected.shape[1]):
                    export_to_ply(join(self.eigenfuncs_retified_vis_folder, scan_name + f'_rectified_{i:03d}.ply'),
                                target_eigendata['verts'], target_eigendata['faces'],
                                target_eigen_corrected.cpu().numpy()[:, i])

    def deform_refining_stage(self):
        for scan_id in tqdm(range(self.start_index, self.end_index + 1)):
            scan_name = f'{self.subject_name}_{scan_id:05d}'

            template_eigenfuncs = self.template_eigenfuncs[:, self.eigens_used].clone() * self.eigen_scale_refine / self.template_eigenvals[self.eigens_used].sqrt().clip(min=1e-10)
            template_v_colors = self.template_v_colors * self.color_scale_refine

            _, target_v_boundary_mask, _, _, target_v_colors = self._load_packed_data(scan_id)

            target_eigenfuncs_data = np.load(join(self.eigenfuncs_retified_folder, f'{scan_name}_rectified.npz'))
            target_eigenfuncs = target_eigenfuncs_data['eigenfuncs'][:, self.eigens_used] * self.eigen_scale_refine / np.sqrt(target_eigenfuncs_data['eigenvals'][self.eigens_used]).clip(min=1e-10)
            target_eigenfuncs = torch.from_numpy(target_eigenfuncs).float().cuda()
            target_eigenfuncs_w_colors = torch.cat([target_eigenfuncs, target_v_colors], -1)

            net = DeformationField(dim=len(self.eigens_used), out_dim=len(self.eigens_used)).cuda()
            optimizer = torch.optim.Adam(net.parameters(), lr=self.lr_refine)

            bar = tqdm(range(self.num_steps_refine))
            for i in bar:
                eigenfuncs_deformation = net.forward(template_eigenfuncs) * self.deform_scale_refine
                template_eigenfuncs_deformed = template_eigenfuncs + eigenfuncs_deformation
                template_eigenfuncs_deformed_w_colors = torch.cat([template_eigenfuncs_deformed, template_v_colors], -1)

                knn_dists_s2m, _, _ = knn_points(target_eigenfuncs_w_colors[None], template_eigenfuncs_deformed_w_colors[None])
                knn_dists_m2s, _, _ = knn_points(template_eigenfuncs_deformed_w_colors[None], target_eigenfuncs_w_colors[None])
                loss_scan = (knn_dists_m2s.mean() + knn_dists_s2m.mean())

                # no color
                knn_dists_s2m_boundary, _, _ = knn_points(target_eigenfuncs[None, target_v_boundary_mask], template_eigenfuncs_deformed[None, self.template_v_boundary_mask])
                knn_dists_m2s_boundary, _, _ = knn_points(template_eigenfuncs_deformed[None, self.template_v_boundary_mask], target_eigenfuncs[None, target_v_boundary_mask])
                loss_scan_boundary_w_colors = (knn_dists_m2s_boundary.mean() + knn_dists_s2m_boundary.mean())
                

                l_reg = (eigenfuncs_deformation ** 2).sum(-1).mean()

                loss = loss_scan + l_reg * 1e-1 + loss_scan_boundary_w_colors * 0.1


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                bar.set_description(f'[{i:05d}] l_s = {loss_scan:.4e}, l_b = {loss_scan_boundary_w_colors:.4e}, l_reg = {l_reg:.4e}')
            
            np.save(join(self.eigenfuncs_deformed_folder, f'{scan_name}_deformed_eigenfuncs.npy'), template_eigenfuncs_deformed.detach().cpu().numpy())
    
            del net
            del optimizer

    def shape_transfer(self):
        for scan_id in tqdm(range(self.start_index, self.end_index + 1)):
            scan_name = f'{self.subject_name}_{scan_id:05d}'
            template_eigenfuncs_deformed = np.load(join(self.eigenfuncs_deformed_folder, f'{scan_name}_deformed_eigenfuncs.npy'))
            template_eigenfuncs_deformed = torch.from_numpy(template_eigenfuncs_deformed).float().cuda()

            target_eigenfuncs_data = np.load(join(self.eigenfuncs_retified_folder, f'{scan_name}_rectified.npz'))
            target_eigenfuncs = target_eigenfuncs_data['eigenfuncs'][:, self.eigens_used] * self.eigen_scale_refine / np.sqrt(target_eigenfuncs_data['eigenvals'][self.eigens_used]).clip(min=1e-10)
            target_eigenfuncs = torch.from_numpy(target_eigenfuncs).float().cuda()
            
            target_verts = target_eigenfuncs_data['verts']
            target_verts = torch.from_numpy(target_verts).float().cuda()

            target_faces = target_eigenfuncs_data['faces']
            target_faces = torch.from_numpy(target_faces).long().cuda()

            _, target_v_boundary_mask, _, _, _ = self._load_packed_data(scan_id)

            knn_dists, knn_index, _ = knn_points(template_eigenfuncs_deformed[None], target_eigenfuncs[None], K=4)

            knn_3d_coords = target_verts[knn_index[0]]
            knn_weights = 1.0 / knn_dists[0].clamp(min=1e-12).sqrt()

            boundary_weighting = torch.ones(target_verts.shape[0]).cuda()
            boundary_weighting[target_v_boundary_mask] = 2.5
            knn_boundary_weighting = boundary_weighting[knn_index[0]]
            knn_boundary_weighting[self.template_v_boundary_mask] = knn_boundary_weighting[self.template_v_boundary_mask] ** 2
            knn_weights *= knn_boundary_weighting

            knn_weights /= knn_weights.sum(-1, keepdim=True)

            transferred_3d_verts = torch.einsum('ik,ikc->ic', knn_weights, knn_3d_coords)

            ########### gradient domain transfer (uniform lap)
            L_template = laplacian(self.template_verts, self.template_edges)
            L_template = L_template.coalesce()


            target_edges = compute_edges(target_verts, target_faces)
            L_target = laplacian(target_verts, target_edges)

            lap_coords = (L_target @ target_verts)

            knn_lap_coords_K_1 = lap_coords[knn_index[0, ..., 0]]

            ########## new #############
            E_idx = torch.arange(L_template.shape[0], device='cuda')
            E_idx = torch.stack([E_idx + L_template.shape[0], E_idx], 0)
            E_vals = torch.ones(L_template.shape[0], device='cuda')
            A_idx = torch.cat([L_template.indices(), E_idx], 1)
            A_vals = torch.cat([L_template.values() * self.lap_scale, E_vals], 0)
            A = torch.sparse.FloatTensor(A_idx, A_vals, (L_template.shape[0]*2, L_template.shape[0])).to_dense()
            ### using to_dense since torch has no lstsq backend for sparse cuda tensors
            ### NOTE maybe try scipy solver?

            b = torch.cat([
                knn_lap_coords_K_1 * self.lap_scale,
                transferred_3d_verts
            ], 0)

            print(A.shape, b.shape)

            v = torch.linalg.lstsq(A, b).solution
            
            assert True == igl.write_triangle_mesh(join(self.refined_folder, f'{scan_name}_transfered_vcolored_w_lap.ply'), v.cpu().numpy(), self.template_faces.cpu().numpy())
