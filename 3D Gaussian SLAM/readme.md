<!-- PROJECT LOGO -->

<p align="center">

  <h1 align="center">SplaTAM: Splat, Track & Map 3D Gaussians for Dense RGB-D SLAM</h1>
  <p align="center">
    <a href="https://nik-v9.github.io/"><strong>Nikhil Keetha</strong></a>
    ·
    <a href="https://jaykarhade.github.io/"><strong>Jay Karhade</strong></a>
    ·
    <a href="https://krrish94.github.io/"><strong>Krishna Murthy Jatavallabhula</strong></a>
    ·
    <a href="https://gengshan-y.github.io/"><strong>Gengshan Yang</strong></a>
    ·
    <a href="https://theairlab.org/team/sebastian/"><strong>Sebastian Scherer</strong></a>
    <br>
    <a href="https://www.cs.cmu.edu/~deva/"><strong>Deva Ramanan</strong></a>
    ·
    <a href="https://www.vision.rwth-aachen.de/person/216/"><strong>Jonathon Luiten</strong></a>
  </p>
  <h3 align="center"><a href="https://arxiv.org/pdf/2312.02126.pdf">Paper</a> | <a href="https://youtu.be/jWLI-OFp3qU">Video</a> | <a href="https://spla-tam.github.io/">Project Page</a></h3>
  <div align="center"></div>
</p>

## Coding-process
To run Splatam etc.
##### 1. 读取`Config`dict格式数据信息并输出在窗口, `Config`包含信息如下:
<details>
<summary>[基础通用配置]</summary>

```py
    'Workdir'
    'run_name'
    'seed':种子用于固定随机
    'primary_device'
    'map_every'
    'keyframe_every' # 每隔多少frame添加一个keyframe
    'mapping_window_size'
    'report_global_progress_every' # 每隔多少 frame report Final Tracking Progress
    'eval_every'
    'scene_radius_depth_ratio'
    'mean_sq_dist_method'
    'report_iter_progress'
    'load_checkpoint'
    'checkpoint_time_idx'
    'save_checkpoints'
    'checkpoint_interval'
    'use_wandb'
    'wandb'
    'data'
    'tracking'
    'mapping'
    'viz' 
```
</details>

<details>
<summary>[wandb]</summary>
</details>

<details>
<summary>[data]</summary>

```py
    'basedir',
    'gradslam_data_cfg',
    'sequence',
    'desired_image_height',
    'desired_image_width',
    'start',
    'end',
    'stride',
    'num_frames',
    'ignore_bad',
    'use_train_split',
    'densification_image_height',
    'densification_image_width',
    'tracking_image_height',
    'tracking_image_width'
```
</details>

<details>
<summary>[tracking]</summary>

```py
    'use_gt_poses', # 是否使用ground truth
    'forward_prop', # 是否使用匀速模型预测当前帧相机位姿
    'num_iters',
    'use_sil_for_loss',# 计算 loss 时是否使用 silhouette
    'sil_thres',# 计算 loss 时是否忽略
    'use_l1',
    'ignore_outlier_depth_loss',
    'loss_weights',
    'lrs',
    'use_depth_loss_thres', # 是否根据 depth loss 阈值结束迭代
    'depth_loss_thres', # depth loss 阈值
    'visualize_tracking_loss' # 可视化tracking loss
```
</details>

<details>
<summary>[mapping]</summary>

```py
    'num_iters', # Mapping 时的迭代数量
    'add_new_gaussians',
    'sil_thres',
    'use_l1',
    'use_sil_for_loss',# 计算 loss 时是否使用 silhouette
    'ignore_outlier_depth_loss', # 计算 loss 时是否忽略outlier
    'loss_weights', # depth 和 color 损失的权重
    'lrs',
    'prune_gaussians', # 是否对3D高斯进行修剪
    'pruning_dict: {
        'start_after' # num_iters 迭代时，若迭代次数 >= start_after, 才开始考虑修剪
        'remove_big_after' # 多少帧结束后才考虑移除过大的3D高斯
        'stop_after', # num_iters 迭代时，若迭代次数 > stop_after, 则停止修剪
        'prune_every', # num_iters 迭代时，每隔多少次进行一次修剪
        'removal_opacity_threshold', # opacity阈值
        'final_removal_opacit',# opacity阈值
        '_threshold', 
        'reset_opacities', 
        'reset_opacities_every'
    }',
    'use_gaussian_splatting_densification',
    'densify_dict'
```
</details>

<details>
<summary>[viz]</summary>

```
    'render_mode',
    'offset_first_viz_cam',
    'show_sil',
    'visualize_cams',
    'viz
```
</details>

--------------------------------     
##### 2. 代码`dataset_config = Config["data"] `并添加额外keys(如果不存在的话):
<details>
<summary>[dataset_config]</summary>

```
    'basedir',
    'gradslam_data_cfg',
    'sequence',
    'desired_image_height',
    'desired_image_width',
    'start',
    'end',
    'stride',
    'num_frames',
    'ignore_bad',
    'use_train_split',
    'densification_image_height',
    'densification_image_width',
    'tracking_image_height',
    'tracking_image_width'
```
</details>

--------------------------------   

##### 3. 通过`get_dataset`函数读取相机照片数据得到变量`dataset`:<span style="color: red;">
💡 特别注意: </span>若 `relative_pose=True`，则`transformed_poses`为所有frame相对于第一个frame的变换矩阵($T_{Cam1,Cam}$)，<span style="color: red;">相当于 World frame 为第一帧camera的位置</span>，且后续使用的都是 `transformed_poses` (非常坑爹的设计，十分混乱，因为后续代码让`poses = transformed_poses`，非常容易混淆)</br>
<details>
<summary>[get_dataset 代码]</summary>

```
dataset = get_dataset(
        config_dict=gradslam_data_cfg,
        basedir=dataset_config["basedir"],
        sequence=os.path.basename(dataset_config["sequence"]),
        start=dataset_config["start"],
        end=dataset_config["end"],
        stride=dataset_config["stride"],
        desired_height=dataset_config["desired_image_height"],
        desired_width=dataset_config["desired_image_width"],
        device=device,
        relative_pose=True,
        ignore_bad=dataset_config["ignore_bad"],
        use_train_split=dataset_config["use_train_split"],
    )

```
</details> 

<details>
<summary>[dataset 成员]</summary>

```
    'channels_first',
    'color_paths',
    'crop_edge',
    'crop_size',
    'cx',
    'cy',
    'depth_paths',
    'desired_height',
    'desired_width',
    'device',
    'distortion',
    'dtype',
    'embedding_dim',
    'embedding_dir',
    'embedding_paths',
    'end',
    'fx',
    'fy',
    'get_cam_K',
    'get_filepaths',
    'height_downsample_ratio',
    'input_folder',
    'load_embeddings',
    'load_poses',
    'name',
    'normalize_color',
    'num_imgs',
    'orig_height',
    'orig_width',
    'png_depth_scale',
    'pose_path',
    'poses',
    'read_embedding_from_file',
    'relative_pose',
    'retained_inds',
    'start',
    'transformed_poses',
    'width_downsample_ratio'
```
</details>

--------------------------------

##### 4. 通过函数 `initialize_first_timestep` 初始化参数、Canonical和 Densification相机参数 (如果需要，启动单独的数据加载器进行密集化处理)

###### 4.1 初始化Tupel变量 `cam`，参与后续 diff-gaussian-rasterization-w-depth 的计算:
<details> <summary>[cam 属性]</summary>

```
  cam = Camera(
        image_height=h,
        image_width=w,
        tanfovx=w / (2 * fx),
        tanfovy=h / (2 * fy),
        bg=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=w2c,
        projmatrix=full_proj,
        sh_degree=0,
        campos=cam_center,
        prefiltered=False
    )
```
</details> 

`cam` 包含信息均为第一帧的情况，且被用于 `diff-gaussian-rasterization-w-depth` C++代码中
* `w2c，opengl_proj` 需要转置(适配C++代码)
* <span style="color: red;">`opengl_proj` 为以相机内参为基础建立的管线渲染中的Projection matrix，其将View坐标转化为NDC坐标 </span>
* 简化了模型，未使用球谐函数
  
###### 4.2 通过函数 `initialize_params` 得到变量 `params，variables`:
<details>
<summary>[params 为3D高斯和Camera训练参数]</summary>

```
'means3D', 
'rgb_colors', 
'unnorm_rotations', 
'logit_opacities', 
'log_scales', 
'cam_unnorm_rots',
'cam_trans'
```
</details>

* `mean_3D` 表示3D高斯中心坐标，通过函数 `get_pointcloud` 得到，若前文提到的 `transform_pts = True`，`mean_3D` 处于世界坐标系 (💡 特别注意: </span>若 `relative_pose=True`，<span style="color: red;">World frame 为第一帧camera的位置，后续不再做提醒</span>)，否则位于Camera frame
* `rgb_colors` 表示3D高斯颜色(简化了SH部分，不考虑视角)
* `unnorm_rotations` 表示3D高斯椭球旋转(本文简化为isotropic)，使用未归一化四元数表示   
* `logit_opacities` 表示不透明度，默认为0.5
* `log_scales` 表示 $log$ 形式3D高斯轴长: $\log\left(\frac{2\cdot Depth_z}{FX + FY}\right)$
* `cam_unnorm_rots` 表示 $R_{Cam,Cam_1}$
* `cam_trans` 表示 $t_{Cam,Cam_1}$

<details>
<summary>[variables 为一些变量]</summary>

```
'max_2D_radius', 
'means2D_gradient_accum', 
'denom', 
'timestep', 
'scene_radius',

```
</details>

* `scene_radius` 表示一个初始化的 estimate of scene radius for Gaussian-Splatting Densification: `torch.max(depth)/scene_radius_depth_ratio`

###### 4.3 返回 `params, variables, intrinsics, w2c, cam`
--------------------------------
##### 5. 初始化一些变量
```python
# Initialize list to keep track of Keyframes
keyframe_list = []
keyframe_time_indices = []
    
# Init Variables to keep track of ground truth poses and runtimes
gt_w2c_all_frames = []
tracking_iter_time_sum = 0
tracking_iter_time_count = 0
mapping_iter_time_sum = 0
mapping_iter_time_count = 0
tracking_frame_time_sum = 0
tracking_frame_time_count = 0 # 每次iter+1
mapping_frame_time_sum = 0
mapping_frame_time_count = 0

```
--------------------------------
##### 6. 核心步骤，迭代每一个frame
###### 6.1 前置步骤
➡️ 获取当前帧相片的颜色(进行归一化)，深度和 ground_truth (` curr_gt_w2c ` 保存截至当前帧所有的ground_truth)
> 此处的`ground_truth`，若 `relative_pose=True`，则`ground_truth`为相对坐标，表示 $T_{Cam,Cam_1}$

➡️ 初始化选定帧的Mapping数据为变量 `curr_data`:
<details>
<summary>[curr_data]</summary>

```
curr_data = {
              'cam': cam, 
              'im': color, 
              'depth': depth, 
              'id': iter_time_idx, 
              'intrinsics': intrinsics, 
              'w2c': first_frame_w2c, 
              'iter_gt_w2c_list': curr_gt_w2c
            }
```
</details>

* id 为当前帧的编号
* `w2c` 为 world frame 到 first camera frame 的变换矩阵

➡️ 初始化选定帧的Tracking数据为变量 `tracking_curr_data = curr_data`

➡️ 初始化当前帧相机位置(第一帧除外)
若 `config['tracking']['forward_prop']`，则认为相机匀速运动，根据上一帧推测当前位姿，否则则认为当前帧位姿等于上一帧


###### 6.2 Tracking
➡️ `config['tracking']['use_gt_poses'] == True`: 如果使用ground truth
...
...
➡️ `config['tracking']['use_gt_poses'] == false`: 如果不使用ground truth
<u>Tracking 部分只更新相机位姿而不更新3D高斯参数</u>

根据 `config['tracking']['num_iters']` 迭代优化:

首先通过 `get_loss` 函数计算损失(函数内可以确定不同策略，比如只更新高斯，只更新相机位姿等)。函数具体步骤如下:
* `Render` 函数渲染得到 color, radii(3倍椭球长半轴，保证99%概率)，depth 和 silhouett
* 通过 outlier 和 silhouett 确定一个Mask，过滤不需要的数据
* 计算 Depth loss，使用 L1 Loss / 计算 RGB loss，使用 L1 Loss + SSIM Loss -> 加权得到最终 loss
* 根据 `Render` 得到的 radii (radii > 0)得到一个 Mask `seen`，存储成 `variables['seen']`，并更新 `variables['max_2D_radius'][seen]`:
```py
seen = radius > 0
variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
variables['seen'] = seen
weighted_losses['loss'] = loss
```
* 返回变量 `loss(加权后的最终损失), variables, losses(depth，color和加权后最终损失)`

若当前 loss < current_min_loss (current_min_loss 初始化为$10^{20}$)，则更新位姿和 current_min_loss，并根据 `config['use_wandb']` 选择是否 report progress(relative pose error，euclidean distance error，ATE RMSE..)
```py
if loss < current_min_loss:
                        current_min_loss = loss
                        candidate_cam_unnorm_rot = params['cam_unnorm_rots'][..., time_idx].detach().clone()
                        candidate_cam_tran = params['cam_trans'][..., time_idx].detach().clone()
```
若迭代次数达到 `config['tracking']['num_iters']`:
* 若 `config['tracking']['use_depth_loss_thres'] = True` 且 depth loss < `depth_loss_thres`，则结束迭代，否则再迭代一轮，迭代结束时无论如何都结束迭代
* 若 `config['tracking']['use_depth_loss_thres'] = False`，则立刻停止迭代

➡️ `if time_idx == 0 or (time_idx+1) % config['report_global_progress_every'] == 0:` 
每隔 `report_global_progress_every` report一次 progress ( relative pose error，euclidean distance error，ATE RMSE)。<u>此部分虽然属于Tracking, 但只用于输出结果，因此并不会对Camera Poses进行更新。</u>

然后3D高斯椭球中心点坐标将被转换到 Camera Frame 定义为 `transformed_pts`, 进行后续计算。 

>接下来就涉及管线渲染C++部分代码。代码链接: [diff-gaussian-rasterization-w-depth](这部分https://github.com/JonathonLuiten/diff-gaussian-rasterization-w-depth)
>
>:roll_eyes: 这部分非常繁琐，暂时挖个坑.. Diff-gaussian代码是通过Cuda手撸出来整套流程，包括前向传播得到颜色深度，再反向传播更新参数。细节方面比如如何把3D高斯投影到2D，论文提到的 Fast differentiable rasterizer(快速可微渲染器)的实现都在这部分代码里。
>
>然后利用 `PYBIND11_MODULE` 模块将接口暴露给C++:
```CPP
PYBIND11_MODULE(my_cuda_extension, m) {
    m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
    m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
    m.def("mark_visible", &markVisible);
}
```
>此外在 `__init__.py`  文件内定义类 `_RasterizeGaussians` 继承 `torch.autograd.Function`，手动定义 `forward， backward`函数并调用C++接口实现渲染部分, 并通过 `setuptools` 对 `setup.py`进行设置，打包整个包，最终可通过 `pip install git+` 直接安装。

回到 Tracking 部分，通过 `transformed_params2rendervar` 函数得到(颜色)渲染所需参数:
```python
def transformed_params2rendervar(params, transformed_pts):
    rendervar = {
        'means3D': transformed_pts,
        'colors_precomp': params['rgb_colors'],
        'rotations': F.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(torch.tile(params['log_scales'], (1, 3))),
        'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device="cuda") + 0
    }
    return rendervar

```

通过 `transformed_params2depthplussilhouette` 函数得到深度和silhouette渲染所需参数:

```py
def transformed_params2depthplussilhouette(params, w2c, transformed_pts):
    rendervar = {
        'means3D': transformed_pts,
        'colors_precomp': get_depth_and_silhouette(transformed_pts, w2c),
        'rotations': F.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(torch.tile(params['log_scales'], (1, 3))),
        'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device="cuda") + 0
    }
    return rendervar

    def get_depth_and_silhouette(pts_3D, w2c):
    """
    Function to compute depth and silhouette for each gaussian.
    These are evaluated at gaussian center.
    """
    # Depth of each gaussian center in camera frame
    pts4 = torch.cat((pts_3D, torch.ones_like(pts_3D[:, :1])), dim=-1)
    pts_in_cam = (w2c @ pts4.transpose(0, 1)).transpose(0, 1)
    depth_z = pts_in_cam[:, 2].unsqueeze(-1) # [num_gaussians, 1]
    depth_z_sq = torch.square(depth_z) # [num_gaussians, 1]

    # Depth and Silhouette
    depth_silhouette = torch.zeros((pts_3D.shape[0], 3)).cuda().float()
    depth_silhouette[:, 0] = depth_z.squeeze(-1)
    depth_silhouette[:, 1] = 1.0
    depth_silhouette[:, 2] = depth_z_sq.squeeze(-1)
    
    return depth_silhouette
```
通过可微渲染器前向传播得到深度，颜色和silhouette渲染结果:
```py
depth_sil, _, _, = Renderer(raster_settings=data['cam'])(**depth_sil_rendervar)
im, _, _, = Renderer(raster_settings=data['cam'])(**rendervar)
```
提取 depth>0 和 silhouett > 阈值 的Mask，并根据Mask计算psnr, rmse, L1 Loss。后续代码涉及Wandb和可视化部分不多作介绍。

###### 6.3 Densification & KeyFrame-based Mapping
➡️ `time_idx == 0 or (time_idx+1) % config['map_every'] == 0:`第一帧和每`config['map_every']`帧进行一次 Mapping和 Densification

若 `config['mapping']['add_new_gaussians'] = True` 且不为第一帧，则使用函数 `add_new_gaussians` 进行 Densification:
```py
# Add new Gaussians to the scene based on the Silhouette
params, variables = add_new_gaussians(params, variables,
                                      densify_curr_data,  
                                      config['mapping']['sil_thres'], time_idx,
                                      config['mean_sq_dist_method'])

```
整个 Densification 的逻辑是先根据当前参数渲染一个 silhouett，因为 silhouett类似于可视性占有度，在$\alpha$渲染后若小于阈值(默认设定为0.5)，则证明此处3D高斯过少密度较低。或者如果渲染出的深度大于深度图中得到深度且此处的L1误差大于50倍的MDE(median depth error)，则认为在目前的3D高斯前方应该还存在3D高斯。
>此外本人理解的不是特别到位，猜测作者的意思是因为depth是通过$\alpha$渲染得到，本质上是一种加权得到的深度，若得到的深度过大，则可能是缺少位于前方的3D高斯。

通过这些限定条件将符合条件的像素类似于 <b>4.2</b> 转化为新的3D高斯，并更新 `params, variables`。

Densification 结束后，为 Mapping 做一些准备，获取当前的 w2c 矩阵:
```py
# Get the current estimated rotation & translation
curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
curr_cam_tran = params['cam_trans'][..., time_idx].detach()
curr_w2c = torch.eye(4).cuda().float()
curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
curr_w2c[:3, 3] = curr_cam_tran
```
选择k个关键帧进行Mapping。关键帧包含当前帧，上一个关键帧和k-2个和当前帧拥有最多Overlapping的关键帧。

K-2 关键帧选取原则:
* 从当前帧随机选取1600(default)个像素点根据内参和外参和深度图得到world frame下3D坐标
* 遍历除当前帧和上一关键帧之外的所有关键帧，根据估计到的外参(w2c)将随机选取的3D坐标投影到这些关键帧，确认多少点落入了图像得到百分比，选取百分比最高的k-2个关键帧

➡️ Mapping: 迭代 `config['mapping']['num_iters']` 次，每次随机从候选关键帧中选择一个Keyframe，计算Loss, 并更新且只更新3D高斯参数(本人认为是改进空间很大的地方, "伪全局"更新)

首先通过 Tracking 提到的 `get_loss` 函数计算损失。不同之处在于 silhouett 在 Mapping 时不使用，作者说希望优化整个场景。

计算完 loss 之后进行后向传播，更新参数，然后对3D高斯进行修剪。移除 opacities 小于阈值和半径过大的3D高斯，并考虑是否每隔一定数目的迭代就重置 opacities。

紧接着，代码中可以选择是否在 Mapping 过程中使用 Gaussian-Splatting's Gradient-based Densification(挖个坑)，但实际上并没使用此方法，作者不希望在当前高斯已经能准确表现场景几何形状的地方再添加高斯。
###### 6.4 Add frame to keyframe list
若当前帧为第一帧和倒数第三帧，或每隔 `config['keyframe_every']` 个帧，则这些 frame 被当作 keyframe 储存。一次frame的迭代结束，回到 <b> 6.1 </b>，进入下一次迭代。






