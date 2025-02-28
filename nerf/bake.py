from .utils_mobile import *
from .network_mobile import *
from nerf.provider import NeRFDataset
import numpy as np
from tqdm import tqdm
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def write_floatpoint_image(name,img):
  img = np.clip(np.array(img)*255,0,255).astype(np.uint8)
  cv2.imwrite(name,img[:,:,::-1])

def export_pytorch_mlp_explicit(model, in_dim, hidden_dim, out_dim, file_path):
    """
    Exports a simple 2-hidden-layer MLP with shape:
        Linear(in_dim -> hidden_dim)
        ReLU
        Linear(hidden_dim -> hidden_dim)
        ReLU
        Linear(hidden_dim -> out_dim)
    to a JSON file that matches your old tinycudann-like format.
    """
    # Move model to CPU
    model = model.cpu()

    # Grab each linear layer explicitly from the sequential model
    # (Assumes your model is: [Linear, ReLU, Linear, ReLU, Linear])
    linear1 = model.model[0]
    linear2 = model.model[2]
    linear3 = model.model[4]

    # Get the weights/bias from each layer. PyTorch's Linear weight is
    # shaped (out_features, in_features), so we transpose to get (in_features, out_features).
    l0_w = linear1.weight.detach().cpu().numpy().T  # (in_dim, hidden_dim)
    l0_b = linear1.bias.detach().cpu().numpy()      # (hidden_dim,)

    l1_w = linear2.weight.detach().cpu().numpy().T  # (hidden_dim, hidden_dim)
    l1_b = linear2.bias.detach().cpu().numpy()      # (hidden_dim,)

    l2_w = linear3.weight.detach().cpu().numpy().T  # (hidden_dim, out_dim)
    l2_b = linear3.bias.detach().cpu().numpy()      # (out_dim,)

    # Build a dictionary with the same keys as in your original approach
    mlp_params = {
        "0_weights": l0_w.tolist(),
        "0_bias":    l0_b.tolist(),
        "1_weights": l1_w.tolist(),
        "1_bias":    l1_b.tolist(),
        "2_weights": l2_w.tolist(),
        "2_bias":    l2_b.tolist(),
    }

    # Save to JSON
    with open(file_path, "w") as f:
        json.dump(mlp_params, f, indent=2)

    print(f"MLP parameters exported to {file_path}")
    model.to(device)

def export_tcnn_fullyfused_mlp(net, in_dim, hidden_dim, out_dim, file_path):
    net = net.to("cpu")

    # 1) Get the flattened param buffer from state_dict
    st = net.state_dict()
    print("Keys in state_dict:", list(st.keys()))
    for name, param in net.named_parameters():
        print(name, param.shape)

    # Usually: ['params'] for a FullyFusedMLP
    param_buf = st["params"].detach().cpu().numpy()
    print("param buf shape", param_buf.shape)
    # 2) Slice param_buf to each layer's shape, just like you'd do in get_params()
    # For 3 layers (2 hidden + 1 output) example:
    size_l0w = in_dim * hidden_dim
    size_l0b = hidden_dim
    size_l1w = hidden_dim * hidden_dim
    size_l1b = hidden_dim
    size_l2w = hidden_dim * out_dim
    size_l2b = out_dim

    offset = 0
    l0_w = param_buf[offset : offset+size_l0w].reshape(in_dim, hidden_dim)
    offset += size_l0w
    l0_b = param_buf[offset : offset+size_l0b]
    offset += size_l0b
    l1_w = param_buf[offset : offset+size_l1w].reshape(hidden_dim, hidden_dim)
    offset += size_l1w
    l1_b = param_buf[offset : offset+size_l1b]
    offset += size_l1b
    l2_w = param_buf[offset : offset+size_l2w].reshape(hidden_dim, out_dim)
    offset += size_l2w
    l2_b = param_buf[offset : offset+size_l2b]
    offset += size_l2b
    print('final offset', offset)

    # 3) Build JSON dict
    mlp_params = {
        "0_weights": l0_w.tolist(),
        "0_bias":    l0_b.tolist(),
        "1_weights": l1_w.tolist(),
        "1_bias":    l1_b.tolist(),
        "2_weights": l2_w.tolist(),
        "2_bias":    l2_b.tolist(),
    }
    

    # 4) Save to file
    # torch.save(net.state_dict(), "mlp.pth")
    import json
    with open(file_path, 'w') as f:
        json.dump(mlp_params, f)
    print("Saved MLP parameters to", file_path)
    net.to(device)


def get_feature_png(feat, out_feat_num):
  h,w,c = feat.shape
  #deal with opencv BGR->RGB
  if c%4!=0:
    print("ERROR: c%4!=0")
    1/0
  out = []
  for i in range(out_feat_num):
    ff = np.zeros([h,w,4],np.uint8)
    ff[...,0] = feat[..., i*4+2] #B
    ff[...,1] = feat[..., i*4+1] #G
    ff[...,2] = feat[..., i*4+0] #R
    ff[...,3] = feat[..., i*4+3] #A
    out.append(ff)
  return out

def check_triangle_visible(mask,out_cell_num, out_img_w, out_cell_size, quad_t1_mask, quad_t2_mask):
  py = out_cell_num//out_img_w
  px = out_cell_num%out_img_w

  tsy = py*out_cell_size
  tey = py*out_cell_size+out_cell_size
  tsx = px*out_cell_size
  tex = px*out_cell_size+out_cell_size

  quad_m = mask[tsy:tey,tsx:tex]
  t1_visible = np.any(quad_m*quad_t1_mask)
  t2_visible = np.any(quad_m*quad_t2_mask)

  return (t1_visible or t2_visible), t1_visible, t2_visible

def mask_triangle_invisible(mask,out_cell_num,imga, out_img_w, out_cell_size, quad_t1_mask, quad_t2_mask):
  py = out_cell_num//out_img_w
  px = out_cell_num%out_img_w

  tsy = py*out_cell_size
  tey = py*out_cell_size+out_cell_size
  tsx = px*out_cell_size
  tex = px*out_cell_size+out_cell_size

  quad_m = mask[tsy:tey,tsx:tex]
  t1_visible = np.any(quad_m*quad_t1_mask)
  t2_visible = np.any(quad_m*quad_t2_mask)

  if not (t1_visible or t2_visible):
    imga[tsy:tey,tsx:tex] = 0

  elif not t1_visible:
    imga[tsy:tey,tsx:tex] = imga[tsy:tey,tsx:tex]*quad_t2_mask[:,:,None]

  elif not t2_visible:
    imga[tsy:tey,tsx:tex] = imga[tsy:tey,tsx:tex]*quad_t1_mask[:,:,None]

  return (t1_visible or t2_visible), t1_visible, t2_visible

def get_density_color(pts, acc_grid, model, threshold, ):
    #redefine net
    pts = torch.tensor(pts, dtype=torch.float32, device=device)
    acc_grid_masks = get_acc_grid_masks(pts, acc_grid.to(device), model.grid_min, model.grid_max, model.grid_size)
    B, N, _ = pts.shape
    with torch.no_grad():
        # Now use the MLP to compute density and features
        alphas_info = model.density(pts.view(-1,3))
        alphas = alphas_info['sigma']
        alphas = alphas.view(B, N)
        # print("alphas shape ", alphas.shape)
        # print("acc_grid_masks shape ", acc_grid_masks.shape)
        alphas = alphas * (acc_grid_masks>=threshold)
        alphas = (alphas>0.5).to(torch.uint8)
        #previous: (features+dirs)->MLP->(RGB)
        mlp_features = alphas_info['geo_feat']
        # print("mlp_features shape ", mlp_features.shape)
        #discretize
        # 1) Round and convert to uint8
        mlp_features_ = (mlp_features * 255).round().to(torch.uint8)
        # print("mlp_features_ shape ", mlp_features_.shape)
        # 2) Clip the first channel in [1, 255] and multiply by alpha
        mlp_features_0 = torch.clamp(mlp_features_[..., 0:1], min=1, max=255).float() * alphas.unsqueeze(-1)
        # print("mlp_features_0 shape ", mlp_features_0.shape)
        # 3) Multiply the remaining channels by alpha
        mlp_features_1 = mlp_features_[..., 1:].float() * alphas.unsqueeze(-1)
        # print("mlp_features_1 shape ", mlp_features_1.shape)
        # 4) Concatenate them back along the last dimension
        mlp_features_ = torch.cat([mlp_features_0, mlp_features_1], dim=-1)
        # print("mlp_features_ shape ", mlp_features_.shape)
        return mlp_features_

def write_patch_to_png(out_img,out_cell_num,out_img_w,j,k,feats, quad_size, out_cell_size, out_feat_num):
    py = out_cell_num//out_img_w
    px = out_cell_num%out_img_w

    osy = j*quad_size
    oey = j*quad_size+out_cell_size
    tsy = py*out_cell_size
    tey = py*out_cell_size+out_cell_size
    osx = k*quad_size
    oex = k*quad_size+out_cell_size
    tsx = px*out_cell_size
    tex = px*out_cell_size+out_cell_size

    for i in range(out_feat_num):
        out_img[i][tsy:tey,tsx:tex] = feats[i][osy:oey,osx:oex]

def get_png_uv(out_cell_num,out_img_w,out_img_size, out_cell_size):
    py = out_cell_num//out_img_w
    px = out_cell_num%out_img_w

    uv0 = np.array([py*out_cell_size+0.5,     px*out_cell_size+0.5],np.float32)/out_img_size
    uv1 = np.array([(py+1)*out_cell_size-0.5, px*out_cell_size+0.5],np.float32)/out_img_size
    uv2 = np.array([py*out_cell_size+0.5,     (px+1)*out_cell_size-0.5],np.float32)/out_img_size
    uv3 = np.array([(py+1)*out_cell_size-0.5, (px+1)*out_cell_size-0.5],np.float32)/out_img_size

    return uv0,uv1,uv2,uv3


@torch.no_grad()
def bake(model: NeRFNetwork, dir:str, train_loader: DataLoader):
    print('starting bake!')
    print(train_loader._data.poses.shape)
    export_pytorch_mlp_explicit(
        model=model.color_net,
        in_dim=model.in_dim_color,
        hidden_dim=model.hidden_dim_color,
        out_dim=3,
        file_path="mlp.json"
    )

    layer_num = model.grid_size
    grid_max = model.grid_max.cpu().numpy()
    grid_min = model.grid_min.cpu().numpy()
    point_grid_size = model.grid_size
    n_device = 1
    point_grid = model.point_grid.detach().clone().cpu()
    acc_grid = model.acc_grid.detach().clone().cpu()
    v_grid = torch.zeros([layer_num+1,layer_num+1,layer_num+1,3]).cpu().float()
    v_grid[:-1,:-1,:-1] = torch.tensor(point_grid)*model.point_grid_diff_lr_scale
    
    texture_size = 1024*2
    batch_num = 8*8*8
    num_bottleneck_features = 8

    test_threshold = 0.1


    out_feat_num = num_bottleneck_features//4

    quad_size = texture_size//layer_num
    assert quad_size*layer_num == texture_size
    
    quad_weights = np.zeros([quad_size,quad_size,4],np.float32)
    
    for i in range(quad_size):
        for j in range(quad_size):
            x = (i)/quad_size
            y = (j)/quad_size
            if x>y:
                quad_weights[i,j,0] = 1-x
                quad_weights[i,j,1] = x-y
                quad_weights[i,j,2] = 0
                quad_weights[i,j,3] = y
            else:
                quad_weights[i,j,0] = 1-y
                quad_weights[i,j,1] = 0
                quad_weights[i,j,2] = y-x
                quad_weights[i,j,3] = x
    quad_weights = np.reshape(quad_weights,[quad_size*quad_size,4])
    quad_weights = np.transpose(quad_weights, (1,0)) #[4,quad_size*quad_size]

    grid_max_np = np.array(grid_max,np.float32)
    grid_min_np = np.array(grid_min,np.float32)

    i_grid = np.zeros([layer_num,layer_num,layer_num],np.int32)
    j_grid = np.zeros([layer_num,layer_num,layer_num],np.int32)
    k_grid = np.zeros([layer_num,layer_num,layer_num],np.int32)

    i_grid[:,:,:] = np.reshape(np.arange(layer_num),[-1,1,1])
    j_grid[:,:,:] = np.reshape(np.arange(layer_num),[1,-1,1])
    k_grid[:,:,:] = np.reshape(np.arange(layer_num),[1,1,-1])
    
    if os.path.exists(dir+"/out_img.pth"):
        print("skipping loading")
    elif os.path.exists(dir+"/z_plane.pth"):
        buffer_z = torch.load(dir+"/z_plane.pth")
        print("Loaded from cache")
    else:
        print("Computing z plane")

        ##### z planes

        x,y,z = j_grid,k_grid,i_grid
        p0 = v_grid[x,y,z] + (np.stack([x,y,z],axis=-1).astype(np.float32)+0.5)*((grid_max_np - grid_min_np)/point_grid_size) + grid_min_np
        x,y,z = j_grid+1,k_grid,i_grid
        p1 = v_grid[x,y,z] + (np.stack([x,y,z],axis=-1).astype(np.float32)+0.5)*((grid_max_np - grid_min_np)/point_grid_size) + grid_min_np
        x,y,z = j_grid,k_grid+1,i_grid
        p2 = v_grid[x,y,z] + (np.stack([x,y,z],axis=-1).astype(np.float32)+0.5)*((grid_max_np - grid_min_np)/point_grid_size) + grid_min_np
        x,y,z = j_grid+1,k_grid+1,i_grid
        p3 = v_grid[x,y,z] + (np.stack([x,y,z],axis=-1).astype(np.float32)+0.5)*((grid_max_np - grid_min_np)/point_grid_size) + grid_min_np
        p0123 = np.stack([p0,p1,p2,p3],axis=-1) #[M,N,K,3,4]
        p0123 = p0123 @ quad_weights #[M,N,K,3,quad_size*quad_size]
        p0123 = np.reshape(p0123, [layer_num,layer_num,layer_num,3,quad_size,quad_size]) #[M,N,K,3,quad_size,quad_size]
        p0123 = np.transpose(p0123, (0,1,4,2,5,3)) #[M,N,quad_size,K,quad_size,3]
        #positions_z = np.reshape(np.ascontiguousarray(p0123), [layer_num,layer_num*quad_size,layer_num*quad_size,3])
        positions_z = np.reshape(np.ascontiguousarray(p0123), [-1,3])

        p0 = None
        p1 = None
        p2 = None
        p3 = None
        p0123 = None

        total_len = len(positions_z)
        batch_len = total_len//batch_num
        coarse_feature_z = np.zeros([total_len,num_bottleneck_features],np.uint8)
        for i in range(batch_num):
            t0 = np.reshape(positions_z[i*batch_len:(i+1)*batch_len], [n_device,-1,3])
            t0 = get_density_color(t0, acc_grid, model, test_threshold)
            #for some reason, i thought the alpha channel was included in the 8 feature vector
            # zero_channel = torch.zeros(
            #     (1, t0.shape[1], 1), 
            #     dtype=t0.dtype, 
            #     device=t0.device
            # )
            # t0 = torch.cat([t0, zero_channel], dim=-1)
            # # print("t0 shape ", t0.shape)
            coarse_feature_z[i*batch_len:(i+1)*batch_len] = np.reshape(t0.cpu(),[-1,num_bottleneck_features])
        coarse_feature_z = np.reshape(coarse_feature_z,[layer_num,texture_size,texture_size,num_bottleneck_features])
        coarse_feature_z[:,-quad_size:,:] = 0
        coarse_feature_z[:,:,-quad_size:] = 0
        # print("coarse shape", coarse_feature_z.shape)

        positions_z = None

        buffer_z = []
        for i in range(layer_num):
            if not np.any(coarse_feature_z[i,:,:,0]>0):
                buffer_z.append(None)
                continue
            feats = get_feature_png(coarse_feature_z[i], out_feat_num)
            buffer_z.append(feats)

        coarse_feature_z = None
        print("saving z plane")
        torch.save(buffer_z, dir+"/z_plane.pth")
    if os.path.exists(dir+"/out_img.pth"):
        print("skipping loading")
    elif os.path.exists(dir+"/x_plane.pth"):
        buffer_x = torch.load(dir+"/x_plane.pth")
        print("Loaded from cache")
    else:
        print("Computing x plane")
        ##### x planes

        x,y,z = i_grid,j_grid,k_grid
        p0 = v_grid[x,y,z] + (np.stack([x,y,z],axis=-1).astype(np.float32)+0.5)*((grid_max_np - grid_min_np)/point_grid_size) + grid_min_np
        x,y,z = i_grid,j_grid+1,k_grid
        p1 = v_grid[x,y,z] + (np.stack([x,y,z],axis=-1).astype(np.float32)+0.5)*((grid_max_np - grid_min_np)/point_grid_size) + grid_min_np
        x,y,z = i_grid,j_grid,k_grid+1
        p2 = v_grid[x,y,z] + (np.stack([x,y,z],axis=-1).astype(np.float32)+0.5)*((grid_max_np - grid_min_np)/point_grid_size) + grid_min_np
        x,y,z = i_grid,j_grid+1,k_grid+1
        p3 = v_grid[x,y,z] + (np.stack([x,y,z],axis=-1).astype(np.float32)+0.5)*((grid_max_np - grid_min_np)/point_grid_size) + grid_min_np
        p0123 = np.stack([p0,p1,p2,p3],axis=-1) #[M,N,K,3,4]
        p0123 = p0123 @ quad_weights #[M,N,K,3,quad_size*quad_size]
        p0123 = np.reshape(p0123, [layer_num,layer_num,layer_num,3,quad_size,quad_size]) #[M,N,K,3,quad_size,quad_size]
        p0123 = np.transpose(p0123, (0,1,4,2,5,3)) #[M,N,quad_size,K,quad_size,3]
        #positions_x = np.reshape(np.ascontiguousarray(p0123), [layer_num,layer_num*quad_size,layer_num*quad_size,3])
        positions_x = np.reshape(np.ascontiguousarray(p0123), [-1,3])

        p0 = None
        p1 = None
        p2 = None
        p3 = None
        p0123 = None

        total_len = len(positions_x)
        batch_len = total_len//batch_num
        coarse_feature_x = np.zeros([total_len,num_bottleneck_features],np.uint8)
        for i in range(batch_num):
            t0 = np.reshape(positions_x[i*batch_len:(i+1)*batch_len], [n_device,-1,3])
            t0 = get_density_color(t0, acc_grid, model, test_threshold)
            # zero_channel = torch.zeros(
            #     (1, t0.shape[1], 1), 
            #     dtype=t0.dtype, 
            #     device=t0.device
            # )
            # t0 = torch.cat([t0, zero_channel], dim=-1)
            coarse_feature_x[i*batch_len:(i+1)*batch_len] = np.reshape(t0.cpu(),[-1,num_bottleneck_features])
        coarse_feature_x = np.reshape(coarse_feature_x,[layer_num,texture_size,texture_size,num_bottleneck_features])
        coarse_feature_x[:,-quad_size:,:] = 0
        coarse_feature_x[:,:,-quad_size:] = 0

        positions_x = None

        buffer_x = []
        for i in range(layer_num):
            if not np.any(coarse_feature_x[i,:,:,0]>0):
                buffer_x.append(None)
                continue
            feats = get_feature_png(coarse_feature_x[i], out_feat_num)
            buffer_x.append(feats)

        coarse_feature_x = None
        print("saving x plane")
        torch.save(buffer_x, dir+"/x_plane.pth")
    if os.path.exists(dir+"/out_img.pth"):
        print("skipping loading")
    elif os.path.exists(dir+"/y_plane.pth"):
        buffer_y = torch.load(dir+"/y_plane.pth")
        print("Loaded from cache")
    else:
        print("Computing y plane")


        ##### y planes

        x,y,z = j_grid,i_grid,k_grid
        p0 = v_grid[x,y,z] + (np.stack([x,y,z],axis=-1).astype(np.float32)+0.5)*((grid_max_np - grid_min_np)/point_grid_size) + grid_min_np
        x,y,z = j_grid+1,i_grid,k_grid
        p1 = v_grid[x,y,z] + (np.stack([x,y,z],axis=-1).astype(np.float32)+0.5)*((grid_max_np - grid_min_np)/point_grid_size) + grid_min_np
        x,y,z = j_grid,i_grid,k_grid+1
        p2 = v_grid[x,y,z] + (np.stack([x,y,z],axis=-1).astype(np.float32)+0.5)*((grid_max_np - grid_min_np)/point_grid_size) + grid_min_np
        x,y,z = j_grid+1,i_grid,k_grid+1
        p3 = v_grid[x,y,z] + (np.stack([x,y,z],axis=-1).astype(np.float32)+0.5)*((grid_max_np - grid_min_np)/point_grid_size) + grid_min_np
        p0123 = np.stack([p0,p1,p2,p3],axis=-1) #[M,N,K,3,4]
        p0123 = p0123 @ quad_weights #[M,N,K,3,quad_size*quad_size]
        p0123 = np.reshape(p0123, [layer_num,layer_num,layer_num,3,quad_size,quad_size]) #[M,N,K,3,quad_size,quad_size]
        p0123 = np.transpose(p0123, (0,1,4,2,5,3)) #[M,N,quad_size,K,quad_size,3]
        #positions_y = np.reshape(np.ascontiguousarray(p0123), [layer_num,layer_num*quad_size,layer_num*quad_size,3])
        positions_y = np.reshape(np.ascontiguousarray(p0123), [-1,3])

        p0 = None
        p1 = None
        p2 = None
        p3 = None
        p0123 = None

        total_len = len(positions_y)
        batch_len = total_len//batch_num
        coarse_feature_y = np.zeros([total_len,num_bottleneck_features],np.uint8)
        for i in range(batch_num):
            t0 = np.reshape(positions_y[i*batch_len:(i+1)*batch_len], [n_device,-1,3])
            t0 = get_density_color(t0, acc_grid, model, test_threshold)
            # zero_channel = torch.zeros(
            #     (1, t0.shape[1], 1), 
            #     dtype=t0.dtype, 
            #     device=t0.device
            # )
            # t0 = torch.cat([t0, zero_channel], dim=-1)
            coarse_feature_y[i*batch_len:(i+1)*batch_len] = np.reshape(t0.cpu(),[-1,num_bottleneck_features])
        coarse_feature_y = np.reshape(coarse_feature_y,[layer_num,texture_size,texture_size,num_bottleneck_features])
        coarse_feature_y[:,-quad_size:,:] = 0
        coarse_feature_y[:,:,-quad_size:] = 0

        positions_y = None

        buffer_y = []
        for i in range(layer_num):
            if not np.any(coarse_feature_y[i,:,:,0]>0):
                buffer_y.append(None)
                continue
            feats = get_feature_png(coarse_feature_y[i], out_feat_num)
            buffer_y.append(feats)

        coarse_feature_y = None
        print("saving y plane")
        torch.save(buffer_y, dir+"/y_plane.pth")

    #%%
    # write_floatpoint_image(dir+"/s3_slice_sample.png",buffer_z[layer_num//2][0]/255.0)
    #%%
    out_img_size = 1024*20
    out_cell_num = 0
    out_cell_size = quad_size+1
    out_img_h = out_img_size//out_cell_size
    out_img_w = out_img_size//out_cell_size
    if os.path.exists(dir+"/out_img.pth"):
        out_img = torch.load(dir+"/out_img.pth")
        print("Loaded from cache")
        point_UV_grid = torch.load(dir+"/point_UV_grid.pth")
        print("Loaded from cache")
        bag_of_v = torch.load(dir+"/bag_of_v.pth")
        print("Loaded from cache")
        out_cell_num = torch.load(dir+"/out_cell_num.pth")
        print("Loaded from cache")
    else:
        print("Computing output image")
        out_img = []
        for i in range(out_feat_num):
            out_img.append(np.zeros([out_img_size,out_img_size,4], np.uint8))
        
        #for eval
        point_UV_grid = np.zeros([point_grid_size,point_grid_size,point_grid_size,3,4,2], np.float32)

        #mesh vertices
        bag_of_v = []
        
        for k in range(layer_num-1,-1,-1):
            for i in range(layer_num):
                for j in range(layer_num):

                    # z plane
                    if not(k==0 or k==layer_num-1 or i==layer_num-1 or j==layer_num-1):
                        feats = buffer_z[k]
                        if feats is not None and np.max(feats[0][i*quad_size:(i+1)*quad_size+1,j*quad_size:(j+1)*quad_size+1,2])>0:

                            write_patch_to_png(out_img,out_cell_num,out_img_w,i,j,feats, quad_size, out_cell_size, out_feat_num)
                            uv0,uv1,uv2,uv3 = get_png_uv(out_cell_num,out_img_w,out_img_size, out_cell_size)
                            out_cell_num += 1

                            p0 = v_grid[i,j,k] + (np.array([i,j,k],np.float32)+0.5)*((grid_max_np - grid_min_np)/point_grid_size) + grid_min_np
                            p1 = v_grid[i+1,j,k] + (np.array([i+1,j,k],np.float32)+0.5)*((grid_max_np - grid_min_np)/point_grid_size) + grid_min_np
                            p2 = v_grid[i,j+1,k] + (np.array([i,j+1,k],np.float32)+0.5)*((grid_max_np - grid_min_np)/point_grid_size) + grid_min_np
                            p3 = v_grid[i+1,j+1,k] + (np.array([i+1,j+1,k],np.float32)+0.5)*((grid_max_np - grid_min_np)/point_grid_size) + grid_min_np

                            p0 = inverse_taper_coord_np(p0)
                            p1 = inverse_taper_coord_np(p1)
                            p2 = inverse_taper_coord_np(p2)
                            p3 = inverse_taper_coord_np(p3)

                            point_UV_grid[i,j,k,2,0] = uv0
                            point_UV_grid[i,j,k,2,1] = uv1
                            point_UV_grid[i,j,k,2,2] = uv2
                            point_UV_grid[i,j,k,2,3] = uv3

                            bag_of_v.append([p0,p1,p2,p3])

                    # x plane
                    if not(i==0 or i==layer_num-1 or j==layer_num-1 or k==layer_num-1):
                        feats = buffer_x[i]
                        if feats is not None and np.max(feats[0][j*quad_size:(j+1)*quad_size+1,k*quad_size:(k+1)*quad_size+1,2])>0:

                            write_patch_to_png(out_img,out_cell_num,out_img_w,j,k,feats, quad_size, out_cell_size, out_feat_num)
                            uv0,uv1,uv2,uv3 = get_png_uv(out_cell_num,out_img_w,out_img_size, out_cell_size)
                            out_cell_num += 1

                            p0 = v_grid[i,j,k] + (np.array([i,j,k],np.float32)+0.5)*((grid_max_np - grid_min_np)/point_grid_size) + grid_min_np
                            p1 = v_grid[i,j+1,k] + (np.array([i,j+1,k],np.float32)+0.5)*((grid_max_np - grid_min_np)/point_grid_size) + grid_min_np
                            p2 = v_grid[i,j,k+1] + (np.array([i,j,k+1],np.float32)+0.5)*((grid_max_np - grid_min_np)/point_grid_size) + grid_min_np
                            p3 = v_grid[i,j+1,k+1] + (np.array([i,j+1,k+1],np.float32)+0.5)*((grid_max_np - grid_min_np)/point_grid_size) + grid_min_np

                            p0 = inverse_taper_coord_np(p0)
                            p1 = inverse_taper_coord_np(p1)
                            p2 = inverse_taper_coord_np(p2)
                            p3 = inverse_taper_coord_np(p3)

                            point_UV_grid[i,j,k,0,0] = uv0
                            point_UV_grid[i,j,k,0,1] = uv1
                            point_UV_grid[i,j,k,0,2] = uv2
                            point_UV_grid[i,j,k,0,3] = uv3

                            bag_of_v.append([p0,p1,p2,p3])

                    # y plane
                    if not(j==0 or j==layer_num-1 or i==layer_num-1 or k==layer_num-1):
                        feats = buffer_y[j]
                        if feats is not None and np.max(feats[0][i*quad_size:(i+1)*quad_size+1,k*quad_size:(k+1)*quad_size+1,2])>0:

                            write_patch_to_png(out_img,out_cell_num,out_img_w,i,k,feats, quad_size, out_cell_size, out_feat_num)
                            uv0,uv1,uv2,uv3 = get_png_uv(out_cell_num,out_img_w,out_img_size, out_cell_size)
                            out_cell_num += 1

                            p0 = v_grid[i,j,k] + (np.array([i,j,k],np.float32)+0.5)*((grid_max_np - grid_min_np)/point_grid_size) + grid_min_np
                            p1 = v_grid[i+1,j,k] + (np.array([i+1,j,k],np.float32)+0.5)*((grid_max_np - grid_min_np)/point_grid_size) + grid_min_np
                            p2 = v_grid[i,j,k+1] + (np.array([i,j,k+1],np.float32)+0.5)*((grid_max_np - grid_min_np)/point_grid_size) + grid_min_np
                            p3 = v_grid[i+1,j,k+1] + (np.array([i+1,j,k+1],np.float32)+0.5)*((grid_max_np - grid_min_np)/point_grid_size) + grid_min_np

                            p0 = inverse_taper_coord_np(p0)
                            p1 = inverse_taper_coord_np(p1)
                            p2 = inverse_taper_coord_np(p2)
                            p3 = inverse_taper_coord_np(p3)

                            point_UV_grid[i,j,k,1,0] = uv0
                            point_UV_grid[i,j,k,1,1] = uv1
                            point_UV_grid[i,j,k,1,2] = uv2
                            point_UV_grid[i,j,k,1,3] = uv3

                            bag_of_v.append([p0,p1,p2,p3])
        print("saving output image")
        torch.save(out_img, dir+"/out_img.pth")
        torch.save(point_UV_grid, dir+"/point_UV_grid.pth")
        torch.save(bag_of_v, dir+"/bag_of_v.pth")
        torch.save(out_cell_num, dir+"/out_cell_num.pth")
    
        
    
    print("Number of quad faces:", out_cell_num)
    buffer_x = None
    buffer_y = None
    buffer_z = None
    texture_alpha = np.zeros([out_img_size,out_img_size,1], np.uint8)
    texture_features = np.zeros([out_img_size,out_img_size,8], np.uint8)

    texture_alpha[:,:,0] = (out_img[0][:,:,2]>0)

    texture_features[:,:,0:3] = out_img[0][:,:,2::-1]
    texture_features[:,:,3] = out_img[0][:,:,3]
    texture_features[:,:,4:7] = out_img[1][:,:,2::-1]
    texture_features[:,:,7] = out_img[1][:,:,3]
    
    selected_test_index = 97
    preview_image_height = 800
    dataset = train_loader._data  # Because loader._data = self in NeRFDataset
    render_poses = dataset.poses  # shape [N, 4, 4]
    intrinsics = dataset.intrinsics  # np.array([fx, fy, cx, cy])
    H, W = dataset.H, dataset.W
    batch_size = 8192
    
   
    
    texture_mask = np.zeros([out_img_size,out_img_size], np.uint8)
    print("Removing invisible triangles")
    iterator = tqdm(render_poses[:15])
    for p in iterator:
       
        p = p.unsqueeze(0)
        rays = get_rays(p, intrinsics, H, W, N=-1, error_map=None, patch_size=1, supersampling = True)
        rays_o_tot = rays['rays_o']
        rays_d_tot = rays['rays_d']
        for a in tqdm(range(0, rays_o_tot.shape[1], batch_size)):
            rays_o = rays_o_tot[:, a:a+batch_size]
            rays_d = rays_d_tot[:, a:a+batch_size]
            print("rays_o shape ", rays_o.shape)
            print("rays_d shape ", rays_d.shape)
            print(point_grid_size)
            #---------- ray-plane intersection points
            grid_indices, grid_masks = gridcell_from_rays_bake(rays_o, rays_d, model.grid_min.detach().clone().cpu(), model.grid_max.detach().clone().cpu(), point_grid_size)
            grid_indices = grid_indices.squeeze(0)
            grid_masks = grid_masks.squeeze(0)
            print("grid_indices shape ", grid_indices.shape)
            print("grid_masks shape ", grid_masks.shape)
            print("Point UV grid shape ", point_UV_grid.shape)
            print("Texture alpha shape ", texture_alpha.shape)
            
            N = grid_indices.shape[0]
            world_alpha, world_uv = compute_undc_intersection_and_return_uv(point_grid, 
                                                                            torch.tensor(point_UV_grid),
                                                                            torch.tensor(texture_alpha),
                                                                            grid_indices,
                                                                            grid_masks,
                                                                            rays_o,
                                                                            rays_d,
                                                                            model.grid_min.detach().clone().cpu(),
                                                                            model.grid_max.detach().clone().cpu(),
                                                                            point_grid_size,
                                                                            model.point_grid_diff_lr_scale,
                                                                            model.cell_size_x.detach().clone().cpu(),
                                                                            model.cell_size_y.detach().clone().cpu(),
                                                                            model.cell_size_z.detach().clone().cpu(),
                                                                            out_img_size,
            )
            world_alpha = world_alpha.float()
            print("world_alpha shape ", world_alpha.shape)
            print("world_uv shape ", world_uv.shape)

            # Now use the MLP to compute density and features
            mlp_alpha_b = world_alpha[..., 0]  # [N, 4, P]
            print("mlp_alpha_b shape ", mlp_alpha_b.shape)
            weights_b = compute_volumetric_rendering_weights_with_alpha(mlp_alpha_b)  # [N, 4, P]
            print("weights_b shape ", weights_b.shape)

            # 2) Compute 'acc_b' and argmax index along last dimension
            acc_b = (weights_b.sum(dim=-1) > 0.5)  # [N, 4]
            print("acc_b shape ", acc_b.shape)
            ind = weights_b.argmax(dim=-1, keepdim=True)  # [N, 4, 1]
            print("ind shape ", ind.shape)

            # 3) Gather (like take_along_axis) from world_uv along dim=-2 => the 'P' dimension
            index_expanded = ind.unsqueeze(-1).expand(-1, -1, -1, world_uv.size(-1))  
            print("index_expanded shape ", index_expanded.shape)
            # index_expanded shape: [N, 4, 1, 2]

            selected_uv = torch.gather(world_uv, dim=-2, index=index_expanded)  # [N, 4, 1, 2]
            print("selected_uv shape ", selected_uv.shape)

            # 4) Remove the extra dimension and mask out via acc_b
            selected_uv = selected_uv[..., 0, :] * acc_b.unsqueeze(-1)  # => [N, 4, 2]
            selected_uv = selected_uv.squeeze(0)
            print("selected_uv shape ", selected_uv.shape)
            
            selected_uv = selected_uv.numpy()
            mlp_features_b = torch.tensor(texture_features[selected_uv[...,0],selected_uv[...,1]])

            mlp_features_b = mlp_features_b.float().div(255.)
            acc_b = acc_b.squeeze(0)
            mlp_features_b *= acc_b.unsqueeze(-1)              # [N,4,C]
            print("mlp_features_b shape ", mlp_features_b.shape)

            # 2) Average across the 4-sample dimension => [N,C]
            # mlp_features_b = mlp_features_b.mean(dim=1)        # [N,C]
            mlp_features_b = mlp_features_b.squeeze(0)
            print("mlp_features_b shape ", mlp_features_b.shape)

            # 3) Compute average acc_b
            # acc_b = acc_b.float().mean(dim=-1)                 # [N]
            acc_b = acc_b.float()
            print("acc_b shape ", acc_b.shape)

            # 4) Normalize directions, then average => [N,3]
            dirs = rays_d.squeeze(0)
            # dirs = F.normalize(rays_d, dim=-1).mean(dim=1)
            print("dirs shape ", dirs.shape)

            # 5) Concatenate features + directions => [N, C+3]
            features_dirs_enc_b = torch.cat([mlp_features_b, dirs], dim=-1)
            features_dirs_enc_b = features_dirs_enc_b.contiguous().to(device)
            print("features_dirs_enc_b shape ", features_dirs_enc_b.shape)

            # 6) Predict color with a simple MLP + sigmoid
            rgb_b = model.just_color(features_dirs_enc_b.view(-1, 11))  # [N,3]
            rgb_b = rgb_b.view(N, 3).cpu()
            # White background
            
            rgb_b = rgb_b * acc_b.unsqueeze(-1) + (1. - acc_b).unsqueeze(-1)
            
            texture_mask[selected_uv[...,0],selected_uv[...,1]] = 1
            
        
        

    num_visible_quads = 0

    quad_t1_mask = np.zeros([out_cell_size,out_cell_size],np.uint8)
    quad_t2_mask = np.zeros([out_cell_size,out_cell_size],np.uint8)
    for i in range(out_cell_size):
        for j in range(out_cell_size):
            if i>=j:
                quad_t1_mask[i,j] = 1
            if i<=j:
                quad_t2_mask[i,j] = 1 
    def check_triangle_visible(mask,out_cell_num):
        py = out_cell_num//out_img_w
        px = out_cell_num%out_img_w

        tsy = py*out_cell_size
        tey = py*out_cell_size+out_cell_size
        tsx = px*out_cell_size
        tex = px*out_cell_size+out_cell_size

        quad_m = mask[tsy:tey,tsx:tex]
        t1_visible = np.any(quad_m*quad_t1_mask)
        t2_visible = np.any(quad_m*quad_t2_mask)

        return (t1_visible or t2_visible), t1_visible, t2_visible

    def mask_triangle_invisible(mask,out_cell_num,imga):
        py = out_cell_num//out_img_w
        px = out_cell_num%out_img_w

        tsy = py*out_cell_size
        tey = py*out_cell_size+out_cell_size
        tsx = px*out_cell_size
        tex = px*out_cell_size+out_cell_size

        quad_m = mask[tsy:tey,tsx:tex]
        t1_visible = np.any(quad_m*quad_t1_mask)
        t2_visible = np.any(quad_m*quad_t2_mask)

        if not (t1_visible or t2_visible):
            imga[tsy:tey,tsx:tex] = 0

        elif not t1_visible:
            imga[tsy:tey,tsx:tex] = imga[tsy:tey,tsx:tex]*quad_t2_mask[:,:,None]

        elif not t2_visible:
            imga[tsy:tey,tsx:tex] = imga[tsy:tey,tsx:tex]*quad_t1_mask[:,:,None]

        return (t1_visible or t2_visible), t1_visible, t2_visible

    
    for i in range(out_cell_num):
        quad_visible, t1_visible, t2_visible = mask_triangle_invisible(texture_mask, i, texture_alpha)
        if quad_visible:
            num_visible_quads += 1

    print("Number of quad faces:", num_visible_quads)
    
    new_img_sizes = [
    [1024,1024],
    [2048,1024],
    [2048,2048],
    [4096,2048],
    [4096,4096],
    [8192,4096],
    [8192,8192],
    [16384,8192],
    [16384,16384],
    ]

    fit_flag = False
    for i in range(len(new_img_sizes)):
        new_img_size_w,new_img_size_h = new_img_sizes[i]
        new_img_size_ratio = new_img_size_w/new_img_size_h
        new_img_h = new_img_size_h//out_cell_size
        new_img_w = new_img_size_w//out_cell_size
        if num_visible_quads<=new_img_h*new_img_w:
            fit_flag = True
            break

    if fit_flag:
        print("Texture image size:", new_img_size_w,new_img_size_h)
    else:
        print("Texture image too small", new_img_size_w,new_img_size_h)
        1/0


    new_img = []
    for i in range(out_feat_num):
        new_img.append(np.zeros([new_img_size_h,new_img_size_w,4], np.uint8))
    new_cell_num = 0


    def copy_patch_to_png(out_img,out_cell_num,new_img,new_cell_num):
        py = out_cell_num//out_img_w
        px = out_cell_num%out_img_w

        ny = new_cell_num//new_img_w
        nx = new_cell_num%new_img_w

        tsy = py*out_cell_size
        tey = py*out_cell_size+out_cell_size
        tsx = px*out_cell_size
        tex = px*out_cell_size+out_cell_size
        nsy = ny*out_cell_size
        ney = ny*out_cell_size+out_cell_size
        nsx = nx*out_cell_size
        nex = nx*out_cell_size+out_cell_size

        for i in range(out_feat_num):
            new_img[i][nsy:ney,nsx:nex] = out_img[i][tsy:tey,tsx:tex]

        return True



    #write mesh

    obj_save_dir = "test"
    if not os.path.exists(obj_save_dir):
        os.makedirs(obj_save_dir)

    obj_f = open(obj_save_dir+"/shape.obj",'w')

    vcount = 0

    for i in range(out_cell_num):
        quad_visible, t1_visible, t2_visible = check_triangle_visible(texture_mask, i)
        if quad_visible:
            copy_patch_to_png(out_img,i,new_img,new_cell_num)
            p0,p1,p2,p3 = bag_of_v[i]
            uv0,uv1,uv2,uv3 = get_png_uv(new_cell_num,new_img_w,new_img_size_w, out_cell_size)
            new_cell_num += 1

            obj_f.write("v %.6f %.6f %.6f\n" % (p0[0],p0[2],-p0[1]))
            obj_f.write("v %.6f %.6f %.6f\n" % (p1[0],p1[2],-p1[1]))
            obj_f.write("v %.6f %.6f %.6f\n" % (p2[0],p2[2],-p2[1]))
            obj_f.write("v %.6f %.6f %.6f\n" % (p3[0],p3[2],-p3[1]))


            obj_f.write("vt %.6f %.6f\n" % (uv0[1],1-uv0[0]*new_img_size_ratio))
            obj_f.write("vt %.6f %.6f\n" % (uv1[1],1-uv1[0]*new_img_size_ratio))
            obj_f.write("vt %.6f %.6f\n" % (uv2[1],1-uv2[0]*new_img_size_ratio))
            obj_f.write("vt %.6f %.6f\n" % (uv3[1],1-uv3[0]*new_img_size_ratio))
            if t1_visible:
                obj_f.write("f %d/%d %d/%d %d/%d\n" % (vcount+1,vcount+1,vcount+2,vcount+2,vcount+4,vcount+4))
            if t2_visible:
                obj_f.write("f %d/%d %d/%d %d/%d\n" % (vcount+1,vcount+1,vcount+4,vcount+4,vcount+3,vcount+3))
            vcount += 4

    for j in range(out_feat_num):
        cv2.imwrite(obj_save_dir+"/shape.pngfeat"+str(j)+".png", new_img[j], [cv2.IMWRITE_PNG_COMPRESSION, 9])
    obj_f.close()
    
    
    target_dir = obj_save_dir+"_phone"

    texture_size = 4096
    patchsize = 17
    texture_patch_size = texture_size//patchsize

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)


    source_obj_dir = obj_save_dir+"/shape.obj"
    source_png0_dir = obj_save_dir+"/shape.pngfeat0.png"
    source_png1_dir = obj_save_dir+"/shape.pngfeat1.png"

    source_png0 = cv2.imread(source_png0_dir,cv2.IMREAD_UNCHANGED)
    source_png1 = cv2.imread(source_png1_dir,cv2.IMREAD_UNCHANGED)

    img_h,img_w,_ = source_png0.shape



    num_splits = 0 #this is a counter


    fin = open(source_obj_dir,'r')
    lines = fin.readlines()
    fin.close()



    current_img_idx = 0
    current_img0 = np.zeros([texture_size,texture_size,4],np.uint8)
    current_img1 = np.zeros([texture_size,texture_size,4],np.uint8)
    current_quad_count = 0
    current_obj = open(target_dir+"/shape"+str(current_img_idx)+".obj",'w')
    current_v_count = 0
    current_v_offset = 0

    #v-vt-f cycle

    for i in range(len(lines)):
        line = lines[i].split()
        if len(line)==0:
            continue

        elif line[0] == 'v':
            current_obj.write(lines[i])
            current_v_count += 1

        elif line[0] == 'vt':
            if lines[i-1].split()[0] == "v":

                line = lines[i].split()
                x0 = float(line[1])
                y0 = 1-float(line[2])

                line = lines[i+1].split()
                x1 = float(line[1])
                y1 = 1-float(line[2])

                line = lines[i+2].split()
                x2 = float(line[1])
                y2 = 1-float(line[2])

                line = lines[i+3].split()
                x3 = float(line[1])
                y3 = 1-float(line[2])

                xc = (x0+x1+x2+x3)*img_w/4
                yc = (y0+y1+y2+y3)*img_h/4

                old_cell_x = int(xc/patchsize)
                old_cell_y = int(yc/patchsize)

                new_cell_x = current_quad_count%texture_patch_size
                new_cell_y = current_quad_count//texture_patch_size
                current_quad_count += 1

                #copy patch

                tsy = old_cell_y*patchsize
                tey = old_cell_y*patchsize+patchsize
                tsx = old_cell_x*patchsize
                tex = old_cell_x*patchsize+patchsize
                nsy = new_cell_y*patchsize
                ney = new_cell_y*patchsize+patchsize
                nsx = new_cell_x*patchsize
                nex = new_cell_x*patchsize+patchsize

                current_img0[nsy:ney,nsx:nex] = source_png0[tsy:tey,tsx:tex]
                current_img1[nsy:ney,nsx:nex] = source_png1[tsy:tey,tsx:tex]

                #write uv

                uv0_y = (new_cell_y*patchsize+0.5)/texture_size
                uv0_x = (new_cell_x*patchsize+0.5)/texture_size

                uv1_y = ((new_cell_y+1)*patchsize-0.5)/texture_size
                uv1_x = (new_cell_x*patchsize+0.5)/texture_size

                uv2_y = (new_cell_y*patchsize+0.5)/texture_size
                uv2_x = ((new_cell_x+1)*patchsize-0.5)/texture_size

                uv3_y = ((new_cell_y+1)*patchsize-0.5)/texture_size
                uv3_x = ((new_cell_x+1)*patchsize-0.5)/texture_size

                current_obj.write("vt %.6f %.6f\n" % (uv0_x,1-uv0_y))
                current_obj.write("vt %.6f %.6f\n" % (uv1_x,1-uv1_y))
                current_obj.write("vt %.6f %.6f\n" % (uv2_x,1-uv2_y))
                current_obj.write("vt %.6f %.6f\n" % (uv3_x,1-uv3_y))


        elif line[0] == 'f':
            f1 = int(line[1].split("/")[0])-current_v_offset
            f2 = int(line[2].split("/")[0])-current_v_offset
            f3 = int(line[3].split("/")[0])-current_v_offset
            current_obj.write("f %d/%d %d/%d %d/%d\n" % (f1,f1,f2,f2,f3,f3))

            #create new texture image if current is fill
            if i==len(lines)-1 or (lines[i+1].split()[0]!='f' and current_quad_count==texture_patch_size*texture_patch_size):
                current_obj.close()

                # the following is only required for iphone
                # because iphone runs alpha test before the fragment shader
                # the viewer code is also changed accordingly
                current_img0[:,:,3] = current_img0[:,:,3]//2+128
                current_img1[:,:,3] = current_img1[:,:,3]//2+128

                cv2.imwrite(target_dir+"/shape"+str(current_img_idx)+".pngfeat0.png", current_img0, [cv2.IMWRITE_PNG_COMPRESSION,9])
                cv2.imwrite(target_dir+"/shape"+str(current_img_idx)+".pngfeat1.png", current_img1, [cv2.IMWRITE_PNG_COMPRESSION,9])
                current_img_idx += 1
                current_img0 = np.zeros([texture_size,texture_size,4],np.uint8)
                current_img1 = np.zeros([texture_size,texture_size,4],np.uint8)
                current_quad_count = 0
                if i!=len(lines)-1:
                    current_obj = open(target_dir+"/shape"+str(current_img_idx)+".obj",'w')
                current_v_offset += current_v_count
                current_v_count = 0