import torch

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def get_acc_grid_masks(
    taper_positions: torch.Tensor,
    acc_grid: torch.Tensor,
    grid_min: torch.Tensor,
    grid_max: torch.Tensor,
    point_grid_size: int
):
    """
    Parameters:
    -----------
    taper_positions : (N, 3) float tensor of 3D points in world coordinates
    acc_grid        : (point_grid_size, point_grid_size, point_grid_size)
                      or (D, H, W) 3D tensor containing occupancy or mask values
    grid_min        : (3,) float tensor, min corner of bounding box
    grid_max        : (3,) float tensor, max corner of bounding box
    point_grid_size : int, the size of the grid in each dimension

    Returns:
    --------
    acc_grid_masks  : (N,) float/bool tensor, the mask values looked up from acc_grid
                      at each taper_position. If position is out of bounds, value is 0.
    """
    device = taper_positions.device
    dtype  = taper_positions.dtype

    # 1) Convert positions from world-space to grid coordinates
    grid_positions = (taper_positions - grid_min) * (
        point_grid_size / (grid_max - grid_min)
    )

    # 2) Create a boolean mask for valid positions in [1, point_grid_size - 1]
    grid_masks = (
        (grid_positions[..., 0] >= 1) & (grid_positions[..., 0] < point_grid_size - 1) &
        (grid_positions[..., 1] >= 1) & (grid_positions[..., 1] < point_grid_size - 1) &
        (grid_positions[..., 2] >= 1) & (grid_positions[..., 2] < point_grid_size - 1)
    )

    # 3) Zero out invalid positions (so we don't get out-of-bounds indices)
    grid_positions = grid_positions * grid_masks.unsqueeze(-1)

    # 4) Convert to integer indices
    grid_indices = grid_positions.long()  # shape (N, 3)

    # 5) Use advanced indexing into acc_grid
    idx0 = grid_indices[..., 0]
    idx1 = grid_indices[..., 1]
    idx2 = grid_indices[..., 2]

    acc_grid_masks = acc_grid[idx0, idx1, idx2]  # shape (N,)

    # 6) Zero out any positions that were out-of-bounds
    # print(acc_grid_masks.shape, grid_masks.shape)
    acc_grid_masks = acc_grid_masks * grid_masks

    return acc_grid_masks


def get_barycentric(p1, p2, p3, O, d, epsilon=1e-10):
    """
    Computes barycentric coordinates (a, b, c) for the intersection
    of the ray (O + t*d) with the triangle (p1, p2, p3).

    Returns:
    --------
    a, b, c: Tensors of the same shape as the inputs (broadcasted),
             containing barycentric coordinates.
    mask   : Boolean tensor indicating which points are inside the triangle
             (and denominator != 0).
    """
    device = p1.device
    dtype  = p1.dtype
    # Ensure epsilon is on the same device/dtype:
    eps_t  = torch.tensor(epsilon, device=device, dtype=dtype)

    # Extract components
    r1x = p1[..., 0] - p3[..., 0]
    r1y = p1[..., 1] - p3[..., 1]
    r1z = p1[..., 2] - p3[..., 2]

    r2x = p2[..., 0] - p3[..., 0]
    r2y = p2[..., 1] - p3[..., 1]
    r2z = p2[..., 2] - p3[..., 2]

    p3x = p3[..., 0]
    p3y = p3[..., 1]
    p3z = p3[..., 2]

    Ox = O[..., 0]
    Oy = O[..., 1]
    Oz = O[..., 2]

    dx = d[..., 0]
    dy = d[..., 1]
    dz = d[..., 2]

    # Compute the denominator (for intersection)
    denominator = (
         - dx * r1y * r2z
         + dx * r1z * r2y
         + dy * r1x * r2z
         - dy * r1z * r2x
         - dz * r1x * r2y
         + dz * r1y * r2x
    )

    # Create a mask where the denominator is near zero
    denominator_mask = (torch.abs(denominator) < eps_t)

    # Avoid zero division by adding 1 where denominator is near zero
    # (those points get excluded later by denominator_mask)
    denominator = denominator + denominator_mask.float()

    # Compute a_numerator
    a_numerator = (
          (Ox - p3x) * dy * r2z
        + (p3x - Ox) * dz * r2y
        + (p3y - Oy) * dx * r2z
        + (Oy - p3y) * dz * r2x
        + (Oz - p3z) * dx * r2y
        + (p3z - Oz) * dy * r2x
    )

    # Compute b_numerator
    b_numerator = (
          (p3x - Ox) * dy * r1z
        + (Ox - p3x) * dz * r1y
        + (Oy - p3y) * dx * r1z
        + (p3y - Oy) * dz * r1x
        + (p3z - Oz) * dx * r1y
        + (Oz - p3z) * dy * r1x
    )

    # Barycentric coords
    a = a_numerator / denominator
    b = b_numerator / denominator
    c = 1 - (a + b)

    # The final mask checks:
    # 1) a, b, c >= 0  (inside the triangle)
    # 2) denominator not zero
    mask = (a >= 0) & (b >= 0) & (c >= 0) & (~denominator_mask)

    return a, b, c, mask


def get_taper_coord(p):
    return p  # As-is


def inverse_taper_coord(p):
    return p  # As-is

def inverse_taper_coord_np(p):
    return p  # As-is

def get_inside_cell_mask(
    P, ooxyz,
    half_cell_size_x, half_cell_size_y, half_cell_size_z,
    neg_half_cell_size_x, neg_half_cell_size_y, neg_half_cell_size_z
):
    # Use device/dtype from P
    device = P.device
    dtype  = P.dtype

    P_ = get_taper_coord(P) - ooxyz
    return (
        (P_[..., 0] >= neg_half_cell_size_x) &
        (P_[..., 0] <  half_cell_size_x) &
        (P_[..., 1] >= neg_half_cell_size_y) &
        (P_[..., 1] <  half_cell_size_y) &
        (P_[..., 2] >= neg_half_cell_size_z) &
        (P_[..., 2] <  half_cell_size_z)
    )

def compute_volumetric_rendering_weights_with_alpha(alpha: torch.Tensor) -> torch.Tensor:
    """
    Computes volume-rendering weights (similar to the standard NeRF procedure),
    given alpha (1 - exp(-sigma * delta)) values.

    Parameters:
    -----------
    alpha : (..., T) torch.Tensor
        Alpha values per sample along each ray.

    Returns:
    --------
    weights : (..., T) torch.Tensor
        The volume-rendering weights for each sample.
    """
    # 1) density_exp = 1 - alpha
    density_exp = 1.0 - alpha

    # 2) density_exp_shifted: shift by 1 along the last dimension
    density_exp_shifted = torch.cat([
        torch.ones_like(density_exp[..., :1]),  # leading ones
        density_exp[..., :-1]
    ], dim=-1)

    # 3) trans = cumulative product of density_exp_shifted
    trans = torch.cumprod(density_exp_shifted, dim=-1)

    # 4) weights = alpha * trans
    weights = alpha * trans

    return weights

# -------------------------------------------------------------------
# Main intersection function
# -------------------------------------------------------------------
def compute_undc_intersection(
    point_grid, cell_xyz, masks, rays_o, rays_d, keep_num,
    grid_min, grid_max, point_grid_size,
    cell_size_x, cell_size_y, cell_size_z,
    point_grid_diff_lr_scale
):
    """
    PyTorch version of compute_undc_intersection.
    
    Parameters:
    -----------
    point_grid : (point_grid_size, point_grid_size, point_grid_size, 3) torch.Tensor
        Some 3D grid storing offsets or coordinates at each cell.
    cell_xyz   : (..., 3) torch.Tensor of integer cell indices [x, y, z].
    masks      : (...) torch.BoolTensor indicating which ray-cell combos are valid.
    rays       : tuple of (ray_origins, ray_directions), each of shape (..., 3).
    keep_num   : int, how many intersection points to keep after sorting along the ray.
    
    Returns:
    --------
    taper_positions : (..., keep_num, 3) torch.Tensor
    world_masks     : (..., keep_num) torch.BoolTensor
    ooo_ * masks[..., None] : The origin offsets (or some debug value) times the mask
    world_tx        : (..., keep_num) torch.Tensor, the t-values (detached)
    """
    ray_origins = rays_o
    ray_directions = rays_d
    dtype  = ray_origins.dtype
    device = ray_origins.device

    # shape (...), integer indexing for each dimension
    cell_x = cell_xyz[..., 0].long()
    cell_y = cell_xyz[..., 1].long()
    cell_z = cell_xyz[..., 2].long()

    # Convert cell_xyz to "center" in world coords
    ooxyz = (cell_xyz.to(dtype) + 0.5) * ((grid_max - grid_min) / point_grid_size) + grid_min

    # Gather from point_grid for each cell
    ooo_ = point_grid[cell_x, cell_y, cell_z] * point_grid_diff_lr_scale
    ooo  = inverse_taper_coord(ooo_ + ooxyz)

    # Helper to gather from point_grid + offset + ooxyz -> inverse_taper_coord
    def gather_and_inverse_offset(ix, iy, iz, offset):
        coords = point_grid[ix, iy, iz] * point_grid_diff_lr_scale + offset + ooxyz
        return inverse_taper_coord(coords)

    off_xp = torch.tensor([ cell_size_x,     0,           0], device=device, dtype=dtype)
    off_xn = torch.tensor([-cell_size_x,     0,           0], device=device, dtype=dtype)
    off_yp = torch.tensor([     0,      cell_size_y,      0], device=device, dtype=dtype)
    off_yn = torch.tensor([     0,     -cell_size_y,      0], device=device, dtype=dtype)
    off_zp = torch.tensor([     0,           0,     cell_size_z], device=device, dtype=dtype)
    off_zn = torch.tensor([     0,           0,    -cell_size_z], device=device, dtype=dtype)

    # Build corners
    obb = gather_and_inverse_offset(cell_x, cell_y - 1, cell_z - 1, off_yn + off_zn)
    obd = gather_and_inverse_offset(cell_x, cell_y - 1, cell_z + 1, off_yn + off_zp)
    odb = gather_and_inverse_offset(cell_x, cell_y + 1, cell_z - 1, off_yp + off_zn)
    odd = gather_and_inverse_offset(cell_x, cell_y + 1, cell_z + 1, off_yp + off_zp)
    obo = gather_and_inverse_offset(cell_x, cell_y - 1, cell_z, off_yn)
    oob = gather_and_inverse_offset(cell_x, cell_y, cell_z - 1, off_zn)
    odo = gather_and_inverse_offset(cell_x, cell_y + 1, cell_z, off_yp)
    ood = gather_and_inverse_offset(cell_x, cell_y, cell_z + 1, off_zp)

    bob = gather_and_inverse_offset(cell_x - 1, cell_y, cell_z - 1, off_xn + off_zn)
    bod = gather_and_inverse_offset(cell_x - 1, cell_y, cell_z + 1, off_xn + off_zp)
    dob = gather_and_inverse_offset(cell_x + 1, cell_y, cell_z - 1, off_xp + off_zn)
    dod = gather_and_inverse_offset(cell_x + 1, cell_y, cell_z + 1, off_xp + off_zp)
    boo = gather_and_inverse_offset(cell_x - 1, cell_y, cell_z, off_xn)
    doo = gather_and_inverse_offset(cell_x + 1, cell_y, cell_z, off_xp)

    bbo = gather_and_inverse_offset(cell_x - 1, cell_y - 1, cell_z, off_xn + off_yn)
    bdo = gather_and_inverse_offset(cell_x - 1, cell_y + 1, cell_z, off_xn + off_yp)
    dbo = gather_and_inverse_offset(cell_x + 1, cell_y - 1, cell_z, off_xp + off_yn)
    ddo = gather_and_inverse_offset(cell_x + 1, cell_y + 1, cell_z, off_xp + off_yp)

    # Prepare for barycentric
    o = ray_origins.unsqueeze(-2)
    d = ray_directions.unsqueeze(-2)

    # Tri-intersect helper
    def tri_intersect(p1, p2, p3):
        a_, b_, c_, tri_mask = get_barycentric(p1, p2, p3, o, d)
        P_ = p1 * a_.unsqueeze(-1) + p2 * b_.unsqueeze(-1) + p3 * c_.unsqueeze(-1)
        inside_mask = get_inside_cell_mask(
            P_, ooxyz,
            cell_size_x * 0.5, cell_size_y * 0.5, cell_size_z * 0.5,
            -cell_size_x * 0.5, -cell_size_y * 0.5, -cell_size_z * 0.5
        )
        return P_, inside_mask & tri_mask & masks

    # X direction
    P_x_1, P_x_1m = tri_intersect(obb, obo, ooo)
    P_x_2, P_x_2m = tri_intersect(obb, oob, ooo)
    P_x_3, P_x_3m = tri_intersect(odd, odo, ooo)
    P_x_4, P_x_4m = tri_intersect(odd, ood, ooo)
    P_x_5, P_x_5m = tri_intersect(oob, odo, ooo)
    P_x_6, P_x_6m = tri_intersect(oob, odo, odb)
    P_x_7, P_x_7m = tri_intersect(obo, ood, ooo)
    P_x_8, P_x_8m = tri_intersect(obo, ood, obd)

    # Y direction
    P_y_1, P_y_1m = tri_intersect(bob, boo, ooo)
    P_y_2, P_y_2m = tri_intersect(bob, oob, ooo)
    P_y_3, P_y_3m = tri_intersect(dod, doo, ooo)
    P_y_4, P_y_4m = tri_intersect(dod, ood, ooo)
    P_y_5, P_y_5m = tri_intersect(oob, doo, ooo)
    P_y_6, P_y_6m = tri_intersect(oob, doo, dob)
    P_y_7, P_y_7m = tri_intersect(boo, ood, ooo)
    P_y_8, P_y_8m = tri_intersect(boo, ood, bod)

    # Z direction
    P_z_1, P_z_1m = tri_intersect(bbo, boo, ooo)
    P_z_2, P_z_2m = tri_intersect(bbo, obo, ooo)
    P_z_3, P_z_3m = tri_intersect(ddo, doo, ooo)
    P_z_4, P_z_4m = tri_intersect(ddo, odo, ooo)
    P_z_5, P_z_5m = tri_intersect(obo, doo, ooo)
    P_z_6, P_z_6m = tri_intersect(obo, doo, dbo)
    P_z_7, P_z_7m = tri_intersect(boo, odo, ooo)
    P_z_8, P_z_8m = tri_intersect(boo, odo, bdo)

    # Concatenate masks => (..., 24)
    world_masks = torch.cat([
        P_x_1m, P_x_2m, P_x_3m, P_x_4m,
        P_x_5m, P_x_6m, P_x_7m, P_x_8m,
        P_y_1m, P_y_2m, P_y_3m, P_y_4m,
        P_y_5m, P_y_6m, P_y_7m, P_y_8m,
        P_z_1m, P_z_2m, P_z_3m, P_z_4m,
        P_z_5m, P_z_6m, P_z_7m, P_z_8m,
    ], dim=-1)

    # Concatenate positions => (..., 24, 3)
    world_positions = torch.cat([
        P_x_1, P_x_2, P_x_3, P_x_4,
        P_x_5, P_x_6, P_x_7, P_x_8,
        P_y_1, P_y_2, P_y_3, P_y_4,
        P_y_5, P_y_6, P_y_7, P_y_8,
        P_z_1, P_z_2, P_z_3, P_z_4,
        P_z_5, P_z_6, P_z_7, P_z_8,
    ], dim=-2)

    # Dot with ray_directions to get t
    inf = torch.tensor(1000.0, device=device, dtype=dtype)
    world_tx = (world_positions * ray_directions.unsqueeze(-2)).sum(dim=-1)
    # Large t for invalid
    world_tx = world_tx * world_masks + inf * (~world_masks).to(dtype)

    # Sort
    ind = torch.argsort(world_tx, dim=-1)
    ind = ind[..., :keep_num]
    world_tx = torch.gather(world_tx, -1, ind)
    world_masks = torch.gather(world_masks, -1, ind)

    # Gather positions
    expanded_ind = ind.unsqueeze(-1).expand(*ind.shape, 3)
    world_positions = torch.gather(world_positions, -2, expanded_ind)

    # Taper coords, zero out invalid
    taper_positions = get_taper_coord(world_positions)
    taper_positions = taper_positions * world_masks.unsqueeze(-1)

    return (
        taper_positions,
        world_masks,
        ooo_ * masks.unsqueeze(-1),  # or just "ooo_"
        world_tx.detach()            # mimic stop_gradient
    )


def compute_undc_intersection_and_return_uv(
    point_grid,          # (point_grid_size, point_grid_size, point_grid_size, 3) torch.Tensor
    point_UV_grid,       # (point_grid_size, point_grid_size, point_grid_size, 3, 4, 2) torch.Tensor
    texture_alpha,       # (tex_size, tex_size, 1) torch.Tensor
    cell_xyz,            # (..., 3) torch.LongTensor
    masks,               # (...) torch.BoolTensor
    rays_o,                # tuple (ray_origins, ray_directions), each (..., 3)
    rays_d,
    grid_min, grid_max,  # float or torch scalar
    point_grid_size,     # int
    point_grid_diff_lr_scale,  # float or torch scalar
    cell_size_x, cell_size_y, cell_size_z,  # float or torch scalar
    out_img_size,        # int
):
    """
    PyTorch version of the JAX function compute_undc_intersection_and_return_uv.

    Returns
    -------
    world_alpha : (..., keep, 1) torch.Tensor
        The per-triangle 'alpha' lookups from the texture, sorted along the ray.
        (Multiplied by the valid intersection mask.)
    world_uv    : (..., keep, 2) torch.LongTensor
        The integer UV coordinates (after scaling/clipping), sorted along the ray.
    """

    ray_origins = rays_o
    ray_directions = rays_d
    dtype  = ray_origins.dtype
    device = ray_origins.device

    # Derive integer indices for each cell
    cell_x = cell_xyz[..., 0]
    cell_y = cell_xyz[..., 1]
    cell_z = cell_xyz[..., 2]

    cell_x1 = cell_x + 1
    cell_y1 = cell_y + 1
    cell_z1 = cell_z + 1
    cell_x0 = cell_x - 1
    cell_y0 = cell_y - 1
    cell_z0 = cell_z - 1

    # Convert cell_xyz to "center" in world coords
    ooxyz = (cell_xyz.to(dtype) + 0.5) * ((grid_max - grid_min) / point_grid_size) + grid_min

    # Gather the "origin" corner offset
    ooo_ = point_grid[cell_x, cell_y, cell_z] * point_grid_diff_lr_scale
    ooo  = inverse_taper_coord(ooo_ + ooxyz)

    # Convenience: define a helper that gathers from point_grid, adds an offset, then applies inverse_taper_coord
    def gather_inverse_taper(px, py, pz, offset):
        # shape of offset is (3,) typically
        coords = point_grid[px, py, pz] * point_grid_diff_lr_scale + offset + ooxyz
        return inverse_taper_coord(coords)

    # Prebuild small offsets as tensors
    off_zero = torch.zeros(3, dtype=dtype, device=device)
    off_xp   = torch.tensor([ cell_size_x,     0,           0], dtype=dtype, device=device)
    off_xn   = torch.tensor([-cell_size_x,     0,           0], dtype=dtype, device=device)
    off_yp   = torch.tensor([     0,      cell_size_y,      0], dtype=dtype, device=device)
    off_yn   = torch.tensor([     0,     -cell_size_y,      0], dtype=dtype, device=device)
    off_zp   = torch.tensor([     0,           0,     cell_size_z], dtype=dtype, device=device)
    off_zn   = torch.tensor([     0,           0,    -cell_size_z], dtype=dtype, device=device)

    # -- X direction corners
    obb = gather_inverse_taper(cell_x,  cell_y0, cell_z0, off_yn + off_zn)
    obd = gather_inverse_taper(cell_x,  cell_y0, cell_z1, off_yn + off_zp)
    odb = gather_inverse_taper(cell_x,  cell_y1, cell_z0, off_yp + off_zn)
    odd = gather_inverse_taper(cell_x,  cell_y1, cell_z1, off_yp + off_zp)
    obo = gather_inverse_taper(cell_x,  cell_y0, cell_z,  off_yn)
    oob = gather_inverse_taper(cell_x,  cell_y,  cell_z0, off_zn)
    odo = gather_inverse_taper(cell_x,  cell_y1, cell_z,  off_yp)
    ood = gather_inverse_taper(cell_x,  cell_y,  cell_z1, off_zp)

    # -- Y direction corners
    bob = gather_inverse_taper(cell_x0, cell_y,  cell_z0, off_xn + off_zn)
    bod = gather_inverse_taper(cell_x0, cell_y,  cell_z1, off_xn + off_zp)
    dob = gather_inverse_taper(cell_x1, cell_y,  cell_z0, off_xp + off_zn)
    dod = gather_inverse_taper(cell_x1, cell_y,  cell_z1, off_xp + off_zp)
    boo = gather_inverse_taper(cell_x0, cell_y,  cell_z,  off_xn)
    doo = gather_inverse_taper(cell_x1, cell_y,  cell_z,  off_xp)

    # -- Z direction corners
    bbo = gather_inverse_taper(cell_x0, cell_y0, cell_z,  off_xn + off_yn)
    bdo = gather_inverse_taper(cell_x0, cell_y1, cell_z,  off_xn + off_yp)
    dbo = gather_inverse_taper(cell_x1, cell_y0, cell_z,  off_xp + off_yn)
    ddo = gather_inverse_taper(cell_x1, cell_y1, cell_z,  off_xp + off_yp)

    # Expand ray origins/directions so we can broadcast
    # shape => (..., 1, 3)
    o = ray_origins.unsqueeze(-2)
    d = ray_directions.unsqueeze(-2)

    # Helper to compute intersection + UV + alpha
    # This mirrors each block in the JAX code
    def intersect_and_uv(p1, p2, p3, 
                         uv1, uv2, uv3,  # shape (..., 2)
                         uv_select,      # e.g. [cell_x, cell_y0, cell_z0, 0, 0] style
                         global_mask):
        """
        p1, p2, p3: corner coords of the triangle, shape (..., 3).
        uv1, uv2, uv3: the corner UV coords from point_UV_grid, shape (..., 2).
        global_mask: the existing boolean mask (masks) plus bary mask, etc.

        Returns:
          P_   : intersection coords
          mask : final mask
          alpha: texture lookup
          P_uv : integer UV coords
        """
        a, b, c, tri_mask = get_barycentric(p1, p2, p3, o, d)
        # Weighted combination
        P_ = p1 * a.unsqueeze(-1) + p2 * b.unsqueeze(-1) + p3 * c.unsqueeze(-1)
        # Inside cell? (You likely have your own logic/thresholds in get_inside_cell_mask)
        inside_mask = get_inside_cell_mask(
            P_, ooxyz,
            cell_size_x * 0.5, cell_size_y * 0.5, cell_size_z * 0.5,
            -cell_size_x * 0.5, -cell_size_y * 0.5, -cell_size_z * 0.5
        ) & tri_mask & global_mask

        # UV:
        # shape of uv1, uv2, uv3 => same as p1,p2,p3 except for the last dimension=2
        # Weighted sum for UV
        P_uv = uv1 * a.unsqueeze(-1) + uv2 * b.unsqueeze(-1) + uv3 * c.unsqueeze(-1)
        # Scale, clamp, convert to long
        P_uv = (P_uv * out_img_size).long().clamp(0, out_img_size - 1)

        # Grab alpha from texture_alpha. shape => (...), if texture_alpha has shape (H,W,1)
        # we can index with [x,y,0], then unsqueeze(-1).
        # (If texture_alpha[x,y] automatically yields shape (...,1), you may not need the extra [...,0].)
        P_alpha = texture_alpha[P_uv[..., 0], P_uv[..., 1], 0].unsqueeze(-1)

        return P_, inside_mask, P_alpha, P_uv

    # Now replicate the 8 triangles for each axis, EXACTLY as in the JAX code.
    # The difference is we also gather the correct UV corners from point_UV_grid.

    # ------ x direction triangles ------ #
    # 1) P[x,y-1,z-1], P[x,y-1,z], P[x,y,z]
    p1, p2, p3 = obb, obo, ooo
    p1_uv_1 = point_UV_grid[cell_x, cell_y0, cell_z0, 0, 0]  # shape (..., 2)
    p2_uv_1 = point_UV_grid[cell_x, cell_y0, cell_z0, 0, 2]
    p3_uv_1 = point_UV_grid[cell_x, cell_y0, cell_z0, 0, 3]
    P_x_1, P_x_1m, P_x_1c, P_x_1uv = intersect_and_uv(p1, p2, p3,
                                                     p1_uv_1, p2_uv_1, p3_uv_1,
                                                     None,
                                                     masks)

    # 2) P[x,y-1,z-1], P[x,y,z-1], P[x,y,z]
    p1, p2, p3 = obb, oob, ooo
    p1_uv_2 = point_UV_grid[cell_x, cell_y0, cell_z0, 0, 0]
    p2_uv_2 = point_UV_grid[cell_x, cell_y0, cell_z0, 0, 1]
    p3_uv_2 = point_UV_grid[cell_x, cell_y0, cell_z0, 0, 3]
    P_x_2, P_x_2m, P_x_2c, P_x_2uv = intersect_and_uv(p1, p2, p3,
                                                     p1_uv_2, p2_uv_2, p3_uv_2,
                                                     None,
                                                     masks)

    # 3) P[x,y+1,z+1], P[x,y+1,z], P[x,y,z]
    p1, p2, p3 = odd, odo, ooo
    p1_uv_3 = point_UV_grid[cell_x, cell_y, cell_z, 0, 3]
    p2_uv_3 = point_UV_grid[cell_x, cell_y, cell_z, 0, 1]
    p3_uv_3 = point_UV_grid[cell_x, cell_y, cell_z, 0, 0]
    P_x_3, P_x_3m, P_x_3c, P_x_3uv = intersect_and_uv(p1, p2, p3,
                                                     p1_uv_3, p2_uv_3, p3_uv_3,
                                                     None,
                                                     masks)

    # 4) P[x,y+1,z+1], P[x,y,z+1], P[x,y,z]
    p1, p2, p3 = odd, ood, ooo
    p1_uv_4 = point_UV_grid[cell_x, cell_y, cell_z, 0, 3]
    p2_uv_4 = point_UV_grid[cell_x, cell_y, cell_z, 0, 2]
    p3_uv_4 = point_UV_grid[cell_x, cell_y, cell_z, 0, 0]
    P_x_4, P_x_4m, P_x_4c, P_x_4uv = intersect_and_uv(p1, p2, p3,
                                                     p1_uv_4, p2_uv_4, p3_uv_4,
                                                     None,
                                                     masks)

    # 5) P[x,y,z-1], P[x,y+1,z], P[x,y,z]
    p1, p2, p3 = oob, odo, ooo
    p1_uv_5 = point_UV_grid[cell_x, cell_y, cell_z0, 0, 0]
    p2_uv_5 = point_UV_grid[cell_x, cell_y, cell_z0, 0, 3]
    p3_uv_5 = point_UV_grid[cell_x, cell_y, cell_z0, 0, 2]
    P_x_5, P_x_5m, P_x_5c, P_x_5uv = intersect_and_uv(p1, p2, p3,
                                                     p1_uv_5, p2_uv_5, p3_uv_5,
                                                     None,
                                                     masks)

    # 6) P[x,y,z-1], P[x,y+1,z], P[x,y+1,z-1]
    p1, p2, p3 = oob, odo, odb
    p1_uv_6 = point_UV_grid[cell_x, cell_y, cell_z0, 0, 0]
    p2_uv_6 = point_UV_grid[cell_x, cell_y, cell_z0, 0, 3]
    p3_uv_6 = point_UV_grid[cell_x, cell_y, cell_z0, 0, 1]
    P_x_6, P_x_6m, P_x_6c, P_x_6uv = intersect_and_uv(p1, p2, p3,
                                                     p1_uv_6, p2_uv_6, p3_uv_6,
                                                     None,
                                                     masks)

    # 7) P[x,y-1,z], P[x,y,z+1], P[x,y,z]
    p1, p2, p3 = obo, ood, ooo
    p1_uv_7 = point_UV_grid[cell_x, cell_y0, cell_z, 0, 0]
    p2_uv_7 = point_UV_grid[cell_x, cell_y0, cell_z, 0, 3]
    p3_uv_7 = point_UV_grid[cell_x, cell_y0, cell_z, 0, 1]
    P_x_7, P_x_7m, P_x_7c, P_x_7uv = intersect_and_uv(p1, p2, p3,
                                                     p1_uv_7, p2_uv_7, p3_uv_7,
                                                     None,
                                                     masks)

    # 8) P[x,y-1,z], P[x,y,z+1], P[x,y-1,z+1]
    p1, p2, p3 = obo, ood, obd
    p1_uv_8 = point_UV_grid[cell_x, cell_y0, cell_z, 0, 0]
    p2_uv_8 = point_UV_grid[cell_x, cell_y0, cell_z, 0, 3]
    p3_uv_8 = point_UV_grid[cell_x, cell_y0, cell_z, 0, 2]
    P_x_8, P_x_8m, P_x_8c, P_x_8uv = intersect_and_uv(p1, p2, p3,
                                                     p1_uv_8, p2_uv_8, p3_uv_8,
                                                     None,
                                                     masks)

    # ------ y direction triangles ------ #
    # Similarly, replicate for Y:
    p1, p2, p3 = bob, boo, ooo
    p1_uv_1 = point_UV_grid[cell_x0, cell_y, cell_z0, 1, 0]
    p2_uv_1 = point_UV_grid[cell_x0, cell_y, cell_z0, 1, 2]
    p3_uv_1 = point_UV_grid[cell_x0, cell_y, cell_z0, 1, 3]
    P_y_1, P_y_1m, P_y_1c, P_y_1uv = intersect_and_uv(p1, p2, p3,
                                                     p1_uv_1, p2_uv_1, p3_uv_1,
                                                     None,
                                                     masks)

    p1, p2, p3 = bob, oob, ooo
    p1_uv_2 = point_UV_grid[cell_x0, cell_y, cell_z0, 1, 0]
    p2_uv_2 = point_UV_grid[cell_x0, cell_y, cell_z0, 1, 1]
    p3_uv_2 = point_UV_grid[cell_x0, cell_y, cell_z0, 1, 3]
    P_y_2, P_y_2m, P_y_2c, P_y_2uv = intersect_and_uv(p1, p2, p3,
                                                     p1_uv_2, p2_uv_2, p3_uv_2,
                                                     None,
                                                     masks)

    p1, p2, p3 = dod, doo, ooo
    p1_uv_3 = point_UV_grid[cell_x, cell_y, cell_z, 1, 3]
    p2_uv_3 = point_UV_grid[cell_x, cell_y, cell_z, 1, 1]
    p3_uv_3 = point_UV_grid[cell_x, cell_y, cell_z, 1, 0]
    P_y_3, P_y_3m, P_y_3c, P_y_3uv = intersect_and_uv(p1, p2, p3,
                                                     p1_uv_3, p2_uv_3, p3_uv_3,
                                                     None,
                                                     masks)

    p1, p2, p3 = dod, ood, ooo
    p1_uv_4 = point_UV_grid[cell_x, cell_y, cell_z, 1, 3]
    p2_uv_4 = point_UV_grid[cell_x, cell_y, cell_z, 1, 2]
    p3_uv_4 = point_UV_grid[cell_x, cell_y, cell_z, 1, 0]
    P_y_4, P_y_4m, P_y_4c, P_y_4uv = intersect_and_uv(p1, p2, p3,
                                                     p1_uv_4, p2_uv_4, p3_uv_4,
                                                     None,
                                                     masks)

    p1, p2, p3 = oob, doo, ooo
    p1_uv_5 = point_UV_grid[cell_x, cell_y, cell_z0, 1, 0]
    p2_uv_5 = point_UV_grid[cell_x, cell_y, cell_z0, 1, 3]
    p3_uv_5 = point_UV_grid[cell_x, cell_y, cell_z0, 1, 2]
    P_y_5, P_y_5m, P_y_5c, P_y_5uv = intersect_and_uv(p1, p2, p3,
                                                     p1_uv_5, p2_uv_5, p3_uv_5,
                                                     None,
                                                     masks)

    p1, p2, p3 = oob, doo, dob
    p1_uv_6 = point_UV_grid[cell_x, cell_y, cell_z0, 1, 0]
    p2_uv_6 = point_UV_grid[cell_x, cell_y, cell_z0, 1, 3]
    p3_uv_6 = point_UV_grid[cell_x, cell_y, cell_z0, 1, 1]
    P_y_6, P_y_6m, P_y_6c, P_y_6uv = intersect_and_uv(p1, p2, p3,
                                                     p1_uv_6, p2_uv_6, p3_uv_6,
                                                     None,
                                                     masks)

    p1, p2, p3 = boo, ood, ooo
    p1_uv_7 = point_UV_grid[cell_x0, cell_y, cell_z, 1, 0]
    p2_uv_7 = point_UV_grid[cell_x0, cell_y, cell_z, 1, 3]
    p3_uv_7 = point_UV_grid[cell_x0, cell_y, cell_z, 1, 1]
    P_y_7, P_y_7m, P_y_7c, P_y_7uv = intersect_and_uv(p1, p2, p3,
                                                     p1_uv_7, p2_uv_7, p3_uv_7,
                                                     None,
                                                     masks)

    p1, p2, p3 = boo, ood, bod
    p1_uv_8 = point_UV_grid[cell_x0, cell_y, cell_z, 1, 0]
    p2_uv_8 = point_UV_grid[cell_x0, cell_y, cell_z, 1, 3]
    p3_uv_8 = point_UV_grid[cell_x0, cell_y, cell_z, 1, 2]
    P_y_8, P_y_8m, P_y_8c, P_y_8uv = intersect_and_uv(p1, p2, p3,
                                                     p1_uv_8, p2_uv_8, p3_uv_8,
                                                     None,
                                                     masks)

    # ------ z direction triangles ------ #
    p1, p2, p3 = bbo, boo, ooo
    p1_uv_1 = point_UV_grid[cell_x0, cell_y0, cell_z, 2, 0]
    p2_uv_1 = point_UV_grid[cell_x0, cell_y0, cell_z, 2, 2]
    p3_uv_1 = point_UV_grid[cell_x0, cell_y0, cell_z, 2, 3]
    P_z_1, P_z_1m, P_z_1c, P_z_1uv = intersect_and_uv(p1, p2, p3,
                                                     p1_uv_1, p2_uv_1, p3_uv_1,
                                                     None,
                                                     masks)

    p1, p2, p3 = bbo, obo, ooo
    p1_uv_2 = point_UV_grid[cell_x0, cell_y0, cell_z, 2, 0]
    p2_uv_2 = point_UV_grid[cell_x0, cell_y0, cell_z, 2, 1]
    p3_uv_2 = point_UV_grid[cell_x0, cell_y0, cell_z, 2, 3]
    P_z_2, P_z_2m, P_z_2c, P_z_2uv = intersect_and_uv(p1, p2, p3,
                                                     p1_uv_2, p2_uv_2, p3_uv_2,
                                                     None,
                                                     masks)

    p1, p2, p3 = ddo, doo, ooo
    p1_uv_3 = point_UV_grid[cell_x, cell_y, cell_z, 2, 3]
    p2_uv_3 = point_UV_grid[cell_x, cell_y, cell_z, 2, 1]
    p3_uv_3 = point_UV_grid[cell_x, cell_y, cell_z, 2, 0]
    P_z_3, P_z_3m, P_z_3c, P_z_3uv = intersect_and_uv(p1, p2, p3,
                                                     p1_uv_3, p2_uv_3, p3_uv_3,
                                                     None,
                                                     masks)

    p1, p2, p3 = ddo, odo, ooo
    p1_uv_4 = point_UV_grid[cell_x, cell_y, cell_z, 2, 3]
    p2_uv_4 = point_UV_grid[cell_x, cell_y, cell_z, 2, 2]
    p3_uv_4 = point_UV_grid[cell_x, cell_y, cell_z, 2, 0]
    P_z_4, P_z_4m, P_z_4c, P_z_4uv = intersect_and_uv(p1, p2, p3,
                                                     p1_uv_4, p2_uv_4, p3_uv_4,
                                                     None,
                                                     masks)

    p1, p2, p3 = obo, doo, ooo
    p1_uv_5 = point_UV_grid[cell_x, cell_y0, cell_z, 2, 0]
    p2_uv_5 = point_UV_grid[cell_x, cell_y0, cell_z, 2, 3]
    p3_uv_5 = point_UV_grid[cell_x, cell_y0, cell_z, 2, 2]
    P_z_5, P_z_5m, P_z_5c, P_z_5uv = intersect_and_uv(p1, p2, p3,
                                                     p1_uv_5, p2_uv_5, p3_uv_5,
                                                     None,
                                                     masks)

    p1, p2, p3 = obo, doo, dbo
    p1_uv_6 = point_UV_grid[cell_x, cell_y0, cell_z, 2, 0]
    p2_uv_6 = point_UV_grid[cell_x, cell_y0, cell_z, 2, 3]
    p3_uv_6 = point_UV_grid[cell_x, cell_y0, cell_z, 2, 1]
    P_z_6, P_z_6m, P_z_6c, P_z_6uv = intersect_and_uv(p1, p2, p3,
                                                     p1_uv_6, p2_uv_6, p3_uv_6,
                                                     None,
                                                     masks)

    p1, p2, p3 = boo, odo, ooo
    p1_uv_7 = point_UV_grid[cell_x0, cell_y, cell_z, 2, 0]
    p2_uv_7 = point_UV_grid[cell_x0, cell_y, cell_z, 2, 3]
    p3_uv_7 = point_UV_grid[cell_x0, cell_y, cell_z, 2, 1]
    P_z_7, P_z_7m, P_z_7c, P_z_7uv = intersect_and_uv(p1, p2, p3,
                                                     p1_uv_7, p2_uv_7, p3_uv_7,
                                                     None,
                                                     masks)

    p1, p2, p3 = boo, odo, bdo
    p1_uv_8 = point_UV_grid[cell_x0, cell_y, cell_z, 2, 0]
    p2_uv_8 = point_UV_grid[cell_x0, cell_y, cell_z, 2, 3]
    p3_uv_8 = point_UV_grid[cell_x0, cell_y, cell_z, 2, 2]
    P_z_8, P_z_8m, P_z_8c, P_z_8uv = intersect_and_uv(p1, p2, p3,
                                                     p1_uv_8, p2_uv_8, p3_uv_8,
                                                     None,
                                                     masks)

    # ------------------------------
    # Concatenate all intersections
    # ------------------------------
    # Each set has 8 triangles => 24 total.
    # world_masks: shape (..., 24)
    world_masks = torch.cat([
        P_x_1m, P_x_2m, P_x_3m, P_x_4m,
        P_x_5m, P_x_6m, P_x_7m, P_x_8m,
        P_y_1m, P_y_2m, P_y_3m, P_y_4m,
        P_y_5m, P_y_6m, P_y_7m, P_y_8m,
        P_z_1m, P_z_2m, P_z_3m, P_z_4m,
        P_z_5m, P_z_6m, P_z_7m, P_z_8m,
    ], dim=-1)

    # world_positions: shape (..., 24, 3)
    world_positions = torch.cat([
        P_x_1, P_x_2, P_x_3, P_x_4,
        P_x_5, P_x_6, P_x_7, P_x_8,
        P_y_1, P_y_2, P_y_3, P_y_4,
        P_y_5, P_y_6, P_y_7, P_y_8,
        P_z_1, P_z_2, P_z_3, P_z_4,
        P_z_5, P_z_6, P_z_7, P_z_8,
    ], dim=-2)

    # world_alpha: shape (..., 24, 1)
    world_alpha = torch.cat([
        P_x_1c, P_x_2c, P_x_3c, P_x_4c,
        P_x_5c, P_x_6c, P_x_7c, P_x_8c,
        P_y_1c, P_y_2c, P_y_3c, P_y_4c,
        P_y_5c, P_y_6c, P_y_7c, P_y_8c,
        P_z_1c, P_z_2c, P_z_3c, P_z_4c,
        P_z_5c, P_z_6c, P_z_7c, P_z_8c,
    ], dim=-2)

    # world_uv: shape (..., 24, 2)
    world_uv = torch.cat([
        P_x_1uv, P_x_2uv, P_x_3uv, P_x_4uv,
        P_x_5uv, P_x_6uv, P_x_7uv, P_x_8uv,
        P_y_1uv, P_y_2uv, P_y_3uv, P_y_4uv,
        P_y_5uv, P_y_6uv, P_y_7uv, P_y_8uv,
        P_z_1uv, P_z_2uv, P_z_3uv, P_z_4uv,
        P_z_5uv, P_z_6uv, P_z_7uv, P_z_8uv,
    ], dim=-2)

    # Compute t-values by dot(world_positions, ray_directions)
    # shape => (..., 24)
    world_tx = (world_positions * ray_directions.unsqueeze(-2)).sum(dim=-1)

    # Invalid intersections => large 't'
    # so they go to the end in the sort.
    world_tx = world_tx * world_masks + 1000.0 * (~world_masks).to(dtype)

    # Sort
    ind = torch.argsort(world_tx, dim=-1)
    # In the JAX code, they keep the first (point_grid_size*3).
    # Adjust if you want fewer or more.
    keep = point_grid_size * 3
    ind = ind[..., :keep]

    # Gather the sorted results
    # world_masks => (..., keep)
    world_masks = torch.gather(world_masks, dim=-1, index=ind)

    # world_positions => (..., keep, 3)
    gather_idx = ind.unsqueeze(-1).expand(*ind.shape, 3)
    world_positions = torch.gather(world_positions, dim=-2, index=gather_idx)

    # world_alpha => (..., keep, 1)
    gather_idx_alpha = ind.unsqueeze(-1).expand(*ind.shape, 1)
    world_alpha = torch.gather(world_alpha, dim=-2, index=gather_idx_alpha)

    # world_uv => (..., keep, 2)
    gather_idx_uv = ind.unsqueeze(-1).expand(*ind.shape, 2)
    world_uv = torch.gather(world_uv, dim=-2, index=gather_idx_uv)

    # Finally, mask out invalid alpha
    # shape => (..., keep, 1)
    world_alpha = world_alpha * world_masks.unsqueeze(-1)

    return world_alpha, world_uv

# -------------------------------------------------------------------
# Another function
# -------------------------------------------------------------------
def gridcell_from_rays(
    rays_o, rays_d,
    acc_grid, keep_num, threshold,
    grid_min, grid_max, point_grid_size
):
    """
    PyTorch version of gridcell_from_rays.
    
    Returns:
        grid_indices: (..., keep_num, 3) int tensor of cell indices.
        grid_masks  : (..., keep_num) bool tensor indicating valid in-bounds cells.
    """
    device = rays_o.device
    dtype  = rays_o.dtype

    small_step_t = torch.tensor(1e-5, device=device, dtype=dtype)
    epsilon_t    = torch.tensor(1e-5, device=device, dtype=dtype)
    inf_t        = torch.tensor(1000.0, device=device, dtype=dtype)

    # Expand dims
    ox = rays_o[..., 0:1]
    oy = rays_o[..., 1:2]
    oz = rays_o[..., 2:3]

    dx = rays_d[..., 0:1]
    dy = rays_d[..., 1:2]
    dz = rays_d[..., 2:3]

    # Mark near-zero direction components
    dxm = (dx.abs() < epsilon_t).float()
    dym = (dy.abs() < epsilon_t).float()
    dzm = (dz.abs() < epsilon_t).float()

    # Avoid division by zero by adding 1
    dx = dx + dxm
    dy = dy + dym
    dz = dz + dzm

    # layers in [0, 1], shape (..., point_grid_size+1)
    layers = torch.linspace(0, 1, steps=point_grid_size+1, device=device, dtype=dtype)
    # Expand to match rays_o shape except the last dim
    for _ in range(rays_o.ndim - 1):
        layers = layers.unsqueeze(0)
    layers = layers.expand(*rays_o.shape[:-1], point_grid_size+1)

    # Compute t-values for crossing
    tx = ((layers * (grid_max[0] - grid_min[0]) + grid_min[0]) - ox) / dx
    ty = ((layers * (grid_max[1] - grid_min[1]) + grid_min[1]) - oy) / dy
    tz = ((layers * (grid_max[2] - grid_min[2]) + grid_min[2]) - oz) / dz

    tx = tx * (1 - dxm) + inf_t * dxm
    ty = ty * (1 - dym) + inf_t * dym
    tz = tz * (1 - dzm) + inf_t * dzm

    # Concatenate
    txyz = torch.cat([tx, ty, tz], dim=-1)
    # Filter out negative t => set them to 1000
    neg_mask = (txyz <= 0).float()
    txyz = txyz * (1 - neg_mask) + inf_t * neg_mask

    # Slight offset
    txyz = txyz + small_step_t

    # Intersection positions
    wpos = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * txyz.unsqueeze(-1)
    # Acc grid mask
    acc_grid_vals = get_acc_grid_masks(wpos, acc_grid, grid_min, grid_max, point_grid_size)
    # Remove below threshold
    below_mask = (acc_grid_vals < threshold).float()
    txyz = txyz * (1 - below_mask) + inf_t * below_mask

    # Sort
    sorted_t, sorted_idx = torch.sort(txyz, dim=-1)
    sorted_t = sorted_t[..., :keep_num]
    sorted_idx = sorted_idx[..., :keep_num]

    # Recompute positions
    wpos = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * sorted_t.unsqueeze(-1)
    # Convert to grid coords
    gpos = (wpos - grid_min) * (point_grid_size / (grid_max - grid_min))

    # Check in-bounds
    grid_masks = (
        (gpos[..., 0] >= 1) & (gpos[..., 0] < point_grid_size - 1) &
        (gpos[..., 1] >= 1) & (gpos[..., 1] < point_grid_size - 1) &
        (gpos[..., 2] >= 1) & (gpos[..., 2] < point_grid_size - 1)
    )

    # Force OOB => index=1
    gpos = gpos * grid_masks.unsqueeze(-1).float() + (~grid_masks).unsqueeze(-1).float()
    # Convert to int
    grid_indices = gpos.long()

    return grid_indices, grid_masks

def gridcell_from_rays_bake(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    grid_min: torch.Tensor,
    grid_max: torch.Tensor,
    point_grid_size: int,
):
    """
    Matches the JAX function exactly:
        gridcell_from_rays(rays)

    Returns
    -------
    grid_indices : (..., 3*(point_grid_size+1), 3) int32
        The integer grid coordinates for all intersection "layers" 
        along X, Y, Z. Out-of-bounds points are set to 1.
    grid_masks   : (..., 3*(point_grid_size+1)) bool
        True if in-bounds, False if out-of-bounds.
    """
    device = rays_o.device
    dtype  = rays_o.dtype

    small_step = 1e-5
    epsilon    = 1e-5
    inf_val    = 1000.0

    # Separate components
    ox = rays_o[..., 0:1]
    oy = rays_o[..., 1:2]
    oz = rays_o[..., 2:3]

    dx = rays_d[..., 0:1]
    dy = rays_d[..., 1:2]
    dz = rays_d[..., 2:3]

    # Mark near-zero components => add them to avoid div-by-zero
    dxm = (dx.abs() < epsilon).to(dtype)  # 1 if near-zero, else 0
    dym = (dy.abs() < epsilon).to(dtype)
    dzm = (dz.abs() < epsilon).to(dtype)

    dx = dx + dxm
    dy = dy + dym
    dz = dz + dzm

    # layers in [0..1], shape => (..., point_grid_size+1)
    # matching "np.arange(point_grid_size+1)/point_grid_size"
    # then broadcast to match rays' batch shape.
    layers = torch.arange(point_grid_size + 1, device=device, dtype=dtype)
    layers = layers / point_grid_size  # => [0, 1/point_grid_size, ..., 1]

    # Expand to match batch dims (everything except the last dimension of rays)
    batch_shape = rays_o.shape[:-1]  # e.g. if rays_o is [B, 3], batch_shape=[B]
    for _ in range(len(batch_shape)):
        layers = layers.unsqueeze(0)
    # Now expand => shape [*batch_shape, (point_grid_size+1)]
    layers = layers.expand(*batch_shape, point_grid_size + 1)

    # Ranges
    rx = (grid_max[0] - grid_min[0]).to(dtype)
    ry = (grid_max[1] - grid_min[1]).to(dtype)
    rz = (grid_max[2] - grid_min[2]).to(dtype)

    # t-values for the x-layers, y-layers, z-layers
    tx = ((layers * rx + grid_min[0]) - ox) / dx  # shape => [..., point_grid_size+1]
    ty = ((layers * ry + grid_min[1]) - oy) / dy
    tz = ((layers * rz + grid_min[2]) - oz) / dz

    # Where direction was near-zero => set t=inf
    inf_t = torch.tensor(inf_val, device=device, dtype=dtype)
    tx = tx * (1 - dxm) + inf_t * dxm
    ty = ty * (1 - dym) + inf_t * dym
    tz = tz * (1 - dzm) + inf_t * dzm

    # Concatenate => shape => (..., 3*(point_grid_size+1))
    # EXACTLY like JAX: np.concatenate([tx, ty, tz], axis=-1)
    txyz = torch.cat([tx, ty, tz], dim=-1)

    # Negative t => clamp to large (1000)
    neg_mask = (txyz <= 0).to(dtype)
    txyz = txyz * (1 - neg_mask) + inf_val * neg_mask

    # Add small step
    txyz = txyz + small_step

    # Expand t to multiply with ray_directions
    # shape => (..., 3*(point_grid_size+1), 1)
    txyz_expanded = txyz.unsqueeze(-1)

    # World positions => shape => (..., 3*(point_grid_size+1), 3)
    ro = rays_o.unsqueeze(-2)  # => (..., 1, 3)
    rd = rays_d.unsqueeze(-2)  # => (..., 1, 3)
    world_positions = ro + rd * txyz_expanded

    # Convert to grid coords
    # grid_positions => shape => (..., 3*(point_grid_size+1), 3)
    grid_range = (grid_max - grid_min).to(dtype)
    factor = point_grid_size / grid_range  # shape [3]
    # Broadcast => factor must match last dimension 3
    # => factor.view(1,1,3) or use unsqueeze
    # But in PyTorch, a direct multiply will broadcast if shape is compatible.
    grid_positions = (world_positions - grid_min) * factor  # subtract min, scale

    # In-bounds check => shape => (..., 3*(point_grid_size+1))
    gx = grid_positions[..., 0]
    gy = grid_positions[..., 1]
    gz = grid_positions[..., 2]
    grid_masks = (
        (gx >= 1) & (gx < (point_grid_size - 1)) &
        (gy >= 1) & (gy < (point_grid_size - 1)) &
        (gz >= 1) & (gz < (point_grid_size - 1))
    )

    # If out-of-bounds => set coords = 1
    # (the JAX code does: grid_positions = grid_positions*mask + (1 * ~mask))
    mask_f    = grid_masks.to(dtype).unsqueeze(-1)    # => (..., 3*(point_grid_size+1), 1)
    not_maskf = (1 - mask_f)                          # => same shape
    grid_positions = grid_positions * mask_f + not_maskf

    # Finally cast to int => shape => (..., 3*(point_grid_size+1), 3)
    grid_indices = grid_positions.to(torch.int32)

    return grid_indices, grid_masks



def compute_TV(acc_grid: torch.Tensor) -> torch.Tensor:
    """
    Computes the total variation (TV) for a 3D tensor acc_grid using PyTorch.

    Arguments:
        acc_grid (torch.Tensor): A 3D tensor for which TV will be computed.
                                 Should be on the desired device before calling.

    Returns:
        torch.Tensor: The mean total variation across x, y, z differences.
    """
    dx = acc_grid[:-1, :, :] - acc_grid[1:, :, :]
    dy = acc_grid[:, :-1, :] - acc_grid[:, 1:, :]
    dz = acc_grid[:, :, :-1] - acc_grid[:, :, 1:]

    tv = (dx.pow(2).mean() 
          + dy.pow(2).mean() 
          + dz.pow(2).mean())
    
    return tv