    def get_sift_orientation_batch(self, img, keypoints, patch_size=20, bin_size=5):
        '''
        img:tensor
        '''
        patch_size=20
        #w_gauss = xxx;

        ori_max = 180
        bins = ori_max // bin_size
        batch, c, h, w = img.shape
        offset = patch_size // 2
        device = img.device
    
        Gx=torch.zeros((batch, c, h+patch_size, w+patch_size), dtype=img.dtype, device=img.device)
        Gy=torch.zeros((batch, c, h+patch_size, w+patch_size), dtype=img.dtype, device=img.device)

        Gx0=torch.zeros_like(img)
        Gx2=torch.zeros_like(img)
        Gy0=torch.zeros_like(img)
        Gy2=torch.zeros_like(img)

        Gx0[:,:,:,1:-1] = img[:,:,:,:-2]
        Gx2[:,:,:,1:-1] = img[:,:,:,2:]
        Gx[:,:,offset:-offset,offset:-offset] = (Gx0 - Gx2)

        Gy0[:,:,1:-1,:] = img[:,:,:-2,:]
        Gy2[:,:,1:-1,:] = img[:,:,2:,:]
        Gy[:,:,offset:-offset,offset:-offset] = (Gy2 - Gy0)

        coor_cells = torch.stack(torch.meshgrid(torch.linspace(-1, 1, patch_size-1), torch.linspace(-1, 1, patch_size-1)), dim=2)  # 产生两个网格
        coor_cells = coor_cells.transpose(0, 1)
        coor_cells = coor_cells.to(self.device)
        coor_cells = coor_cells.contiguous()
        
        keypoints_num = keypoints.size(1)
        keypoints_correct = torch.round(keypoints.clone())
        keypoints_correct += offset
     
        src_pixel_coords = coor_cells.unsqueeze(0).repeat(batch, keypoints_num,1,1,1)
        src_pixel_coords = src_pixel_coords.float() * (patch_size // 2 - 1) + keypoints_correct.unsqueeze(2).unsqueeze(2).repeat(1,1,patch_size-1,patch_size-1,1)
      
        src_pixel_coords = src_pixel_coords.view([batch, keypoints_num, -1, 2])
        batch_image_coords_correct = torch.linspace(0, (batch-1)*(h+patch_size)*(w+patch_size), batch).long().to(device)
        src_pixel_coords_index = (src_pixel_coords[:,:,:,0] + src_pixel_coords[:,:,:,1]*(w+patch_size)).long()
        src_pixel_coords_index  = src_pixel_coords_index + batch_image_coords_correct[:,None,None]

        eps = 1e-12

        a = Gx.take(src_pixel_coords_index)
        b = Gy.take(src_pixel_coords_index)
        degree_value = b / (a + eps)
        angle = torch.atan(degree_value)
    
        angle = angle*57.29578049 #180/(3.1415926)
        
        a_mask = (a >= 0)
        b_mask = (b >= 0)
        apbp_mask = a_mask * b_mask
        apbn_mask = a_mask * (~b_mask)
        anbp_mask = (~a_mask) * b_mask
        anbn_mask = (~a_mask) * (~b_mask)
        angle[apbp_mask] = angle[apbp_mask]
        angle[apbn_mask] = angle[apbn_mask] + 360
        angle[anbp_mask] = angle[anbp_mask] + 180
        angle[anbn_mask] = angle[anbn_mask] + 180
        angle = angle % ori_max
        #高斯加权
        # w_gauss /= torch.sum(w_gauss)
        Amp = torch.sqrt(a**2 + b**2)#*w_gauss[None,None,:]
        angle_d = ((angle // bin_size)).long() % bins
        angle_d_onehot = F.one_hot(angle_d,num_classes=bins)
        hist = torch.sum(Amp.unsqueeze(-1)*angle_d_onehot,dim=-2) #[0,pi)

        #平滑
        h_t=torch.zeros((batch, keypoints_num, hist.size(-1)+4), dtype=hist.dtype, device=hist.device)
        h_t[:,:,2:-2] = hist
        h_t[:,:,-2:] = hist[:,:,:2]
        h_t[:,:,:2] = hist[:,:,-2:]

        h_p2=h_t[:,:,4:]
        h_n2=h_t[:,:,:-4]
        h_p1=h_t[:,:,3:-1]
        h_n1=h_t[:,:,1:-3]

        Hist = (h_p2 + h_n2 + 4*(h_p1 + h_n1) + 6*hist) / 16

        #获取主方向i
        H_p_i = torch.max(Hist,dim=-1).indices
        H_t=torch.zeros((batch, keypoints_num, Hist.size(-1)+2), dtype=Hist.dtype, device=Hist.device)
        H_t[:,:,1:-1] = Hist
        H_t[:,:,-1:] = Hist[:,:,:1]
        H_t[:,:,:1] = Hist[:,:,-1:]

        H_p1=H_t[:,:,2:]
        H_n1=H_t[:,:,:-2]

        H_i_offset = (H_n1 - H_p1) / (2*(H_n1 + H_p1 - 2*Hist) + eps)
        H_p_i_onehot = F.one_hot(H_p_i,num_classes=bins)
        H_p_offset = torch.sum(H_i_offset*H_p_i_onehot,dim=-1)
        H_p = (H_p_i + H_p_offset + 0.5) * bin_size
        H_p = H_p % 180 - 90

        return H_p

