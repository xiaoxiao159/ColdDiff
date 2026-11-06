import math
import torch
import torch.nn as nn


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(
            x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.temb_proj = torch.nn.Linear(temb_channels,
                                         out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)   # b,hw,c
        k = k.reshape(b, c, h*w)  # b,c,hw
        w_ = torch.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x+h_


def _haar_weight(channels):
    """Haar analysis filters for one-level 2D DWT, shape (4C, 1, 2, 2).

    Channel order within each group: LL, LH, HL, HH. The transform is
    orthonormal, so conv_transpose2d with the same weights is its exact
    inverse (up to float error).
    """
    w = torch.zeros(channels * 4, 1, 2, 2)
    for c in range(channels):
        w[4*c+0, 0] = torch.tensor([[ 0.5,  0.5], [ 0.5,  0.5]])  # LL
        w[4*c+1, 0] = torch.tensor([[ 0.5,  0.5], [-0.5, -0.5]])  # LH
        w[4*c+2, 0] = torch.tensor([[ 0.5, -0.5], [ 0.5, -0.5]])  # HL
        w[4*c+3, 0] = torch.tensor([[ 0.5, -0.5], [-0.5,  0.5]])  # HH
    return w


class WaveletGatedSkip(nn.Module):
    """Timestep-conditioned wavelet-gated skip (TWG).

    Placed on high-resolution encoder skips. One-level Haar DWT splits the
    skip feature into LL (anatomy, low freq) and LH/HL/HH (high freq, where
    LDCT noise and true edges coexist). A gate g -- driven by local |HF|
    energy and the timestep embedding -- interpolates every high-freq
    position between keeping it and replacing it with a conv-corrected
    version. The low-freq band only receives a zero-init residual, so it
    passes through unchanged at init.

    Init state: gate bias = +2 -> g ~= 0.88 everywhere, correction branches
    zero-init, so the module starts close to the original skip.
    """

    def __init__(self, channels, temb_channels, gate_bias_init=2.0):
        super().__init__()
        self.channels = channels
        w = _haar_weight(channels)
        self.register_buffer('dwt_w', w)
        self.register_buffer('idwt_w', w.clone())

        ch3 = channels * 3
        # candidate HF correction delta(H)
        self.hf_correct = nn.Sequential(
            nn.Conv2d(ch3, ch3, 3, padding=1, groups=ch3),
            nn.SiLU(),
            nn.Conv2d(ch3, ch3, 1),
        )
        nn.init.zeros_(self.hf_correct[2].weight)
        nn.init.zeros_(self.hf_correct[2].bias)

        # gate from local |HF| energy
        self.gate_conv = nn.Conv2d(ch3, ch3, 3, padding=1, groups=ch3)
        nn.init.zeros_(self.gate_conv.weight)
        nn.init.constant_(self.gate_conv.bias, gate_bias_init)

        # gate from timestep embedding (zero: no t-dependence at init)
        self.gate_temb = nn.Linear(temb_channels, ch3)
        nn.init.zeros_(self.gate_temb.weight)
        nn.init.zeros_(self.gate_temb.bias)

        # light LF correction, zero-init
        self.lf_correct = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 1),
        )
        nn.init.zeros_(self.lf_correct[2].weight)
        nn.init.zeros_(self.lf_correct[2].bias)

    def dwt(self, x):
        c = self.channels
        sub = torch.nn.functional.conv2d(x, self.dwt_w, stride=2, groups=c)
        return sub[:, :c], sub[:, c:2*c], sub[:, 2*c:3*c], sub[:, 3*c:]

    def idwt(self, ll, lh, hl, hh):
        sub = torch.cat([ll, lh, hl, hh], dim=1)
        return torch.nn.functional.conv_transpose2d(
            sub, self.idwt_w, stride=2, groups=self.channels)

    def forward(self, s, temb):
        c = self.channels
        ll, lh, hl, hh = self.dwt(s)
        hf = torch.cat([lh, hl, hh], dim=1)          # (B, 3C, H/2, W/2)
        hf_new = self.hf_correct(hf)
        g = torch.sigmoid(
            self.gate_conv(hf.abs())
            + self.gate_temb(nonlinearity(temb))[:, :, None, None])
        hf = g * hf + (1.0 - g) * hf_new
        ll = ll + self.lf_correct(ll)
        return self.idwt(ll, hf[:, :c], hf[:, c:2*c], hf[:, 2*c:3*c])


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        ch, out_ch, ch_mult = config.model.ch, config.model.out_ch, tuple(config.model.ch_mult)
        num_res_blocks = config.model.num_res_blocks
        attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        in_channels = config.model.in_channels
        resolution = config.data.image_size
        resamp_with_conv = config.model.resamp_with_conv
        num_timesteps = config.diffusion.num_diffusion_timesteps
        
        if config.model.type == 'bayesian':
            self.logvar = nn.Parameter(torch.zeros(num_timesteps))
        
        self.ch = ch
        self.temb_ch = self.ch*4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.twg_levels = list(getattr(config.model, 'twg_levels', []))

        # Consistency guard: the cold-diffusion residual-guidance switch
        # decides whether the network takes 2 or 3 input channels.
        cd_cfg = getattr(config, 'cold_diffusion', None)
        if cd_cfg is not None and getattr(config.model, 'type', '') == 'cold_diffusion':
            use_rm = bool(getattr(cd_cfg, 'use_residual_map', False))
            expected_in = 2 + (1 if use_rm else 0)
            if in_channels != expected_in:
                raise ValueError(
                    f"model.in_channels={in_channels} is inconsistent with "
                    f"cold_diffusion.use_residual_map={use_rm}: expected "
                    f"{expected_in} (2 = [LDCT, x_t], 3 = [LDCT, x_t, r_t])."
                )

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch,
                            self.temb_ch),
            torch.nn.Linear(self.temb_ch,
                            self.temb_ch),
        ])

        # timestep-conditioned wavelet-gated skips, one per gated level
        self.twg = nn.ModuleList()
        self.twg_index = {}
        for lvl in sorted(self.twg_levels):
            if not (0 <= lvl < self.num_resolutions - 1):
                raise ValueError(
                    f"twg level {lvl} out of range for "
                    f"{self.num_resolutions}-level U-Net")
            self.twg_index[lvl] = len(self.twg)
            self.twg.append(WaveletGatedSkip(ch*ch_mult[lvl], self.temb_ch))

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+ch_mult
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = ch*in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in+skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x, t, return_features=False):
        assert x.shape[2] == x.shape[3] == self.resolution

        # timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        # downsampling
        hs = [self.conv_in(x)]
        down_features = [] if return_features else None
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if return_features:
                down_features.append(hs[-1])  # feature at this level before downsample
            ds_in = hs[-1]
            if i_level in self.twg_index:
                # TWG cleans only the skip copy handed to the decoder; the
                # encoder main path (downsample input) keeps the raw feature
                hs[-1] = self.twg[self.twg_index[i_level]](ds_in, temb)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(ds_in))

        # middle
        h = hs[-1]

        mid_features = [] if return_features else None
        h = self.mid.block_1(h, temb)
        if return_features:
            mid_features.append(h)
        h = self.mid.attn_1(h)
        if return_features:
            mid_features.append(h)
        h = self.mid.block_2(h, temb)
        if return_features:
            mid_features.append(h)

        # upsampling
        up_features = [] if return_features else None
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if return_features:
                up_features.append(h)  # feature at this level before upsample
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        if return_features:
            return h, {'down': down_features, 'mid': mid_features, 'up': up_features}
        return h
