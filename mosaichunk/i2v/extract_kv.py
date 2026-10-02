"""Image, text, and camera conditioning; hooks capture unrotated keys."""

import os

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

_LWV2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import wan
from einops import rearrange
from wan.configs import WAN_CONFIGS
from wan.utils.cam_utils import (
    compute_relative_poses,
    get_Ks_transformed,
    get_plucker_embeddings,
    interpolate_camera_poses,
)

LAT_H, LAT_W = (60, 104)
PATCH = (1, 2, 2)
GRID_H, GRID_W = (LAT_H // PATCH[1], LAT_W // PATCH[2])
FRAME_SEQLEN = GRID_H * GRID_W
DEFAULT_PROMPT = "a static indoor scene, smooth camera movement, high quality"


def build_pipeline(
    ckpt_dir, local_attn_size=18, sink_size=6, device_id=0, convert_model_dtype=True
):
    """Construct WanI2VCausal (loads T5 + VAE + DiT) and FREEZE the DiT."""
    cfg = WAN_CONFIGS["i2v-A14B"]
    pipe = wan.WanI2VCausal(
        config=cfg,
        checkpoint_dir=ckpt_dir,
        device_id=device_id,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        convert_model_dtype=convert_model_dtype,
        local_attn_size=local_attn_size,
        sink_size=sink_size,
        infer_mode="causal_fast",
    )
    pipe.model.eval().requires_grad_(False)
    return pipe


class KVExtractor:
    """Registers forward hooks on chosen mid layers' self_attn.norm_k (pre-RoPE content K) and
    self_attn.v (content V), and runs the frozen DiT chunk-by-chunk on GT latents at t~=0."""

    def __init__(self, pipe, layers):
        self.pipe = pipe
        self.layers = list(layers)
        self.device = pipe.device
        self._cap = {l: {} for l in self.layers}
        self.inject = False
        self.inject_kv = {}
        self.capture = True
        self._handles = []
        for l in self.layers:
            attn = pipe.model.blocks[l].self_attn
            self._handles.append(attn.norm_k.register_forward_hook(self._mk_hook(l, "k")))
            self._handles.append(attn.v.register_forward_hook(self._mk_hook(l, "v")))

    def _mk_hook(self, layer, which):

        def hook(_module, _inp, out):
            if self.inject and layer in self.inject_kv:
                return self.inject_kv[layer][which]
            if self.capture:
                self._cap[layer][which] = out.detach()

        return hook

    def close(self):
        for h in self._handles:
            h.remove()

    def _build_y(self, first_frame_path, lat_f):
        """i2v image condition: [20, lat_f, 60, 104] = 4 mask + 16 vae(first-frame)."""
        F = (lat_f - 1) * 4 + 1
        h, w = (LAT_H * 8, LAT_W * 8)
        img = TF.to_tensor(Image.open(first_frame_path).convert("RGB")).sub_(0.5).div_(0.5)
        msk = torch.ones(1, F, LAT_H, LAT_W, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat(
            [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1
        )
        msk = msk.view(1, msk.shape[1] // 4, 4, LAT_H, LAT_W).transpose(1, 2)[0]
        vid = torch.concat(
            [
                torch.nn.functional.interpolate(
                    img[None].cpu(), size=(h, w), mode="bicubic"
                ).transpose(0, 1),
                torch.zeros(3, F - 1, h, w),
            ],
            dim=1,
        ).to(self.device)
        y = self.pipe.vae.encode([vid])[0]
        return torch.concat([msk, y]).to(self.pipe.param_dtype)

    def _build_context(self, prompt):
        self.pipe.text_encoder.model.to(self.device)
        ctx = self.pipe.text_encoder([prompt], self.device)
        self.pipe.text_encoder.model.cpu()
        return ctx

    def _build_plucker(self, action_path, lat_f):
        """Rebuild c2ws_plucker_emb exactly as generate()."""
        h, w = (LAT_H * 8, LAT_W * 8)
        c2ws = np.load(os.path.join(action_path, "poses.npy"))
        Ks = torch.from_numpy(np.load(os.path.join(action_path, "intrinsics.npy"))).float()
        Ks = get_Ks_transformed(
            Ks,
            height_org=480,
            width_org=832,
            height_resize=h,
            width_resize=w,
            height_final=h,
            width_final=w,
        )[0]
        len_c2ws = len(c2ws)
        c2ws_infer = interpolate_camera_poses(
            src_indices=np.linspace(0, len_c2ws - 1, len_c2ws),
            src_rot_mat=c2ws[:, :3, :3],
            src_trans_vec=c2ws[:, :3, 3],
            tgt_indices=np.linspace(0, len_c2ws - 1, lat_f),
        )
        c2ws_infer = compute_relative_poses(c2ws_infer, framewise=True)
        Ks = Ks.repeat(len(c2ws_infer), 1).to(self.device)
        c2ws_infer = c2ws_infer.to(self.device)
        emb = get_plucker_embeddings(c2ws_infer, Ks, h, w)
        emb = rearrange(
            emb, "f (h c1) (w c2) c -> (f h w) (c c1 c2)", c1=int(h // LAT_H), c2=int(w // LAT_W)
        )[None, ...]
        emb = rearrange(emb, "b (f h w) c -> b c f h w", f=lat_f, h=LAT_H, w=LAT_W).to(
            self.pipe.param_dtype
        )
        return emb

    def _resolve_prompt(self, clip_dir, prompt):
        """Use the clip's OWN prompt (prompt.txt = its render/deploy caption) so captured KV matches
        deployment. Explicit `prompt` arg wins; DEFAULT_PROMPT only if neither exists."""
        if prompt is not None:
            return prompt
        pf = os.path.join(clip_dir, "prompt.txt")
        if os.path.isfile(pf):
            with open(pf) as fh:
                txt = fh.read().strip()
            if txt:
                return txt
        return DEFAULT_PROMPT
