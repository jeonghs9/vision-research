"""학습된 SA 모델에서 pre_conv 출력 x 와 더해지는 위치 인코딩 pos 의 크기 비교 + attention 출력(gamma*out) 크기."""
import sys
sys.path.insert(0, "./")
import cv2, mmcv, numpy as np, torch
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
from projects.plugin import *  # noqa
G = "/home/hsjeong/workspace/vision-det/GANet"
IMGS = ["/home/hsjeong/workspace/dataset/etc./test_image/aihub_CCTV고속도로/1/Busan_busanportbrdg1_20201028_1300_TUE_8m_NH_highway_TW2_sunny_FHD_000.png",
        "/home/hsjeong/workspace/dataset/etc./si_vietnam_drone/frames/19700101_001037_T/001392.jpg",
        f"{G}/test/tmp/f33/19700101_000816_T_f990(33s).png"]
for tag, c, k in [("기존", "res34_s8_full_2cls", f"{G}/models/drone-ganet-res34-2cls-v1.pth"),
                  ("P3del", "res34_s16_full_2cls_P3del", f"{G}/test/006-res34-2cls-1280-P3del-150ep/epoch_150.pth")]:
    cfg = mmcv.Config.fromfile(f"projects/cfgs/drone/{c}.py"); cfg.model.train_cfg = None; cfg.model.backbone.init_cfg = None
    m = build_detector(cfg.model, test_cfg=cfg.get("test_cfg")); load_checkpoint(m, k, map_location="cpu"); m = m.cuda().eval()
    W, H = cfg.img_scale; mean = np.array(cfg.img_norm_cfg["mean"], np.float32); std = np.array(cfg.img_norm_cfg["std"], np.float32)
    rec = []
    for li, L in enumerate(m.neck.trans_head.attn_layers):
        def hk(mod, inp, out, li=li):
            rec.append((li, out.detach()))
        L.pre_conv.register_forward_hook(hk)
        def hk2(mod, inp, li=li):
            rec.append((10 + li, inp[0].detach().clone()))
        L.final_conv.register_forward_pre_hook(hk2)
    res = {0: [], 1: []}
    for p in IMGS:
        rgb = cv2.cvtColor(cv2.resize(cv2.imread(p), (W, H)), cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(((rgb.astype(np.float32) - mean) / std).transpose(2, 0, 1)).unsqueeze(0).cuda()
        rec.clear()
        with torch.no_grad():
            m.extract_feat(t)
        d = dict(rec)
        for li in (0, 1):
            x = d[li]
            att = d[10 + li] - x   # final_conv 입력 = gamma*out + (x+pos)
            pos = m.neck.trans_head.pos_embeds[li].to(x.device)
            # hook 출력은 forward 안에서 x += pos 로 in-place 로 바뀌므로 pos 를 빼서 원래 x 를 복원
            x0 = x - pos
            res[li].append((x0.pow(2).sum(1).sqrt().mean().item(), pos.pow(2).sum(1).sqrt().mean().item(), att.pow(2).sum(1).sqrt().mean().item()))
    for li in (0, 1):
        a = np.array(res[li])
        print(f"{tag} 층{li+1}: |x| 평균 {a[:,0].mean():.2f}, |pos| 평균 {a[:,1].mean():.2f}, pos/x 비 {(a[:,1]/a[:,0]).mean():.2f}, |gamma*SA출력| {a[:,2].mean():.3f} (x 대비 {(a[:,2]/a[:,0]).mean():.3f})")
