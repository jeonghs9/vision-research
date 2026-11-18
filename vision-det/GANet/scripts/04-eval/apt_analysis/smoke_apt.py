"""APT 코드 검증: (1) 병합 0 이면 기존 모델과 출력 동일 (2) mask 방식 = 토큰만 모아 돌린 SA (3) 기존 체크포인트 strict 로드 (4) params/속도/backward.
실행: cd GANet/source && PYTHONNOUSERSITE=1 python /home/hsjeong/workspace/vision-det/GANet/scripts/04-eval/apt_analysis/smoke_apt.py"""
import sys, time, copy; sys.path.insert(0, './')
import torch, torch.nn.functional as F
from mmcv import Config
from mmdet.models import build_detector
import projects.plugin  # noqa
from projects.plugin.models import dcn_fpn as D

G = '/home/hsjeong/workspace/vision-det/GANet'
def load_cfg(p):
    for k in [k for k in sys.modules if k.startswith('projects.cfgs')]:
        del sys.modules[k]
    return Config.fromfile(p)
def build(p):
    c = load_cfg(p); m = build_detector(c.model).cuda().eval(); return m, c
torch.manual_seed(0)
def tensors(o):
    if torch.is_tensor(o): return [o]
    if isinstance(o, dict): o = list(o.values())
    if isinstance(o, (list, tuple)): return [t for v in o for t in tensors(v)]
    return []
def maxdiff(a, b):
    return max((x.float() - y.float()).abs().max().item() for x, y in zip(tensors(a), tensors(b)))
from mmdet.datasets import build_dataset
c0 = load_cfg('projects/cfgs/drone/res34_s16_full_2cls_P3del.py')
tr = c0.data.val.copy(); tr['pipeline'] = c0.val_pipeline; ds = build_dataset(tr)
def im(i):
    d = ds[i]['img']; d = d[0] if isinstance(d, list) else d; return d.data
img = torch.stack([im(0), im(5)]).cuda()

# (3) 기존 체크포인트 strict 로드 (기본값이 안 바뀌었는지)
for cfgp, ck in [('res34_s16_full_2cls_P3del', 'test/006-res34-2cls-1280-P3del-150ep/epoch_150.pth'),
                 ('res34_s16_full_2cls_P3del_noSA', 'test/008-res34-2cls-1280-P3del-noSA-150ep/epoch_110.pth'),
                 ('res34_s16_full_2cls_P3del_DAq', 'test/009-res34-2cls-1280-P3del-DAq-150ep/epoch_150.pth')]:
    m, _ = build(f'projects/cfgs/drone/{cfgp}.py')
    sd = torch.load(f'{G}/{ck}', map_location='cpu')['state_dict']
    print('strict', cfgp, m.load_state_dict(sd, strict=True))

pairs = [('res34_s16_full_2cls_P3del', 'res34_s16_full_2cls_P3del_APT', 'test/006-res34-2cls-1280-P3del-150ep/epoch_150.pth'),
         ('res34_s16_full_2cls_P3del_DAq', 'res34_s16_full_2cls_P3del_DAq_APT', 'test/009-res34-2cls-1280-P3del-DAq-150ep/epoch_150.pth'),
         ('res34_s16_full_2cls_P3del_noSA', 'res34_s16_full_2cls_P3del_noSA_APT', 'test/008-res34-2cls-1280-P3del-noSA-150ep/epoch_110.pth')]
for base, apt, ck in pairs:
    mb, _ = build(f'projects/cfgs/drone/{base}.py'); ma, ca = build(f'projects/cfgs/drone/{apt}.py')
    sd = torch.load(f'{G}/{ck}', map_location='cpu')['state_dict']
    mb.load_state_dict(sd, strict=True)
    r = ma.load_state_dict(sd, strict=False)
    print(f'\n== {apt}: 새 키 {r.missing_keys}  남는 키 {r.unexpected_keys}')
    with torch.no_grad():
        # (1) 병합 0 (문턱값 -1) -> 기존과 동일해야 함
        ma.neck.apt_cfg = dict(ma.neck.apt_cfg, threshold=-1.0)
        fb = mb.extract_feat(img); fa = ma.extract_feat(img)
        print('  병합0 최대차이', maxdiff(fa, fb))
        ma.neck.apt_cfg = dict(ma.neck.apt_cfg, threshold=ca.apt_threshold)
        mc, mf = D.apt_merge_mask(img, 32, ca.apt_threshold, ca.img_norm_cfg['mean'], ca.img_norm_cfg['std'])
        print('  병합 묶음 수/이미지', mc.flatten(1).sum(1).tolist(), ' 토큰 수', ((~mf).flatten(1).sum(1) + mc.flatten(1).sum(1)).tolist())
    # (2) SA mask 방식 = 실제 토큰만 모아서 계산 (gamma/zero_conv 를 임의값으로 바꿔 경로가 다 쓰이게)
    if apt == 'res34_s16_full_2cls_P3del_APT':
        L = ma.neck.trans_head.attn_layers[0]
        with torch.no_grad():
            L.gamma.fill_(0.7); torch.nn.init.normal_(L.apt_embed.zero_conv.weight, std=0.05)
            x = torch.randn(2, 512, 23, 40, device='cuda')
            pos = ma.neck.trans_head.pos_embeds[0].cuda(); posc = D.apt_downsample(pos)
            got = L(x, pos, posc, (mc, mf))
            fine = L.pre_conv(x); coarse = L.apt_embed(L.pre_conv(D.apt_downsample(x)), fine)
            fine = fine + pos; coarse = coarse + posc
            errs = []
            for b in range(2):
                tok = torch.cat([fine[b].flatten(1)[:, ~mf[b].flatten()], coarse[b].flatten(1)[:, mc[b].flatten()]], 1)  # C x T
                q = L.query_conv(tok[:, :, None, None].permute(1, 0, 2, 3)).flatten(1)  # T x c
                k = L.key_conv(tok[:, :, None, None].permute(1, 0, 2, 3)).flatten(1)
                v = L.value_conv(tok[:, :, None, None].permute(1, 0, 2, 3)).flatten(1)
                o = torch.softmax(q @ k.t(), -1) @ v  # T x C
                nf = int((~mf[b]).sum())
                out_f = torch.zeros_like(fine[b]).flatten(1); out_f[:, ~mf[b].flatten()] = o[:nf].t()
                out_c = torch.zeros_like(coarse[b]).flatten(1); out_c[:, mc[b].flatten()] = o[nf:].t()
                out = D.apt_repeat(out_c.view(1, 64, 12, 20), out_f.view(1, 64, 23, 40), mf[b:b+1])
                xx = D.apt_repeat(coarse[b:b+1], fine[b:b+1], mf[b:b+1])
                ref = L.final_conv(L.gamma * out + xx)
                errs.append((ref - got[b:b+1]).abs().max().item())
            print('  SA mask방식 vs 토큰만 모은 계산 최대차이', errs)
    # (4) params / 속도 / backward
    n = sum(p.numel() for n_, p in ma.named_parameters() if not n_.startswith('bbox_head')) / 1e6
    one = img[:1]
    with torch.no_grad():
        for _ in range(10): ma.extract_feat(one)
        torch.cuda.synchronize(); t = time.time()
        for _ in range(50): ma.extract_feat(one)
        torch.cuda.synchronize(); ms = (time.time() - t) / 50 * 1000
    ma.train(); out = ma.extract_feat(img); sum(t.float().mean() for t in tensors(out)).backward()
    g = [n_ for n_, p in ma.named_parameters() if 'apt_embed' in n_ and p.grad is not None and p.grad.abs().sum() > 0]
    print(f'  backbone+neck params {n:.3f}M, {ms:.2f}ms(batch1), grad 받은 apt 파라미터 {g}')
