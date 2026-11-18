"""P3del+SA / P3del−SA / P3del+DA(query) 최종 검출 (18번 = 원본 입력, 19번 = BEV 입력). 16·17번 히트맵과 같은 모델.
줄: P3del+SA(test/006 ep150) / P3del−SA(test/008 ep110) / P3del+DA query(test/009 ep150),  칸: 실선 conf, 점선 conf, 최종 검출.
ganet_sa_detection.py(14·15번)에서 모델만 바꾼 것. config 캐시 문제(문서 14 6절) 때문에 build 에서 projects.cfgs 캐시를 지우고 strict 로드.
검출 함수는 ganet_p3del_trained_compare.py(3번 그림), BEV 사다리꼴은 7번 그림과 같다.

실행:
  cd /home/hsjeong/workspace/vision-det/GANet/source
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
    /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python /home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_da_detection.py
"""
_src_txt = open("/home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_p3del_trained_compare.py").read().split("\n# ---- res34 P3del 히트맵용 hook")[0]
_old = _src_txt[_src_txt.index("MODELS = ["):_src_txt.index("CLS_NAME =")]
_new = """MODELS = [
    ("P3del+SA\\n(원래 구조)", "res34_s16_full_2cls_P3del", f"{G}/test/006-res34-2cls-1280-P3del-150ep/epoch_150.pth"),
    ("P3del−SA\\n(SA 제거)", "res34_s16_full_2cls_P3del_noSA", f"{G}/test/008-res34-2cls-1280-P3del-noSA-150ep/epoch_110.pth"),
    ("P3del+DA\\n(SA→DA 교체, query)", "res34_s16_full_2cls_P3del_DAq", f"{G}/test/009-res34-2cls-1280-P3del-DAq-150ep/epoch_150.pth"),
]
"""
_src_txt = _src_txt.replace(_old, _new)
_a = 'def build(cfg_name, ckpt):\n'
_src_txt = _src_txt.replace(_a, _a + '    for _k in [k for k in sys.modules if k.startswith("projects.cfgs")]:\n        del sys.modules[_k]\n')
_a = '    load_checkpoint(m, ckpt, map_location="cpu")\n'
assert _a in _src_txt
_src_txt = _src_txt.replace(_a, '    print(cfg_name, m.load_state_dict(torch.load(ckpt, map_location="cpu")["state_dict"], strict=True))\n')
exec(_src_txt)
kpt_thr = models[0][1].bbox_head.kpt_thr

os.makedirs(OUTDIR, exist_ok=True)
_src = np.float32([[0.3 * W, 0.45 * H], [0.7 * W, 0.45 * H], [W, H], [0, H]])
_dst = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
M_bev = cv2.getPerspectiveTransform(_src, _dst)
for name, path in JOBS:
    orig = cv2.cvtColor(cv2.resize(cv2.imread(path), (W, H)), cv2.COLOR_BGR2RGB)
    bev = cv2.warpPerspective(orig, M_bev, (W, H))
    for itag, rgb, fno in [("원본", orig, "18"), ("BEV", bev, "19")]:
        res = []
        t = torch.from_numpy(((rgb.astype(np.float32) - mean) / std).transpose(2, 0, 1)).unsqueeze(0).cuda(0)
        for tag, m, _ in models:
            hm, rows, lanes = detect(m, t)
            res.append((f"{itag} 입력\n{tag}", rgb, hm, rows, lanes))
        print(f"[프레임{name} {itag}]")
        for tag, _, hm, rows, _ in res:
            print(f"   {tag.replace(chr(10), ' '):34s}: " + ", ".join(
                f"{CLS_NAME[c]} 선 {len(rows[c][2])}개 {rows[c][2]} 버려진 점 {rows[c][3]}" for c in range(2)))

        n = len(res)
        fig, axes = plt.subplots(n, 3, figsize=(21, 4.2 * n), gridspec_kw=dict(hspace=0.1, wspace=0.03))
        for r, (tag, rgb, hm, _, lanes) in enumerate(res):
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            bg = np.stack([gray] * 3, -1).astype(np.float32) / 255 * 0.6 + 0.2
            for c in range(2):
                ax = axes[r, c]
                ax.imshow(bg)
                ax.imshow(cv2.resize(hm[c], (W, H), interpolation=cv2.INTER_NEAREST), cmap="jet", alpha=0.55, vmin=0, vmax=1)
                ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title(f"{CLS_NAME[c]} conf", fontsize=15)
            axes[r, 0].set_ylabel(tag, fontsize=15, rotation=0, ha="right", va="center", labelpad=12)
            ax = axes[r, 2]
            ax.imshow(rgb)
            for c, pts in lanes:
                ax.plot(pts[:, 0], pts[:, 1], "-", color=CLS_COLOR[c], lw=3)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title("최종 검출 (초록 실선 · 빨강 점선)", fontsize=15)
        fig.subplots_adjust(top=1 - 1.2 / (4.2 * n))
        fig.suptitle(f"{name}-{fno}. {itag} 입력 최종 검출 — P3del+SA / P3del−SA(SA 제거) / P3del+DA(SA→DA 교체, query) (res34)", fontsize=17)
        sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, 1))
        cb = fig.colorbar(sm, ax=axes[:, 0:2], fraction=0.02, pad=0.01, ticks=[0, kpt_thr, 1])
        cb.ax.set_yticklabels(["0", f"{kpt_thr}", "1"], fontsize=12)
        fig.savefig(f"{OUTDIR}/{name}-{fno}. 최종검출결과_{itag}입력_SAvs없음vsDA.png", dpi=90, bbox_inches="tight", pad_inches=0.2)
        plt.close(fig)
