"""P3 제거 FPN 단계별 히트맵(ganet_fpn_stage_heatmap_noP3.py, 2단계 1번 그림)의 BEV 이미지 입력 버전.
BEV 로 편 이미지를 모델 전체(backbone -> neck(P3 제거) -> head)에 넣어서 그린다. 배경도 BEV 이미지.
사다리꼴은 ganet_fpn_stage_heatmap_bev.py(1단계 10번 그림)와 같은 고정 비율: 윗변 y=0.45H 가운데 40% 폭, 아랫변 화면 전체 폭 -> 화면 전체로 폄.

실행:
  cd /home/hsjeong/workspace/vision-det/GANet/source
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
    /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python /home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_fpn_stage_heatmap_noP3_bev.py
"""
src = open("/home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_fpn_stage_heatmap_noP3.py").read()
reps = [
    ('    rgb = cv2.cvtColor(cv2.resize(cv2.imread(path), (W, H)), cv2.COLOR_BGR2RGB)\n',
     '    rgb = cv2.cvtColor(cv2.resize(cv2.imread(path), (W, H)), cv2.COLOR_BGR2RGB)\n'
     '    _src = np.float32([[0.3 * W, 0.45 * H], [0.7 * W, 0.45 * H], [W, H], [0, H]])\n'
     '    _dst = np.float32([[0, 0], [W, 0], [W, H], [0, H]])\n'
     '    rgb = cv2.warpPerspective(rgb, cv2.getPerspectiveTransform(_src, _dst), (W, H))  # BEV 이미지\n'),
    ('fig.suptitle(f"{name}-1. P3 제거 시 FPN 단계별 히트맵 (P4가 최종 출력, 1줄: 반응 크기, 2·3줄: 실선·점선 conf 예측에 쓰인 곳)"',
     'fig.suptitle(f"{name}-5. BEV 이미지 입력 + P3 제거 시 FPN 단계별 히트맵 (P4가 최종 출력, 1줄: 반응 크기, 2·3줄: 실선·점선 conf 예측에 쓰인 곳)\\n"\n'
     '                 "경로: 이미지 → BEV 이미지(고정 비율 사다리꼴) → backbone → C4·C5 → attention → P4 → head (전부 BEV 이미지 기준)"'),
    ('{name}-1. FPN단계별_히트맵_P3제거.png', '{name}-5. FPN단계별_히트맵_P3제거_BEV이미지입력.png'),
    ('figsize=(25, 9.3)', 'figsize=(25, 9.9)'),
    ('print(f"[프레임{name}]', 'print(f"[BEV 프레임{name}]'),
    ('ax.text(0.5, 0.5, f"실선 conf {kpt_thr} 넘는 칸 없음"', 'ax.text(0.5, 0.5, f"{\'실선\' if r == 1 else \'점선\'} conf {kpt_thr} 넘는 칸 없음"'),
]
for a, b in reps:
    assert src.count(a) == 1, a
    src = src.replace(a, b)
exec(compile(src, "ganet_fpn_stage_heatmap_noP3_bev", "exec"))
