"""FPN 단계별 히트맵의 BEV 이미지 버전: ganet_fpn_stage_heatmap.py 와 같은 7단계 x 3줄을,
BEV 로 편 이미지를 모델 전체(backbone -> neck -> head)에 넣어서 그린다. 배경도 BEV 이미지, 기여도도 BEV 이미지에서 나온 conf 기준.
사다리꼴은 ganet_bev_image_input.py(9번 그림)와 같은 고정 비율: 윗변 y=0.45H 가운데 40% 폭, 아랫변 화면 전체 폭 -> 화면 전체로 폄.

실행:
  cd /home/hsjeong/workspace/vision-det/GANet/source
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
    /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python /home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_fpn_stage_heatmap_bev.py
"""
src = open("/home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_fpn_stage_heatmap.py").read()
reps = [
    ('    rgb = cv2.cvtColor(cv2.resize(cv2.imread(path), (W, H)), cv2.COLOR_BGR2RGB)\n',
     '    rgb = cv2.cvtColor(cv2.resize(cv2.imread(path), (W, H)), cv2.COLOR_BGR2RGB)\n'
     '    _src = np.float32([[0.3 * W, 0.45 * H], [0.7 * W, 0.45 * H], [W, H], [0, H]])\n'
     '    _dst = np.float32([[0, 0], [W, 0], [W, H], [0, H]])\n'
     '    rgb = cv2.warpPerspective(rgb, cv2.getPerspectiveTransform(_src, _dst), (W, H))  # BEV 이미지\n'),
    ('fig.suptitle(f"{name}-1. FPN 단계별 히트맵 (1줄: 반응 크기, 2·3줄: 실선·점선 conf 예측에 쓰인 곳)"',
     'fig.suptitle(f"{name}-10. BEV 이미지 입력 시 FPN 단계별 히트맵 결과 (1줄: 반응 크기, 2·3줄: 실선·점선 conf 예측에 쓰인 곳)\\n"\n'
     '                 "경로: 이미지 → BEV 이미지(고정 비율 사다리꼴) → backbone → C3·C4·C5 → attention → P4 → P3 → head (전부 BEV 이미지 기준)"'),
    ('{name}-1. FPN단계별_히트맵.png', '{name}-10. FPN단계별_히트맵결과_BEV이미지입력.png'),
    ('figsize=(35, 9.3)', 'figsize=(35, 9.9)'),
]
for a, b in reps:
    assert src.count(a) == 1, a
    src = src.replace(a, b)
exec(compile(src, "ganet_fpn_stage_heatmap_bev", "exec"))
