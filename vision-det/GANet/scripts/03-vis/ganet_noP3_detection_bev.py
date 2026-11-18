"""P3 제거 최종 검출(ganet_noP3_detection.py, 2단계 2번 그림)의 BEV 이미지 입력 버전.
세 줄(기존 P3 / P3 제거 offset 그대로 / P3 제거 offset ½) 모두 BEV 로 편 이미지를 모델 전체에 넣고, 결과도 BEV 이미지 위에 그린다.
사다리꼴은 ganet_fpn_stage_heatmap_noP3_bev.py(2단계 5번 그림)와 같은 고정 비율: 윗변 y=0.45H 가운데 40% 폭, 아랫변 화면 전체 폭 -> 화면 전체로 폄.

실행:
  cd /home/hsjeong/workspace/vision-det/GANet/source
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
    /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python /home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_noP3_detection_bev.py
"""
src = open("/home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_noP3_detection.py").read()
reps = [
    ('    rgb = cv2.cvtColor(cv2.resize(cv2.imread(path), (W, H)), cv2.COLOR_BGR2RGB)\n',
     '    rgb = cv2.cvtColor(cv2.resize(cv2.imread(path), (W, H)), cv2.COLOR_BGR2RGB)\n'
     '    _src = np.float32([[0.3 * W, 0.45 * H], [0.7 * W, 0.45 * H], [W, H], [0, H]])\n'
     '    _dst = np.float32([[0, 0], [W, 0], [W, H], [0, H]])\n'
     '    rgb = cv2.warpPerspective(rgb, cv2.getPerspectiveTransform(_src, _dst), (W, H))  # BEV 이미지\n'),
    ('fig.suptitle(f"{name}-2. P3 제거 시 최종 검출 결과 (P4가 최종 출력, 학습 없이 추론만 변경) vs 기존"',
     'fig.suptitle(f"{name}-6. BEV 이미지 입력 + P3 제거 시 최종 검출 결과 (P4가 최종 출력, 학습 없이 추론만 변경) vs 기존\\n"\n'
     '                 "세 줄 모두 BEV 이미지(고정 비율 사다리꼴)를 입력"'),
    ('fig.subplots_adjust(top=1 - 1.2 / (4.2 * n))', 'fig.subplots_adjust(top=1 - 1.6 / (4.2 * n))'),
    ('{name}-2. 최종검출결과_P3제거.png', '{name}-6. 최종검출결과_P3제거_BEV이미지입력.png'),
    ('print(f"[프레임{name}]")', 'print(f"[BEV 프레임{name}]")'),
]
for a, b in reps:
    assert src.count(a) == 1, a
    src = src.replace(a, b)
exec(compile(src, "ganet_noP3_detection_bev", "exec"))
