"""P3del 학습 모델 비교(ganet_p3del_trained_compare.py, 2단계 3·4번 그림)의 BEV 이미지 입력 버전.
{n}-5. FPN단계별_히트맵_P3del학습_BEV이미지입력.png : res34 P3del(epoch_150) FPN 단계별 히트맵
{n}-6. 최종검출결과_P3del학습_BEV이미지입력.png      : res34 기존 / res34 P3del 최종 검출 (res34 고정)
모든 모델에 BEV 로 편 이미지를 넣고, 결과도 BEV 이미지 위에 그린다.
사다리꼴은 1단계 10번 그림과 같은 고정 비율: 윗변 y=0.45H 가운데 40% 폭, 아랫변 화면 전체 폭 -> 화면 전체로 폄.

실행:
  cd /home/hsjeong/workspace/vision-det/GANet/source
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
    /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python /home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_p3del_trained_compare_bev.py
"""
src = open("/home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_p3del_trained_compare.py").read()
reps = [
    ('    rgb = cv2.cvtColor(cv2.resize(cv2.imread(path), (W, H)), cv2.COLOR_BGR2RGB)\n',
     '    rgb = cv2.cvtColor(cv2.resize(cv2.imread(path), (W, H)), cv2.COLOR_BGR2RGB)\n'
     '    _src = np.float32([[0.3 * W, 0.45 * H], [0.7 * W, 0.45 * H], [W, H], [0, H]])\n'
     '    _dst = np.float32([[0, 0], [W, 0], [W, H], [0, H]])\n'
     '    rgb = cv2.warpPerspective(rgb, cv2.getPerspectiveTransform(_src, _dst), (W, H))  # BEV 이미지\n'),
    ('fig.suptitle(f"{name}-3. P3del 구조로 다시 학습한 모델의 최종 검출 결과 vs 기존", fontsize=17)',
     'fig.suptitle(f"{name}-6. BEV 이미지 입력 시 P3del 학습 모델의 최종 검출 결과 vs 기존\\n"\n'
     '                 "두 모델 모두 BEV 이미지(고정 비율 사다리꼴)를 입력", fontsize=17)'),
    ('fig.subplots_adjust(top=1 - 1.2 / (4.2 * n))', 'fig.subplots_adjust(top=1 - 1.6 / (4.2 * n))'),
    ('{name}-3. 최종검출결과_P3del학습.png', '{name}-6. 최종검출결과_P3del학습_BEV이미지입력.png'),
    ('fig.suptitle(f"{name}-4. P3del 구조로 다시 학습한 res34 모델의 FPN 단계별 히트맵 "\n'
     '                 "(1줄: 반응 크기, 2·3줄: 실선·점선 conf 예측에 쓰인 곳)", fontsize=18)',
     'fig.suptitle(f"{name}-5. BEV 이미지 입력 시 P3del 학습 res34 모델의 FPN 단계별 히트맵 "\n'
     '                 "(1줄: 반응 크기, 2·3줄: 실선·점선 conf 예측에 쓰인 곳)\\n"\n'
     '                 "경로: 이미지 → BEV 이미지(고정 비율 사다리꼴) → backbone → C3 → C4·C5 → attention → P4 → head (전부 BEV 이미지 기준)", fontsize=18)'),
    ('figsize=(30, 9.3)', 'figsize=(30, 9.9)'),
    ('{name}-4. FPN단계별_히트맵_P3del학습.png', '{name}-5. FPN단계별_히트맵_P3del학습_BEV이미지입력.png'),
    ('print(f"[프레임{name}]")', 'print(f"[BEV 프레임{name}]")'),
]
for a, b in reps:
    assert src.count(a) == 1, a
    src = src.replace(a, b)
exec(compile(src, "ganet_p3del_trained_compare_bev", "exec"))
