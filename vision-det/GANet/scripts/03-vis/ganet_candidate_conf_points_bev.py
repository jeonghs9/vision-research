"""12번: 11번(후보점 conf 0.4 기준)의 BEV 이미지 입력 버전.
이미지 -> BEV 이미지(9·10과 같은 고정 비율 사다리꼴) -> backbone -> BC3·BC4·BC5 -> attention(BAC5) -> +BC4 -> +BC3 -> head.
점·배경 모두 BEV 이미지 기준(되돌리기 없음).

실행:
  cd /home/hsjeong/workspace/vision-det/GANet/source
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
    /home/hsjeong/miniconda3/envs/ganet_cu101/bin/python /home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_candidate_conf_points_bev.py
"""
src = open("/home/hsjeong/workspace/vision-det/GANet/scripts/03-vis/ganet_candidate_conf_points.py").read()
reps = [
    ('    rgb = cv2.cvtColor(cv2.resize(cv2.imread(path), (W, H)), cv2.COLOR_BGR2RGB)\n',
     '    rgb = cv2.cvtColor(cv2.resize(cv2.imread(path), (W, H)), cv2.COLOR_BGR2RGB)\n'
     '    _src = np.float32([[0.3 * W, 0.45 * H], [0.7 * W, 0.45 * H], [W, H], [0, H]])\n'
     '    _dst = np.float32([[0, 0], [W, 0], [W, H], [0, H]])\n'
     '    rgb = cv2.warpPerspective(rgb, cv2.getPerspectiveTransform(_src, _dst), (W, H))  # BEV 이미지\n'),
    ('fig.suptitle(f"{name}-11. 차선 후보점 conf (GANet NMS 가로 3칸 최댓값, conf {LOW} 이상)',
     'fig.suptitle(f"{name}-12. BEV 이미지 입력 시 차선 후보점 conf (BEV 이미지 → backbone → BAC5 → +BC4 → +BC3 → head, NMS 가로 3칸 최댓값, conf {LOW} 이상)\\n'),
    ('— 노랑 = {kpt_thr} 초과(키포인트)', '노랑 = {kpt_thr} 초과(키포인트)'),
    ('f"{OUT}/{name}-11. 후보점_conf_0.4기준.png"', 'f"{OUT}/{name}-12. 후보점_conf_0.4기준_BEV이미지입력.png"'),
    ('top=1 - 0.75 / fig.get_figheight()', 'top=1 - 1.2 / fig.get_figheight()'),
]
for a, b in reps:
    assert src.count(a) == 1, a
    src = src.replace(a, b)
exec(compile(src, "ganet_candidate_conf_points_bev", "exec"))
