"""[OpenLane용] 분할 목록(15프레임 간격)의 이미지를 복사하고 CULane 변환 라벨을 split별 폴더로 저장한다.

출력: <splits>/<split>/images/<segment>/<stem>.jpg          (raw 이미지 실제 복사본)
      <splits>/<split>/labels/<segment>/<stem>.lines.txt    (CULane 변환 라벨, 원본 1920x1280 픽셀)
      <splits>/<split>/labels/<segment>/<stem>.cls.txt      (1=실선, 2=점선)
raw 는 수정하지 않는다. 변환 라벨은 processed/culane_lane_ganet 에서 가져온다.

실행:
  PYTHONNOUSERSITE=1 /home/hsjeong/miniconda3/envs/clrkdnet_cu113/bin/python -I \
    /home/hsjeong/workspace/vision-det/GANet/scripts/01-data/openlane_export_splits.py \
    --root $O/raw --processed $O/processed/culane_lane_ganet --splits $O/splits --from-processed-list
(2026-11-18부터 차선 0개 프레임 제외본 = --from-processed-list 로 만든 상태)
"""
import argparse, os, shutil

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True); ap.add_argument("--processed", required=True); ap.add_argument("--splits", required=True)
ap.add_argument("--from-processed-list", action="store_true", help="processed/list/<split>.txt(차선 0개 프레임 제외본) 기준으로 복사")
a = ap.parse_args()
for sp in ("train", "val", "test"):
    d = os.path.join(a.splits, sp)
    for sub in ("images", "labels"):
        shutil.rmtree(os.path.join(d, sub), ignore_errors=True)
    src_list = os.path.join(a.processed, "list", f"{sp}.txt") if a.from_processed_list else os.path.join(a.splits, f"{sp}.txt")
    L = [x.strip() for x in open(src_list) if x.strip()]
    for p in L:  # images/<training|validation>/<segment>/<stem>.jpg
        _, folder, seg, fn = p.split("/"); stem = fn[:-4]
        for sub, src, name in (("images", os.path.join(a.root, p), fn),
                               ("labels", os.path.join(a.processed, p[:-4] + ".lines.txt"), stem + ".lines.txt"),
                               ("labels", os.path.join(a.processed, p[:-4] + ".cls.txt"), stem + ".cls.txt")):
            dst = os.path.join(d, sub, seg); os.makedirs(dst, exist_ok=True)
            shutil.copyfile(src, os.path.join(dst, name))
    print(sp, len(L), "장 복사")
