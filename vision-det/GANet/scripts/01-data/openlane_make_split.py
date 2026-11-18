"""[OpenLane용] OpenLane 분할 목록 생성 (segment 단위 + 프레임 간격 샘플링).

- train: raw/images/training 의 segment 중 test_segments.txt 에 적힌 segment 를 제외한 것
- test : test_segments.txt 의 segment (training 에서 떼어냄)
- val  : raw/images/validation 전체 (공식 validation)
- 각 segment 에서 '이미지와 라벨 JSON 이 모두 있는' 프레임만 이름순 정렬 후 --stride 마다 1장 (첫 프레임부터)
- 출력: <out>/{train,val,test}.txt   (한 줄 = raw 기준 상대 이미지 경로 images/<split>/<segment>/<stem>.jpg)
        <out>/segments_{train,val,test}.txt
- 이미지는 복사하지 않는다 (raw 는 수정하지 않음). 라벨은 raw/lane3d_1000/<split>/<segment>/<stem>.json.
- split 간 공유 segment 0, 모든 이미지/라벨 존재를 assert.
"""
import argparse
import json
import os


def frames(root, split, seg, stride):
    img_dir = os.path.join(root, "images", split, seg)
    lab_dir = os.path.join(root, "lane3d_1000", split, seg)
    lab = {f[:-5] for f in os.listdir(lab_dir) if f.endswith(".json")}
    stems = sorted(f[:-4] for f in os.listdir(img_dir) if f.endswith(".jpg") and f[:-4] in lab)
    return stems[::stride], len(stems)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="OpenLane raw/ 폴더")
    ap.add_argument("--test-segments", required=True, help="test 로 뗄 training segment 목록(txt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=15)
    a = ap.parse_args()

    tr_all = sorted(os.listdir(os.path.join(a.root, "images/training")))
    va = sorted(os.listdir(os.path.join(a.root, "images/validation")))
    test = sorted(x.strip() for x in open(a.test_segments) if x.strip())
    assert set(test) <= set(tr_all), "test segment 가 training 폴더에 없음"
    train = [s for s in tr_all if s not in set(test)]

    sets = {"train": ("training", train), "val": ("validation", va), "test": ("training", test)}
    # split 간 공유 segment 0
    for x in sets:
        for y in sets:
            if x < y:
                assert not set(sets[x][1]) & set(sets[y][1]), f"{x}/{y} 공유 segment 있음"

    os.makedirs(a.out, exist_ok=True)
    for name, (folder, segs) in sets.items():
        lines, total_labeled = [], 0
        for s in segs:
            st, n = frames(a.root, folder, s, a.stride)
            total_labeled += n
            lines += [f"images/{folder}/{s}/{x}.jpg" for x in st]
        for p in lines:  # 이미지·라벨 존재 확인
            assert os.path.exists(os.path.join(a.root, p))
            lab = p.replace("images/", "lane3d_1000/", 1)[:-4] + ".json"
            assert os.path.exists(os.path.join(a.root, lab)), lab
        with open(os.path.join(a.out, f"{name}.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        with open(os.path.join(a.out, f"segments_{name}.txt"), "w") as f:
            f.write("\n".join(segs) + "\n")
        nl = 0
        for p in lines:
            lab = p.replace("images/", "lane3d_1000/", 1)[:-4] + ".json"
            nl += len(json.load(open(os.path.join(a.root, lab)))["lane_lines"])
        print(f"{name}: segments {len(segs)}, 라벨 있는 프레임 {total_labeled} -> 간격 {a.stride} 후 {len(lines)}장, 차선 {nl}개")


if __name__ == "__main__":
    main()
