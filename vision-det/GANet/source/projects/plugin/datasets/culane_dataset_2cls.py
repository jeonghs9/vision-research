# ---------------------------------------------------------------------------- #
# 프로젝트 추가 (vision-det/GANet, GANet 원본이 아님)
# ---------------------------------------------------------------------------- #
"""CulaneDataset을 상속해서 차선별 실제 클래스(실선=1/점선=2)를 읽어온다.

원본 CulaneDataset.load_labels()는 id_classes를 [1,1,...]로 하드코딩한다.
여기서는 <stem>.lines.txt와 같은 폴더의 <stem>.cls.txt(같은 순서, 한 줄에
정수 하나)를 추가로 읽어서 실제 클래스를 돌려준다.
(scripts/01-data/polyline_to_culane.py --write-class 로 생성)
"""
from mmdet.datasets.builder import DATASETS

from .culane_dataset import CulaneDataset


@DATASETS.register_module()
class CulaneDataset2Cls(CulaneDataset):

    def load_labels(self, idx, offset_x, offset_y):
        anno_file = self.img_prefix + "/" + self.img_infos[idx]['anno_file']
        lanes = []
        with open(anno_file, 'r') as anno_f:
            for line in anno_f.readlines():
                coords = []
                coords_str = line.strip().split(' ')
                for i in range(len(coords_str) // 2):
                    coord_x = float(coords_str[2 * i]) + offset_x
                    coord_y = float(coords_str[2 * i + 1]) + offset_y
                    coords.append(coord_x)
                    coords.append(coord_y)
                if len(coords) > 3:
                    lanes.append(coords)

        cls_file = anno_file.replace('.lines.txt', '.cls.txt')
        id_classes = self._read_classes(cls_file, len(lanes))
        id_instances = [i + 1 for i in range(len(lanes))]
        return lanes, id_classes, id_instances

    @staticmethod
    def _read_classes(cls_file, n_lanes):
        try:
            with open(cls_file, 'r') as f:
                vals = [int(x) for x in f.read().split()]
        except FileNotFoundError:
            vals = []
        if len(vals) != n_lanes:
            # 개수가 안 맞으면(예: .cls.txt 없이 생성된 옛 데이터) 전부 클래스 1로 안전하게 대체
            vals = [1] * n_lanes
        return vals
