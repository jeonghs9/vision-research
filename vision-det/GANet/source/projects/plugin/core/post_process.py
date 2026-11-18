# --------------------------------------------------------
# GANet
# Copyright (c) 2022 SenseTime
# @Time    : 2022/04/23
# @Author  : Jinsheng Wang
# @Email   : jswang@stu.pku.edu.cn
# --------------------------------------------------------

import os
import math
import copy
import random

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from mmcv import Timer
from scipy.signal import savgol_filter
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


def remove_lane_outliers(pts, degree=2, residual_threshold=20.0, min_pts=8):
    """RANSAC으로 전체 선 모양에 맞는 다항식을 찾고, 거기서 크게 벗어나는
    점(다른 물체에서 잘못 섞여 들어온 점)을 통째로 제거한다.

    savgol 평활화(`smooth_lane_points`)는 "이미 올바르게 묶인 점들의 작은
    흔들림"만 다듬을 뿐이다 -- 안전지대 사선 줄무늬처럼 차선이 아닌데
    차선처럼 생긴 무늬가 여러 개 연속으로 있으면, 서로 다른 줄무늬에서 나온
    점들이 거리 기준 그룹핑(`cluster_thr`) 단계에서 "같은 차선"으로 잘못
    섞여 들어와 큰 폭(수십~백여 px)으로, 때로는 한 뭉치로 지그재그가
    생긴다 -- 이건 미세 흔들림이 아니라 애초에 잘못 섞인 점이라 평활화로는
    못 줄인다.

    처음엔 "전체 점에 다항식을 한 번에 피팅하고 벗어난 점을 제거"하는
    시그마 클리핑을 시도했는데, 이상치가 뭉쳐서(연속 구간) 끼어든 경우
    **첫 피팅 자체가 이상치 쪽으로 끌려가서** 기준(잔차의 중앙값)까지
    오염되고, 결국 아무것도 못 거르는 걸 합성 테스트로 확인했다(원본 50개
    중 뭉친 이상치 10개가 하나도 안 걸러짐). RANSAC은 "무작위로 뽑은 소수
    점들로만 피팅 → 그 피팅에 맞는 점이 얼마나 많은지로 좋은 피팅을 선택"
    하는 방식이라, 이상치가 전체의 상당 부분(실측 20%)을 차지해도 흔들리지
    않는다 -- 같은 합성 테스트에서 뭉친 이상치 10개를 정확히 걸러냄, 정상
    곡선·일반 노이즈는 거의 안 건드림(확인 완료).
    """
    pts = np.array(pts, dtype=np.float64)
    order = np.argsort(pts[:, 1])
    pts = pts[order]
    n = len(pts)
    if n < min_pts:
        return pts
    model = make_pipeline(
        PolynomialFeatures(degree),
        RANSACRegressor(residual_threshold=residual_threshold,
                         min_samples=degree + 2, random_state=0))
    try:
        model.fit(pts[:, 1:2], pts[:, 0])
    except ValueError:
        return pts  # 점이 너무 적거나 퇴화된 경우 안전하게 원본 유지
    inlier_mask = model.named_steps["ransacregressor"].inlier_mask_
    if inlier_mask.sum() < 2:
        return pts
    return pts[inlier_mask]


def smooth_lane_points(pts, window_frac=0.15, min_window=7, polyorder=2):
    """y순으로 정렬한 뒤 x좌표에 Savitzky-Golay 필터를 적용해 지그재그를 줄인다.

    처음엔 GT 타깃 생성용 ploy_fitting_cube()(큐빅 스플라인)를 재사용했는데,
    그 함수는 scipy.splrep에 s(평활화 강도)를 안 넘겨서 기본값 s=0 -- 즉
    "평활화"가 아니라 "노이즈 있는 점을 전부 그대로 통과하는 보간"이라
    지그재그가 전혀 안 줄었다(실측으로 확인). 그래서 실제 평활화 필터로
    교체 -- y는 그대로 두고 x만 윈도우 내 다항식 적합으로 매끈하게 만든다.
    점이 너무 적으면(윈도우를 못 만들면) 원본 그대로 반환.
    """
    pts = np.array(pts, dtype=np.float64)
    order = np.argsort(pts[:, 1])
    pts = pts[order]
    n = len(pts)
    window = max(min_window, int(round(n * window_frac)))
    if window % 2 == 0:
        window += 1
    if window > n:
        window = n if n % 2 == 1 else n - 1
    if window < polyorder + 2 or window > n:
        return [tuple(p) for p in np.round(pts).astype(int)]
    pts[:, 0] = savgol_filter(pts[:, 0], window_length=window, polyorder=polyorder)
    return [tuple(p) for p in np.round(pts).astype(int)]


def compute_locations(shape, device):
    pos = torch.arange(
        0, shape[-1], step=1, dtype=torch.float32, device=device)
    pos = pos.reshape((1, 1, -1))
    pos = pos.repeat(shape[0], shape[1], 1)
    return pos


def choose_highest_score(group):
    highest_score = -1
    highest_idx = -1
    for idx, _, score in group:
        if score > highest_score:
            highest_idx = idx
    return highest_idx


def choose_mean_point(group):
    group_ = np.array(group).reshape(-1, 2)
    mean_point = np.mean(group_, axis=0, dtype=int)
    return mean_point


def cal_dis(p1, p2):
    result = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    return result


def search_groups(coord, groups, thr):
    for idx_group, group in enumerate(groups):
        for group_point in group:
            if isinstance(group_point, tuple):
                group_point_coord = group_point[-1]  # center
            else:
                group_point_coord = group_point
            if cal_dis(coord, group_point_coord) <= thr:
                return idx_group
    return -1


def search_groups_by_centers(coord, cluster_centers, cluster_thr):
    for idx_group, cluster_center in enumerate(cluster_centers):
        dis = cal_dis(coord, cluster_center)
        if dis <= cluster_thr:
            return idx_group
    return -1


def group_points(seeds, center_seeds, thr, by_center_thr=None):
    def update_coords(points, cluster_centers, thr=5):
        groups = []
        groups_centers = []
        groups_centers_mean = []

        # group centers first
        for cluster_center in cluster_centers:
            idx_group = search_groups(cluster_center, groups_centers, thr)
            if idx_group < 0:
                groups_centers.append([cluster_center])
            else:
                groups_centers[idx_group].append(cluster_center)

        # choose mean center
        for group_center in groups_centers:
            group_center_new = choose_mean_point(group_center)
            groups_centers_mean.append(group_center_new)

        for idx, (coord, score, center) in enumerate(points):
            idx_group = search_groups(center, groups, thr)
            if idx_group < 0:
                groups.append([(idx, coord, score, center)])
            else:
                groups[idx_group].append((idx, coord, score, center))  # belong to one line or nearby points
        # TODO
        # print('group size: {}   cluster thr: {}'.format(len(groups), thr))
        return groups, groups_centers_mean

    # TODO group by center
    def update_coords_by_center(points, cluster_centers, thr=5):
        groups = []
        groups_centers = []
        groups_centers_mean = []

        # group centers first
        for cluster_center in cluster_centers:
            idx_group = search_groups(cluster_center, groups_centers, thr)
            if idx_group < 0:
                groups_centers.append([cluster_center])
            else:
                groups_centers[idx_group].append(cluster_center)

        # choose mean center
        for group_center in groups_centers:
            group_center_new = choose_mean_point(group_center)
            groups_centers_mean.append(group_center_new)
            groups.append([])

        # group key points by center
        for idx, (coord, score, center) in enumerate(points):
            idx_group = search_groups_by_centers(coord=center,
                                                 cluster_centers=groups_centers_mean,
                                                 cluster_thr=thr)
            if idx_group == -1:
                continue
            groups[idx_group].append((idx, coord, score, center))
        return groups, groups_centers_mean

    points = [(item['align'], item['score'], item['center']) for item in seeds]
    # centers = [(item['coord'], item['score']) for item in center_seeds]
    if by_center_thr is None:
        groups, groups_centers = update_coords(points=points,
                                               cluster_centers=center_seeds,
                                               thr=thr)
    else:
        groups, groups_centers = update_coords_by_center(points=points,
                                                         cluster_centers=center_seeds,
                                                         thr=by_center_thr)
    return groups, groups_centers


def group_points_fast(seeds, center_seeds, thr, by_center_thr=None):
    def update_coords_fast(points, thr=5):
        groups = []
        groups_centers = []
        for idx, (align, center) in enumerate(points):
            idx_group = search_groups(center, groups, thr)
            if idx_group < 0:
                groups.append([(idx, align, center)])
            else:
                groups[idx_group].append((idx, align, center))  # belong to one line or nearby points
        # TODO
        # print('group size: {}   cluster thr: {}'.format(len(groups), thr))
        return groups, groups_centers

    def update_coords_fast_by_center(points, cluster_centers, thr=5, by_center_thr=5):
        groups = []
        groups_centers = []
        groups_centers_mean = []
        # group centers first
        for cluster_center in cluster_centers:
            idx_group = search_groups(cluster_center, groups_centers, thr)
            if idx_group < 0:
                groups_centers.append([cluster_center])
            else:
                groups_centers[idx_group].append(cluster_center)
        # choose mean center
        for group_center in groups_centers:
            group_center_new = choose_mean_point(group_center)
            groups_centers_mean.append(group_center_new)
            groups.append([])

        # group key points by center
        for idx, (align, center) in enumerate(points):
            idx_group = search_groups_by_centers(center, groups_centers_mean, by_center_thr)
            if idx_group < 0:
                # groups.append([(idx, align, center)])
                continue
            else:
                groups[idx_group].append((idx, align, center))  # belong to one line or nearby points
        # TODO
        # print('group size: {}  cluster thr: {} cluster by center thr {}'.format(len(groups),
        #                                                                         thr,
        #                                                                         by_center_thr))
        return groups, groups_centers_mean


    if by_center_thr is None:
        groups, groups_centers = update_coords_fast(points=seeds, thr=thr)
    else:
        groups, groups_centers = update_coords_fast_by_center(points=seeds,
                                                              cluster_centers=center_seeds,
                                                              thr=by_center_thr,
                                                              by_center_thr=by_center_thr)
    return groups, groups_centers


class PostProcessor(object):

    def __init__(self,
                 points_thr=5,
                 hm_down_scale=16,
                 cluster_thr=4,
                 cluster_by_center_thr=4,
                 group_fast=True,
                 crop_bbox=None,
                 smooth=True,
                 ):
        self.points_thr = points_thr
        self.hm_down_scale = hm_down_scale
        self.cluster_thr = cluster_thr
        self.cluster_by_center_thr = cluster_by_center_thr
        self.group_fast = group_fast
        self.crop_bbox = crop_bbox
        # 예측 점들을 잇기만 하면 특히 점선에서 지그재그가 생긴다(점 하나하나가
        # heatmap 셀 단위로 독립 추정되고, 그 흔들림이 원본 해상도로 24배
        # 가량 증폭됨) -- smooth_lane_points()로 x좌표에 평활화 필터를 적용.
        self.smooth = smooth

    def lane_post_process(self, kpt_groups, cpt_groups):
        lanes = []
        cluster_centers = []
        for lane_idx, group in enumerate(kpt_groups):
            points = []
            centers = []
            if len(group) > 1:
                for point in group:
                    points.append([point[1][0] * self.hm_down_scale, point[1][1] * self.hm_down_scale])
                    centers.append([point[-1][0] * self.hm_down_scale, point[-1][1] * self.hm_down_scale])
                # points = ploy_fitting_cube(points, h=320, w=800, sample_num=150)
                lanes.append(
                    dict(
                        id_class=lane_idx,
                        points=points,
                        centers=centers,
                    )
                )
        for center_idx, center in enumerate(cpt_groups):
            cluster_center = [center[0] * self.hm_down_scale, center[1] * self.hm_down_scale]
            cluster_centers.append(
                dict(
                    id_class=center_idx,
                    center=cluster_center,
                )
            )
        return lanes, cluster_centers
    
    def __call__(self, output, img_metas):
        """output = [(class_id(1-indexed), cpt_seeds, kpt_seeds), ...] -- 클래스별로 하나씩.

        클래스마다 독립적으로 묶고(group) 거리 필터링까지 한 뒤 합친다.
        result_classes가 result와 같은 길이/순서로 각 차선의 클래스를 담는다
        (1클래스일 때도 항상 포함 -- 전부 1이라 결과엔 영향 없음).
        """
        all_result, all_vc, all_cc, all_classes = [], [], [], []
        for class_id, cpt_seeds, kpt_seeds in output:
            if self.group_fast is True:
                kpt_groups, cpt_groups = group_points_fast(kpt_seeds,
                                                           cpt_seeds,
                                                           self.cluster_thr,
                                                           self.cluster_by_center_thr)
            else:
                kpt_groups, cpt_groups = group_points(kpt_seeds,
                                                      cpt_seeds,
                                                      self.cluster_thr,
                                                      self.cluster_by_center_thr)

            lanes, cluster_centers = self.lane_post_process(kpt_groups, cpt_groups)

            result, virtual_center, cluster_center = adjust_result(
                lanes=lanes,
                centers=cluster_centers,
                crop_bbox=self.crop_bbox,
                img_metas=img_metas,
                points_thr=self.points_thr,
                smooth=self.smooth,
            )
            all_result.extend(result)
            all_vc.extend(virtual_center)
            all_cc.extend(cluster_center)
            all_classes.extend([class_id] * len(result))

        return dict(
            final_dict_list = [all_result, all_vc, all_cc, all_classes],
            img_metas = img_metas,
        )

def adjust_result(lanes, centers, crop_bbox, img_metas, points_thr, smooth=True):
    img_shape=img_metas['img_shape']
    ori_shape=img_metas['ori_shape']
    offset_x=img_metas['offset_x']
    offset_y=img_metas['offset_y']
    h_img, w_img = img_shape[:2]
    h_ori_img, w_ori_img = ori_shape[:2]
    if crop_bbox != None: # culane和tusimple是先crop到这个shape后面再进行操作,经过这个比例变换就可以
        ratio_x = (crop_bbox[2] - crop_bbox[0]) / w_img
        ratio_y = (crop_bbox[3] - crop_bbox[1]) / h_img
        offset_x, offset_y = crop_bbox[:2]
    else: # curvelane没有cropbox，直接reshape
        ratio_x = (w_ori_img+offset_x) / w_img
        ratio_y = (h_ori_img+offset_y) / h_img
        offset_x, offset_y = -offset_x, -offset_y
        # 这个offset说的是crop_box，这里就不做了，后面可视化的时候再做

    results = []
    virtual_centers = []
    cluster_centers = []
    if lanes is not None:
        for key in range(len(lanes)):
            pts = []
            cts = []
            for pt in lanes[key]['points']:
                pt[0] = int(pt[0] * ratio_x + offset_x)
                pt[1] = int(pt[1] * ratio_y + offset_y)
                pts.append(tuple(pt))
            for ct in lanes[key]['centers']:
                ct[0] = int(ct[0] * ratio_x + offset_x)
                ct[1] = int(ct[1] * ratio_y + offset_y)
                cts.append(tuple(ct))
            # print('lane {} ====== \npoint nums {}'.format(key, len(pts)))
            # print('lane {} ====== \n point coord {}  \nvirtual center coord {}'.format(key, pts, cts))
            if len(pts) > points_thr:
                if smooth:
                    pts = remove_lane_outliers(pts)
                    pts = smooth_lane_points(pts)
                results.append(pts)
                virtual_centers.append(cts)
        # print('lane number:{}  virtual center number:{}'.format(len(results), len(virtual_centers)))
    if centers is not None:
        for center in centers:
            center_coord = center['center']
            center_coord[0] = int(center_coord[0] * ratio_x + offset_x)
            center_coord[1] = int(center_coord[1] * ratio_y + offset_y)
            cluster_centers.append(tuple(center_coord))

    return results, virtual_centers, cluster_centers