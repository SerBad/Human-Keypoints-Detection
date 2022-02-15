# -*-coding: utf-8 -*-
"""
    @Project: torch-Human-Pose-Estimation-Pipeline
    @File   : demo.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-11-08 15:02:19
"""
import sys
import os

import cv2
import numpy as np
import argparse
import time
import shutil
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import numpy as np
import pandas
from torch.multiprocessing import Pool, Process, set_start_method

sys.path.append(os.path.dirname(__file__))
from configs import val_config
from libs.detector.libs.detector.detector import Detector
from utils import image_processing, debug, file_processing, torch_tools
from models import inference

sys.path.append("/home/zhou/Documents/python/Human-Keypoints-Detection/libs/detector/libs/detector")
project_root = os.path.dirname(__file__)


class PoseEstimation(inference.PoseEstimation):
    """
     mpii_keypoints_v2 = {0: "r_ankle", 1: "r_knee", 2: "r_hip", 3: "l_hip", 4: "l_knee", 5: "l_ankle", 6: "pelvis",
                         7: "thorax", 8: "upper_neck", 9: "head_top", 10: " r_wrist", 11: "r_elbow", 12: "r_shoulder",
                         13: "l_shoulder", 14: "l_elbow", 15: "l_wrist"}

     mpii_keypoints = {"r_ankle": 0, "r_knee": 1, "r_hip": 2, "l_hip": 3, "l_knee": 4, "l_ankle": 5, "pelvis": 6,
                      "thorax": 7, "upper_neck": 8, "head_top": 9, " r_wrist": 10, "r_elbow": 11, "r_shoulder": 12,
                      "l_shoulder": 13, "l_elbow": 14, "l_wrist": 15}
    """

    def __init__(self, config, model_path=None, threshhold=0.3, device="cuda:0"):
        """
        :param config:
        :param threshhold:
        :param device:
        """
        super(PoseEstimation, self).__init__(config, model_path, threshhold, device)
        self.threshhold = threshhold
        self.detector = Detector(detect_type="ultra_person", device=device)

    def start_capture(self, video_path, save_video=None, detect_freq=1):
        """
        start capture video
        :param video_path: *.avi,*.mp4,...
        :param save_video: *.avi
        :param detect_freq:
        :return:
        """
        video_cap = image_processing.get_video_capture(video_path)
        width, height, numFrames, fps = image_processing.get_video_info(video_cap)
        if save_video:
            self.video_writer = image_processing.get_video_writer(save_video, width, height, fps)
        # freq = int(fps / detect_freq)
        count = 0
        while True:
            isSuccess, frame = video_cap.read()
            if not isSuccess:
                break
            if count % detect_freq == 0:
                kp_points, kp_scores, boxes = self.detect_image(frame,
                                                                threshhold=self.threshhold,
                                                                detect_person=False)
                self.show_result(frame, boxes, kp_points, kp_scores, self.skeleton, waitKey=10)
            if save_video:
                self.video_writer.write(frame)
            count += 1
        video_cap.release()

    @debug.run_time_decorator("detect_person")
    def detect_person(self, image):
        bbox_score, labels = self.detector.detect(image, isshow=False)
        boxes = bbox_score[:, 0:4]
        scores = bbox_score[:, 4:5]
        return boxes, scores

    @debug.run_time_decorator("detect_pose")
    def detect_pose(self, image, boxes, threshhold):
        kp_points, kp_scores = self.detect(image, boxes, threshhold=threshhold)
        return kp_points, kp_scores

    @debug.run_time_decorator("detect_image")
    def detect_image(self, frame, threshhold=0.8, detect_person=False):
        '''
        :param frame: bgr image
        :param threshhold:
        :return:
        '''
        if detect_person:
            boxes, scores = self.detect_person(frame)
        else:
            h, w, d = frame.shape
            boxes = [[0, 0, w, h]]
        key_points, kp_scores = self.detect_pose(frame, boxes, threshhold)
        return key_points, kp_scores, boxes

    def detect_image_dir(self, image_dir, detect_person=True, waitKey=0):
        image_list = file_processing.get_files_lists(image_dir)

        for i, image_path in enumerate(image_list):
            bgr_image = cv2.imread(image_path)
            # bgr_image = cv2.cvtColor(bgr_image,cv2.COLOR_BGR2RGB)
            # bgr_image = image_processing.resize_image(bgr_image, resize_height=800)
            kp_points, kp_scores, boxes = self.detect_image(bgr_image, threshhold=self.threshhold,
                                                            detect_person=detect_person)
            print("detect_image_dir", image_path, boxes)
            self.show_result(bgr_image, boxes, kp_points, kp_scores, self.skeleton, waitKey)

    def detect_local_image(self, image_path):
        base_name = os.path.basename(image_path)
        try:
            bgr_image = cv2.imread(image_path)
            kp_points, kp_scores, boxes = self.detect_image(bgr_image, threshhold=self.threshhold, detect_person=True)

            if len(boxes) == 1:
                print("detect_image2", image_path, boxes)
                shutil.copyfile(image_path, os.path.join(bodyRoot, base_name))
                return 1, [base_name, [boxes[0][1], boxes[0][2], boxes[0][3], boxes[0][0]]]
            else:
                print("detect_image2", image_path, boxes, "不包含")
                return 0, [base_name, []]
        except Exception as e:
            print(image_path, e, "不包含")
        return 0, [base_name, []]

    def show_result(self, image, boxes, kp_points, kp_scores, skeleton=None, waitKey=0):
        if not skeleton:
            skeleton = self.skeleton
        image = self.draw_keypoints(image, boxes, kp_points, kp_scores, skeleton)
        cv2.imwrite('test.png', image)
        cv2.imshow('test', image)
        cv2.waitKey(waitKey)

    def draw_keypoints(self,
                       image,
                       boxes,
                       kp_points,
                       kp_scores,
                       skeleton, box_color=(255, 0, 0),
                       circle_color=(0, 255, 0), line_color=(0, 0, 255)):
        """
        :param image:
        :param keypoints:
        :param kp_scores:
        :param bboxes:
        :param scores
        :return:
        """
        vis_image = image.copy()
        vis_image = image_processing.draw_key_point_in_image(vis_image, kp_points,
                                                             circle_color=circle_color,
                                                             line_color=line_color,
                                                             pointline=skeleton,
                                                             thickness=10)
        vis_image = image_processing.draw_image_boxes(vis_image, boxes, color=box_color)
        return vis_image


class ImageOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--path', type=str, required=True, default='results', help='image file path')
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        return self.opt


if __name__ == '__main__':
    # 自定义MPII上半身6个关键点
    # hp = PoseEstimation(config=val_config.body_mpii_192_256, device="cuda:0")
    # COCO共17个关键点
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    opt = ImageOptions().parse()
    print('start detector image', opt)
    tt = time.time()
    path = opt.path
    flist = os.listdir(path)

    bodyRoot = path + "DetectBody"

    if not os.path.exists(bodyRoot):
        os.makedirs(bodyRoot)

    hp = PoseEstimation(config=val_config.person_coco_192_256, device="cuda:0")
    # 自定义COCO上半身11个关键点
    # hp = PoseEstimation(config=val_config.body_coco_192_256, device="cuda:0")

    # hp.start_capture(video_path=video_path, save_video=save_video)
    # hp.start_capture(video_path)

    executor = ThreadPoolExecutor(max_workers=8)
    # executor = ProcessPoolExecutor(max_workers=1)
    columns1 = ["image", "face[top, right, bottom, left]"]
    resultBodyData = []
    errorImageSize = 0
    for index1 in range(0, len(flist)):
        image = path + os.sep + flist[index1]

        futures = []
        task = executor.submit(hp.detect_local_image, image)
        futures.append(task)

        for future in as_completed(futures):
            is_face, data = future.result()
            if is_face == 1:
                resultBodyData.append(data)
            else:
                errorImageSize += 1

            futures.remove(future)

            if len(resultBodyData) + errorImageSize == len(flist):
                resultBody = pandas.ExcelWriter(bodyRoot + os.sep + "tags.xlsx")
                pandas.DataFrame(resultBodyData, columns=columns1).to_excel(resultBody, index=False)
                resultBody.save()

    print('time2 end:', time.time() - tt)
