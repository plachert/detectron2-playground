from __future__ import annotations

import time

import av
import cv2
from real_time_cv.app import run, DEFAULT_ICE_CONFIG
from real_time_cv.processing import ProcessorPlugin
import streamlit as st

from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2
from functools import partial


if 'predictor' in st.session_state:
    predictor = st.session_state['predictor']
    visualiser_factory = st.session_state['visualiser_factory']
else:
    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    visualiser_factory = partial(Visualizer, metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    st.session_state['predictor'] = predictor
    st.session_state['visualiser_factory'] = visualiser_factory



def identity(frame: av.VideoFrame) -> av.VideoFrame:
    return frame


def instance_segmentation_rcnn_r_50_fpn_3x(frame: av.VideoFrame):
    image = frame.to_ndarray(format='bgr24')
    visualiser = visualiser_factory(image)
    outputs = predictor(image)
    out = visualiser.draw_instance_predictions(outputs["instances"].to("cpu"))
    return av.VideoFrame.from_ndarray(out.get_image(), format='bgr24')


def convert2gray(frame: av.VideoFrame) -> av.VideoFrame:
    image = frame.to_ndarray(format='bgr24')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    time.sleep(1) # simulate heavy processing
    return av.VideoFrame.from_ndarray(gray, format='gray')


if __name__ == '__main__':
    dummy_plugin = ProcessorPlugin()
    dummy_plugin.register_ref_processor(identity) # There can be only one ref_processor
    dummy_plugin.register_processor('convert to gray', convert2gray) # There can be more than one
    dummy_plugin.register_processor('instance segmentation', instance_segmentation_rcnn_r_50_fpn_3x) # There can be more than one
    run(
        processor_plugin=dummy_plugin,
        rtc_configuration=DEFAULT_ICE_CONFIG, # you can set your own rtc config (check https://github.com/whitphx/streamlit-webrtc/tree/main)
        layout='vertical', # this controls the layout of the streams (ref/processor) ['vertical', 'horizontal']
        )