from __future__ import annotations

import time

import av
import cv2
from real_time_cv.app import run, DEFAULT_ICE_CONFIG
from real_time_cv.processing import ProcessorPlugin
import streamlit as st

from detectron2.utils.logger import setup_logger
setup_logger()

import cv2
from detectron_utils import model_configs, setup_detectron



if 'predictors' in st.session_state:
    predictors = st.session_state['predictors']
    visualiser_factories = st.session_state['visualiser_factories']
else:
    st.session_state['predictors'] = {}
    st.session_state['visualiser_factories'] = {}
    for conf_name, cfg_file, th in model_configs:
        predictor, visualiser_factory = setup_detectron(cfg_file, th)
        st.session_state['predictors'][conf_name] = predictor
        st.session_state['visualiser_factories'][conf_name] = visualiser_factory


def _run_inference(frame: av.VideoFrame, predictor, visualiser_factory) -> av.VideoFrame:
    image = frame.to_ndarray(format='bgr24')
    visualiser = visualiser_factory(image)
    outputs = predictor(image)
    out = visualiser.draw_instance_predictions(outputs["instances"].to("cpu"))
    return av.VideoFrame.from_ndarray(out.get_image(), format='bgr24')


def instance_segmentation(frame: av.VideoFrame):
    predictor = predictors["instance-seg"]
    visualiser_factory = visualiser_factories["instance-seg"]
    return _run_inference(frame, predictor, visualiser_factory)


def detection(frame: av.VideoFrame):
    predictor = predictors["detection"]
    visualiser_factory = visualiser_factories["detection"]
    return _run_inference(frame, predictor, visualiser_factory)


def panoptic_segmentation(frame: av.VideoFrame):
    predictor = predictors["panoptic-seg"]
    visualiser_factory = visualiser_factories["panoptic-seg"]
    return _run_inference(frame, predictor, visualiser_factory)


def keypoints_detection(frame: av.VideoFrame):
    predictor = predictors["keypoints"]
    visualiser_factory = visualiser_factories["keypoints"]
    return _run_inference(frame, predictor, visualiser_factory)


def identity(frame: av.VideoFrame) -> av.VideoFrame:
    return frame


def convert2gray(frame: av.VideoFrame) -> av.VideoFrame:
    image = frame.to_ndarray(format='bgr24')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    time.sleep(1) # simulate heavy processing
    return av.VideoFrame.from_ndarray(gray, format='gray')


if __name__ == '__main__':
    dummy_plugin = ProcessorPlugin()
    dummy_plugin.register_ref_processor(identity)
    dummy_plugin.register_processor('convert to gray', convert2gray)
    dummy_plugin.register_processor('instance segmentation', instance_segmentation)
    dummy_plugin.register_processor('panoptic segmentation', panoptic_segmentation)
    dummy_plugin.register_processor('keypoints detection', keypoints_detection)
    dummy_plugin.register_processor('detection', detection)
    run(
        processor_plugin=dummy_plugin,
        rtc_configuration=DEFAULT_ICE_CONFIG, # you can set your own rtc config (check https://github.com/whitphx/streamlit-webrtc/tree/main)
        layout='horizontal', # this controls the layout of the streams (ref/processor) ['vertical', 'horizontal']
        )