from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

from functools import partial


model_configs = [
    ("instance-seg", "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", 0.5),
    ("panoptic-seg", "COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml", 0.5),
    ("detection", "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml", 0.5),
    ("keypoints", "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml", 0.7),
    ]


def setup_detectron(cfg_file, th=0.5):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(cfg_file))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = th
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_file)
    predictor = DefaultPredictor(cfg)
    visualiser_factory = partial(Visualizer, metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    return predictor, visualiser_factory