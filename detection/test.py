from glob import glob

import torch

from detectron2.utils.logger import setup_logger
setup_logger()
import pycocotools.mask as mask_util

# import some common libraries
import numpy as np
import os, cv2
import simplejson as json

# import some common detectron2 utilities
import detectron2.data.transforms as T
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog, DatasetMapper
from detectron2.utils.visualizer import ColorMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

from .load_data import load_data
from parser import config_parser


def save_json(outputs, scores, path):
    outputs = outputs['instances'].to('cpu')
    output = outputs.get_fields()
    json_output = dict()
    json_output['scores'] = scores.tolist()#output['scores'].data.numpy().tolist()
    json_output['pred_boxes'] = output['pred_boxes'].tensor.data.numpy().tolist()
    json_output['pred_classes'] = output['pred_classes'].data.numpy().tolist()
    json_output['pred_masks'] = output['pred_masks'].int().data.numpy().astype(np.uint8)
    json_output['pred_masks'] = [mask_util.encode(
        np.asfortranarray(mask)) for mask in json_output['pred_masks']]
    with open(path, 'w') as f:
        json.dump(json_output, f, ensure_ascii=False)

def pred(aug, model, path):
    org_im = cv2.imread(path)
    height, width = org_im.shape[:2]
    with torch.no_grad():
        im = aug.get_transform(org_im).apply_image(org_im)
        im = torch.as_tensor(im.astype("float32").transpose(2, 0, 1))
        inputs = [{"image": im, "height": height, "width": width}]
        images = model.preprocess_image(inputs)
        features = model.backbone(images.tensor)  # set of cnn features
        proposals, _ = model.proposal_generator(images, features, None)  # RPN
        features_ = [features[f] for f in model.roi_heads.box_in_features]
        box_features = model.roi_heads.box_pooler(features_, [x.proposal_boxes for x in proposals])
        box_features = model.roi_heads.box_head(box_features)  # features of all 1k candidates
        predictions = model.roi_heads.box_predictor(box_features)
        pred_instances, pred_inds = model.roi_heads.box_predictor.inference(predictions, proposals)
        pred_instances = model.roi_heads.forward_with_given_boxes(features, pred_instances)

        # output boxes, masks, scores, etc
        pred_instances = model._postprocess(pred_instances, inputs, images.image_sizes)  # scale box to orig size
        # features of the proposed boxes
        scores = torch.softmax(predictions[0][pred_inds], 1).data.cpu().numpy()
        outputs = pred_instances[0]

        ######
        path = path.replace('images','detection_preds')
        name = path.split('/')[-1]
        path = path.replace('/'+name,'')
        os.makedirs(path, exist_ok=True)
        save_json(outputs, scores, f"{path}/{name.replace('png','json')}")
        if args.det_save_images:
            v = Visualizer(org_im[:, :, ::-1], scale=1.0)
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            cv2.imwrite(f'{path}/{name}', out.get_image()[:, :, ::-1])

if __name__ == "__main__":
    parser = config_parser()
    args = parser.parse_args()
    for d in ["train", "test"]:
        DatasetCatalog.register("mydata_" + d, lambda d=d: load_data(d, args))
        MetadataCatalog.get("mydata_" + d).set(thing_classes=["Door","Glass_door","Frame","Window","Kitchen_counter","closet"])
    mydata_metadata = MetadataCatalog.get("mydata_train")

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(args.det_config))
    cfg.DATASETS.TRAIN = ("mydata_train")
    cfg.DATASETS.TEST = ("mydata_test")
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = f"detection/{args.det_model_weight}"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6


    if args.det_is_eval:
        predictor = DefaultPredictor(cfg)
        evaluator = COCOEvaluator("mydata_test", ("bbox", "segm"), False, output_dir="logs/detection")
        mapper = DatasetMapper(cfg, is_train=False)
        loader = build_detection_test_loader(cfg, "mydata_test", mapper=mapper)
        print(inference_on_dataset(predictor.model, loader, evaluator))
    else:
        aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
        mycfg = cfg.clone()
        predictor = build_model(mycfg)
        DetectionCheckpointer(predictor).load(cfg.MODEL.WEIGHTS)
        predictor.eval()
        dataset_dicts = load_data('all', args)
        for d in dataset_dicts:
            pred(aug, predictor, d['file_name'])
