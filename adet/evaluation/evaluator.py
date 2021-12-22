# Copyright (c) Facebook, Inc. and its affiliates.
import datetime
import logging
import time
from collections import OrderedDict, abc
from contextlib import ExitStack, contextmanager
from typing import List, Union
import torch
from torch import nn

from detectron2.utils.comm import get_world_size, is_main_process
from detectron2.utils.logger import log_every_n_seconds
from tqdm import tqdm
from adet.utils.post_process import detector_postprocess

NUM_OCC_IMG, NUM_OCC_INST = 0, 0
def draw_input_output(idx, inputs, outputs):
    global NUM_OCC_IMG, NUM_OCC_INST
    ###########################
    # draw inputs and outputs #
    ###########################
    import cv2
    import numpy as np
    
    for input, output in zip(inputs, outputs):
        instances = detector_postprocess(output["instances"], input["height"], input["width"])                    
        occ_pred = instances.pred_occlusions
        save_dir = "tmp"
        img = input['image'].cpu().numpy()
        rgb = img[:3].transpose(1, 2, 0)
        depth = img[3]
        cv2.imwrite("{}/{}_input_rgb.png".format(save_dir, idx), np.uint8(rgb))
        cv2.imwrite("{}/{}_input_depth.png".format(save_dir, idx), np.uint8(depth))
        # pred - occlusion 
        occ_sum_all, occ_sum_pred = None, None
        for i, occ in enumerate(instances.pred_occluded_masks):
            occ = occ.detach().cpu().numpy()
            # cv2.imwrite("tmp/{}_pred_occ_{}.png".format(idx, i), occ*255)
            if occ_sum_all is None:
                occ_sum_all = np.zeros_like(occ, dtype=np.uint8)
            if occ_sum_pred is None:
                occ_sum_pred = np.zeros_like(occ, dtype=np.uint8)
            occ_sum_all = occ_sum_all + np.uint8(occ)
            if occ_pred[i] == 1:
                occ_sum_pred = occ_sum_pred + np.uint8(occ)             
        cv2.imwrite("{}/{}_pred_occ_sum_all.png".format(save_dir, idx), occ_sum_all*50)
        cv2.imwrite("{}/{}_pred_occ_sum_pred_{}.png".format(save_dir, idx, occ_pred), occ_sum_pred*50)
        # pred - visible mask  
        occ_sum = None
        for i, occ in enumerate(instances.pred_visible_masks):
            occ = occ.detach().cpu().numpy()
            # cv2.imwrite("{}/{}_pred_vis_{}.png".format(save_dir, idx, i), occ*255)
            if occ_sum is None:
                occ_sum = np.zeros_like(occ, dtype=np.uint8)
            occ_sum = occ_sum+np.uint8(occ)
        cv2.imwrite("{}/{}_pred_vis_sum.png".format(save_dir, idx), occ_sum*50)
        # pred - amodal mask  
        occ_sum = None
        for i, occ in enumerate(instances.pred_masks):
            occ = occ.detach().cpu().numpy()
            # cv2.imwrite("{}/{}_pred_mask_{}.png".format(save_dir, idx, i), occ*255)
            if occ_sum is None:
                occ_sum = np.zeros_like(occ, dtype=np.uint8)
            occ_sum = occ_sum+np.uint8(occ)
        cv2.imwrite("{}/{}_pred_mask_sum.png".format(save_dir, idx), occ_sum*50)
        # GT - occlusion 
        occ_sum = np.zeros_like(depth, dtype=np.uint8)
        for i, occ in enumerate(input['instances'].gt_occluded_masks):
            occ = occ.detach().cpu().numpy()
            if np.sum(occ) > 0: NUM_OCC_INST += 1
            # cv2.imwrite("{}/{}_gt_occ_{}.png".format(save_dir, idx, i), occ*255)
            occ_sum = occ_sum+np.uint8(occ)
        if np.sum(occ_sum) > 0: NUM_OCC_IMG += 1
        cv2.imwrite("{}/{}_gt_occ_sum.png".format(save_dir, idx), occ_sum*50)
        # GT - visible mask 
        occ_sum = np.zeros_like(depth, dtype=np.uint8)
        for i, occ in enumerate(input['instances'].gt_visible_masks):
            occ = occ.detach().cpu().numpy()
            # cv2.imwrite("{}/{}_gt_vis_{}.png".format(save_dir, idx, i), occ*255)
            occ_sum = occ_sum+np.uint8(occ)
        cv2.imwrite("{}/{}_gt_vis_sum.png".format(save_dir, idx), occ_sum*50)
        # GT - amodal mask 
        occ_sum = np.zeros_like(depth, dtype=np.uint8)
        for i, occ in enumerate(input['instances'].gt_masks):
            occ = occ.detach().cpu().numpy()
            # cv2.imwrite("{}/{}_gt_mask_{}.png".format(save_dir, idx, i), occ*255)
            occ_sum = occ_sum+np.uint8(occ)
        cv2.imwrite("{}/{}_gt_mask_sum.png".format(save_dir, idx), occ_sum*50)    

class DatasetEvaluator:
    """
    Base class for a dataset evaluator.
    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.
    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def process(self, inputs, outputs):
        """
        Process the pair of inputs and outputs.
        If they contain batches, the pairs can be consumed one-by-one using `zip`:
        .. code-block:: python
            for input_, output in zip(inputs, outputs):
                # do evaluation on single input/output pair
                ...
        Args:
            inputs (list): the inputs that's used to call the model.
            outputs (list): the return value of `model(inputs)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.
        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:
                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass


class DatasetEvaluators(DatasetEvaluator):
    """
    Wrapper class to combine multiple :class:`DatasetEvaluator` instances.
    This class dispatches every evaluation call to
    all of its :class:`DatasetEvaluator`.
    """

    def __init__(self, evaluators):
        """
        Args:
            evaluators (list): the evaluators to combine.
        """
        super().__init__()
        self._evaluators = evaluators

    def reset(self):
        for evaluator in self._evaluators:
            evaluator.reset()

    def process(self, inputs, outputs):
        for evaluator in self._evaluators:
            evaluator.process(inputs, outputs)

    def evaluate(self):
        results = OrderedDict()
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            if is_main_process() and result is not None:
                for k, v in result.items():
                    assert (
                        k not in results
                    ), "Different evaluators produce results with the same key {}".format(k)
                    results[k] = v
        return results


def inference_on_dataset(
    model, data_loader, evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None]
):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.
    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.
            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
            but don't want to do any evaluation.
    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_compute_time = 0
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0

            start_compute_time = time.perf_counter()
            outputs = model(inputs)

            # draw input and ouputs for debugging
            draw_masks = False
            if draw_masks: draw_input_output(idx, inputs, outputs)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            evaluator.process(inputs, outputs)

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_iter = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_iter > 5:
                total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Inference done {}/{}. {:.4f} s / iter. ETA={}".format(
                        idx + 1, total, seconds_per_iter, str(eta)
                    ),
                    n=5,
                )

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )
    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.
    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)