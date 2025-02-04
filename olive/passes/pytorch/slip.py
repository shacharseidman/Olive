# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from copy import deepcopy
import logging
import json
from pathlib import Path
from typing import Any, Dict, Union

import torch
import re

from olive.common.config_utils import ParamCategory
from olive.evaluator.metric import AccuracySubType, Metric, SubMetric
from olive.evaluator.olive_evaluator import OliveEvaluator, OliveEvaluatorConfig
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import CompositeModelHandler
from olive.model.handler.hf import HfModelHandler
from olive.model.handler.pytorch import PyTorchModelHandler
from olive.model.utils.path_utils import normalize_path_suffix
from olive.passes import Pass
from olive.passes.pass_config import PassConfigParam
from olive.passes.pytorch.common import inherit_pytorch_from_pytorch
from olive.data.config import DataConfig  # newly added import

# import sys, os
# sys.path.append('../../../llm_ds')

logger = logging.getLogger(__name__)

class SLIPOptimizer(Pass):

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return {
            "model_partitions_configurations_file": PassConfigParam(
                type_=Union[str, Path],
                category=ParamCategory.PATH,
                required=True,
                description=(
                    "Path to the model partitions configurations file. The file should be a json file with a list of partition definitions."
                ),
            ),
            "exposed_model_evaluator": PassConfigParam(
                type_=OliveEvaluatorConfig,
                required=True,
                description="""
                    The evaluator configuration for the exposed model.
                    The evaluator configuration should be an instance of OliveEvaluatorConfig class.
                """,
                default_value=OliveEvaluatorConfig(
                    metrics=[
                        Metric(
                            name="accuracy",
                            type="accuracy",  # Add the type field here
                            sub_types=[
                                {
                                    "name": AccuracySubType.ACCURACY_SCORE,
                                    "higher_is_better": False,
                                }
                            ],
                        ),
                    ],
                ),
            ),
            "data_config": PassConfigParam(  # new parameter added
                type_=Union[DataConfig, dict],
                required=False,
                default_value=None,
                description="Data configuration for evaluating the exposed model."
            ),
            # "exposed_model_effectiveness_system": PassConfigParam(
            #     type_=SystemConfig,
            #     required=True,
            #     description="""
            #         The system where the exposed model will be evaluated.
            #         The system should be an instance of SystemConfig class.
            #     """,
            # ),
            # "model_partition_config_id": PassConfigParam(
            #     type_=int, 
            #     required=True, 
            #     description="Configuration of how to partition a model using SLIP protocol"
            # ),
        }
    
    def _SRD_slicing_iterator(self, model, slice_enum_full_config):
        for slice_idx in slice_enum_full_config["layer_filter_channel"]:
            logger.debug('performing SRD slicing - idx: ', str(slice_idx))

            # parse layer, filter+channel
            full_layer_name, filter_channel = slice_enum_full_config["layer_filter_channel"][slice_idx]["name"].split('.weight.')
            filter_idx, channel_idx = re.match(r'(\d+)\.(\d+)', filter_channel).groups()
            filter_idx = int(filter_idx)
            channel_idx = int(channel_idx)
            full_layer_name = full_layer_name + '.weight'
            
            # update slice
            W = dict(model.named_parameters())[full_layer_name]
            W.data[filter_idx, channel_idx, :, :] = 0    

        return model
    
    def _transform_into_exposed_model(self, model: torch.nn.Module, partition_config: Dict) -> PyTorchModelHandler:
        exposed_model = deepcopy(model)
        self._SRD_slicing_iterator(exposed_model, partition_config)
        return exposed_model
    
    def _transform_into_olive_model(self, model_config_id: str, pytorch_model: torch.nn.Module, output_model_path: str) -> PyTorchModelHandler:
        output_model_path = normalize_path_suffix(output_model_path, f"exposed_model_{model_config_id}.pt")
        torch.save(pytorch_model, output_model_path)
        return PyTorchModelHandler(model_path=output_model_path)
    
    def _find_best_partition_config(self, model: torch.nn.Module, model_partition_config: Dict, 
                                    evaluator: OliveEvaluator, exposed_model_metric: Metric,
                                    output_model_path: str) -> Dict:
        assert 'enum' in model_partition_config, "model_partition_config should have an 'enum' key"
        best_partition_config_key, best_metric_value = None, None
        for partition_key, partition_config in model_partition_config['enum'].items():
            logger.info(f"Running model partition config {partition_key}")
            exposed_pymodel = self._transform_into_exposed_model(model, partition_config)
            olive_model = self._transform_into_olive_model(partition_key, exposed_pymodel, output_model_path)
            logger.info(f"About to evaluate model partition config {partition_key}")
            metric_results = evaluator.evaluate(olive_model, [exposed_model_metric])
            logger.info(f"Model partition config {partition_key} metric results: {metric_results}")
            metric_value = metric_results.get_value(exposed_model_metric.name, exposed_model_metric.sub_types[0].name)
            if exposed_model_metric.sub_types[0].higher_is_better:
                if best_metric_value is None or metric_value > best_metric_value:
                    logger.info(f"Found new best metric value {metric_value} for model partition config {partition_key}")
                    best_metric_value = metric_value
                    best_partition_config_key = partition_key
            else:
                if best_metric_value is None or metric_value < best_metric_value:
                    logger.info(f"Found new best metric value {metric_value} for model partition config {partition_key}")
                    best_metric_value = metric_value
                    best_partition_config_key = partition_key

        logger.info(f"Best model partition config is {best_partition_config_key} with metric value {best_metric_value}")
        return model_partition_config['enum'][best_partition_config_key]

    def _run_for_config(self, model: Union[HfModelHandler, PyTorchModelHandler], config: Dict[str, Any], output_model_path: str) -> CompositeModelHandler:
        if isinstance(model, PyTorchModelHandler):
            pytorch_model = model.load_model(cache_model=True)
            pytorch_model.eval()

        with open(config["model_partitions_configurations_file"], "r") as f:
            model_partition_config = json.load(f)

        # exposed_model_system = config["exposed_model_system"]
        # evaluator_config = OliveEvaluatorConfig(**config["exposed_model_evaluator"])
        evaluator_config: OliveEvaluatorConfig = config["exposed_model_evaluator"]
        # Inject the data_config into the evaluator metric, if provided.
        exposed_model_metric = evaluator_config.metrics[0]
        # Convert data_config dict to a DataConfig instance if necessary.
        data_config = self.config.get("data_config")
        if data_config and not isinstance(data_config, DataConfig):
            data_config = DataConfig(**data_config)
        exposed_model_metric.data_config = data_config

        evaluator: OliveEvaluator = evaluator_config.create_evaluator(model)

        best_partition_config = self._find_best_partition_config(pytorch_model, 
                                                                 model_partition_config, 
                                                                 evaluator, 
                                                                 exposed_model_metric,
                                                                 output_model_path)
        return best_partition_config

        # hybrid_model_package = self._partition_model(pytorch_model, best_partition_config)
        # return hybrid_model_package