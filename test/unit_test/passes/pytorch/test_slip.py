# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import pytest
import torch
import json

from olive.model import PyTorchModelHandler
from olive.hardware.accelerator import AcceleratorSpec
from olive.systems.system_config import SystemConfig

class DummyEvaluator:
    def evaluate(self, model, metric):
        return DummyMetricResult()

class DummyMetricResult:
    def get_value(self, metric_name, sub_metric_name):
        return 0.9

class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Conv2d(3, 6, 3)
        self.relu1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer2 = torch.nn.Conv2d(6, 12, 3)
        self.relu2 = torch.nn.ReLU()
        self.pool2 = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(12, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = x.squeeze(1)
        return x

@pytest.fixture
def model():
    return SimpleModel()

@pytest.fixture
def model_handler(model):
    return PyTorchModelHandler(model_loader=lambda _: model)

@pytest.fixture
def partition_config_file(tmp_path):
    config = {
        "enum": {
            "config1": {
                "layer_filter_channel": {
                    0: {"name": "layer1.weight.0.0"}
                }
            }
        }
    }
    config_path = tmp_path / "partition_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f)
    return config_path

def test_slip_optimizer_run_for_config(model_handler, partition_config_file, tmp_path):
    config = {
        "model_partitions_configurations_file": str(partition_config_file),
        "data_config": {  # updated dummy data config based on DummyDataContainer
            "name": "dummy_data_config",
            "type": "DummyDataContainer",
            "load_dataset_config": {
                "params": {
                    "input_names": ["input"],
                    "input_shapes": [[1, 3, 224, 224]],
                    "input_types": ["float32"]
                }
            }
        },
        "exposed_model_evaluator_system": SystemConfig.parse_obj({"type": "LocalSystem", "config": {}}),
        # "exposed_model_evaluator_system": SystemConfig.parse_obj(
        #     {
        #         "type": "AzureML", 
        #         "config": {
        #             "azureml_client_config": {
        #                 "subscription_id": "<subscription_id>",
        #                 "resource_group": "<resource_group>",
        #                 "workspace_name": "<workspace_name>",
        #             },
        #             "aml_compute": "<aml_compute_cluster_name>",
        #             "aml_docker_config": {
        #                 "base_image": "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:20240709.v1",
        #                 "conda_file_path": "test/integ_test/aml_model_test/conda.yaml"
        #             },
        #         }
        #     }),
    }
    accelerator_spec = AcceleratorSpec(accelerator_type="cpu")
    from olive.passes.pytorch.slip import SLIPOptimizer
    optimizer = SLIPOptimizer(accelerator_spec=accelerator_spec, config=config)
    config['exposed_model_evaluator'] = optimizer._default_config(accelerator_spec)['exposed_model_evaluator'].default_value

    out_model = optimizer._run_for_config(model_handler, config, str(tmp_path))
    best_partition_config = out_model.model_attributes['slip']
    assert best_partition_config == {
        "best_partition": {
            "layer_filter_channel": {
                '0': {"name": "layer1.weight.0.0"}
            }
        }
    }
