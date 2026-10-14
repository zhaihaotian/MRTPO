# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A unified tracking interface that supports logging data to different backend
"""
import dataclasses
import errno
import json
import time
from enum import Enum
from functools import partial
from pathlib import Path
from typing import List, Union, Dict, Any


class Tracking(object):
    supported_backend = ['wandb', 'mlflow', 'console', 'jsonl']

    def __init__(self, project_name, experiment_name, default_backend: Union[str, List[str]] = 'console', config=None):
        if isinstance(default_backend, str):
            default_backend = [default_backend]
        for backend in default_backend:
            if backend == 'tracking':
                import warnings
                warnings.warn("`tracking` logger is deprecated. use `wandb` instead.", DeprecationWarning)
            else:
                assert backend in self.supported_backend, f'{backend} is not supported'

        self.logger = {}

        if 'jsonl' in default_backend:
            output_dir = Path(config['trainer']['default_local_dir'])
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / 'config.json').write_text(json.dumps(config, indent=2))
            self.logger['jsonl'] = _JsonlLogger(output_dir / 'metrics.jsonl')

        if 'tracking' in default_backend or 'wandb' in default_backend:
            import wandb
            import os
            # Initialize W&B
            WANDB_API_KEY = os.environ.get("WANDB_API_KEY", None)
            if WANDB_API_KEY:
                wandb.login(key=WANDB_API_KEY)
            wandb.init(project=project_name, name=experiment_name, config=config)
            
            self.logger['wandb'] = wandb

        if 'mlflow' in default_backend:
            import mlflow
            mlflow.start_run(run_name=experiment_name)
            mlflow.log_params(_compute_mlflow_params_from_objects(config))
            self.logger['mlflow'] = _MlflowLoggingAdapter()

        if 'console' in default_backend:
            from verl.utils.logger.aggregate_logger import LocalLogger
            self.console_logger = LocalLogger(print_to_console=True)
            self.logger['console'] = self.console_logger

    def log(self, data, step, backend=None):
        for default_backend, logger_instance in self.logger.items():
            if backend is None or default_backend in backend:
                logger_instance.log(data=data, step=step)


class _JsonlLogger:
    def __init__(self, path):
        self.path = path

    def log(self, data, step):
        record = (json.dumps({'step': step, **data}) + '\n').encode('utf-8')
        offset = None
        for attempt in range(30):
            try:
                with self.path.open('ab') as output:
                    if offset is None:
                        offset = output.tell()
                    else:
                        # Replace the current record if a failed write reached disk.
                        output.truncate(offset)
                    output.write(record)
                    output.flush()
                return
            except OSError as error:
                if error.errno != errno.EIO or attempt == 29:
                    raise
                time.sleep(2)


class _MlflowLoggingAdapter:

    def log(self, data, step):
        import mlflow
        mlflow.log_metrics(metrics=data, step=step)


def _compute_mlflow_params_from_objects(params) -> Dict[str, Any]:
    if params is None:
        return {}

    return _flatten_dict(_transform_params_to_json_serializable(params, convert_list_to_dict=True), sep='/')


def _transform_params_to_json_serializable(x, convert_list_to_dict: bool):
    _transform = partial(_transform_params_to_json_serializable, convert_list_to_dict=convert_list_to_dict)

    if dataclasses.is_dataclass(x):
        return _transform(dataclasses.asdict(x))
    if isinstance(x, dict):
        return {k: _transform(v) for k, v in x.items()}
    if isinstance(x, list):
        if convert_list_to_dict:
            return {'list_len': len(x)} | {f'{i}': _transform(v) for i, v in enumerate(x)}
        else:
            return [_transform(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, Enum):
        return x.value

    return x


def _flatten_dict(raw: Dict[str, Any], *, sep: str) -> Dict[str, Any]:
    import pandas as pd
    ans = pd.json_normalize(raw, sep=sep).to_dict(orient='records')[0]
    assert isinstance(ans, dict)
    return ans
