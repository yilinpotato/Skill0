# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
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

import torch
import numpy as np
import json
import os
import re
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from transformers import PreTrainedTokenizer
import uuid
from agent_system.multi_turn_rollout.utils import process_image, to_list_of_dict, torch_to_numpy, filter_group_data
from agent_system.environments import EnvironmentManagerBase
from typing import List, Dict
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

PICK_AND_PLACE_CORRECT_FORM = (
    "Pick And Place correct action format: "
    "<think>brief reason</think><action>exact admissible command</action>. "
    "For a pick-and-place task, first find and take the target object, then go to "
    "the target receptacle, put the object there, and issue <action>done</action> "
    "only after the goal is satisfied."
)


class TrajectoryCollector:
    def __init__(self, config, tokenizer: PreTrainedTokenizer, processor=None):
        """
        Initialize the TrajectoryProcessor class.
        
        Parameters:
            config: Configuration object containing data processing settings
            tokenizer (PreTrainedTokenizer): Tokenizer for text encoding and decoding
            processor: Image processor for multimodal inputs
        """
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor

    def _json_safe(self, value):
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.detach().cpu().item()
            return value.detach().cpu().tolist()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def _trajectory_logging_enabled(self):
        return bool(
            self.config.trainer.get("trajectory_log_path", None)
            or self.config.trainer.get("trajectory_log_dir", None)
            or self.config.trainer.get("readable_trajectory_log_path", None)
            or self.config.trainer.get("print_trajectories", False)
        )

    def _trajectory_log_every_n_steps(self) -> int:
        return max(1, int(self.config.trainer.get("trajectory_log_every_n_steps", 1)))

    def _should_log_trajectories_this_call(self, global_step, validate: bool) -> bool:
        """
        Throttle debug trajectory dumps (trajectories.json,
        trajectories_readable.txt, console print, wandb) to every
        ``trainer.trajectory_log_every_n_steps`` training steps, so long runs
        don't pay the read-modify-write JSON cost on every single rollout.
        Validation calls are left untouched since those already run only
        every ``trainer.test_freq`` steps.
        """
        if validate:
            return True
        every_n = self._trajectory_log_every_n_steps()
        if every_n <= 1 or global_step is None:
            return True
        try:
            return int(global_step) % every_n == 0
        except (TypeError, ValueError):
            return True

    def _compact_console_output_enabled(self):
        return bool(self.config.trainer.get("compact_console_output", False))

    def _print_compact_rollout_summary(
        self,
        *,
        gen_batch: DataProto,
        episode_lengths: np.ndarray,
        success: Dict[str, np.ndarray],
        valid_action_ratios: np.ndarray,
    ):
        if not self._compact_console_output_enabled():
            return
        if gen_batch.meta_info.get("validate", False):
            return

        global_step = gen_batch.meta_info.get("global_step", "?")
        total_steps = self.config.trainer.get("total_training_steps", "?")
        success_values = np.asarray(
            success.get("success_rate", np.zeros(len(episode_lengths), dtype=np.float32)),
            dtype=np.float32,
        )
        batch_success_rate = float(success_values.mean()) if len(success_values) > 0 else 0.0

        print(
            f"[Train] progress={global_step}/{total_steps} "
            f"batch_success_rate={batch_success_rate:.3f}",
            flush=True,
        )
        for traj_idx, (steps, traj_success, valid_ratio) in enumerate(
            zip(episode_lengths, success_values, valid_action_ratios)
        ):
            print(
                f"[Train][traj {traj_idx}] "
                f"steps={int(steps)} success={int(traj_success > 0)} "
                f"valid_action_rate={float(valid_ratio):.3f}",
                flush=True,
            )

    def _trajectory_log_path(self):
        path = self.config.trainer.get("trajectory_log_path", None)
        if path:
            return path
        log_dir = self.config.trainer.get("trajectory_log_dir", None)
        if not log_dir:
            return None
        return os.path.join(log_dir, "trajectories.json")

    def _context_log_path(self):
        path = self.config.trainer.get("context_log_path", None)
        if path:
            return path
        log_dir = self.config.trainer.get("trajectory_log_dir", None)
        if not log_dir:
            return None
        return os.path.join(log_dir, "contexts.json")

    def _wandb_trajectory_logging_enabled(self) -> bool:
        logger_cfg = self.config.trainer.get("logger", [])
        if isinstance(logger_cfg, str):
            logger_cfg = [logger_cfg]
        return bool(
            self.config.trainer.get("wandb_log_trajectories", False)
            and "wandb" in logger_cfg
        )

    def _wandb_context_logging_enabled(self) -> bool:
        logger_cfg = self.config.trainer.get("logger", [])
        if isinstance(logger_cfg, str):
            logger_cfg = [logger_cfg]
        return bool(
            self.config.trainer.get("wandb_log_contexts", False)
            and "wandb" in logger_cfg
        )

    def _truncate_text(self, value, max_chars: int) -> str:
        text = "" if value is None else str(value)
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3] + "..."

    def _extract_think_action(self, text: str):
        text = "" if text is None else str(text)
        think_match = re.search(r"<think>\s*(.*?)\s*</think>", text, flags=re.IGNORECASE | re.DOTALL)
        action_match = re.search(r"<action>\s*(.*?)\s*</action>", text, flags=re.IGNORECASE | re.DOTALL)
        think_text = think_match.group(1).strip() if think_match else text.strip()
        action_text = action_match.group(1).strip() if action_match else None
        return think_text, action_text

    def _log_trajectories_to_wandb(self, trajectories, *, validate: bool):
        if not self._wandb_trajectory_logging_enabled():
            return
        if not trajectories:
            return

        try:
            import wandb
        except Exception as exc:
            print(f"[TrajectoryCollector] Skip W&B trajectory logging: {exc}")
            return

        split = "val" if validate else "train"
        max_traj = max(1, int(self.config.trainer.get("wandb_max_trajectories", 4)))
        max_steps = max(1, int(self.config.trainer.get("wandb_max_trajectory_steps", 8)))
        max_chars = max(64, int(self.config.trainer.get("wandb_max_text_chars", 400)))
        max_summary_rows = max(1, int(self.config.trainer.get("wandb_max_summary_rows", 200)))
        max_step_rows = max(1, int(self.config.trainer.get("wandb_max_step_rows", 1000)))

        selected = trajectories[:max_traj]
        summary_columns = [
            "global_step",
            "trajectory_index",
            "task_type",
            "task",
            "episode_length",
            "episode_reward",
            "valid_action_rate",
            "success_rate",
            "last_observation",
            "last_raw_response",
            "last_think",
            "last_action_text",
            "last_model_output",
            "last_env_action",
            "trajectory_json",
        ]
        step_columns = [
            "global_step",
            "trajectory_index",
            "task",
            "step",
            "active",
            "observation",
            "raw_response",
            "think",
            "action_text",
            "model_output",
            "env_action",
            "is_action_valid",
            "reward",
            "done",
            "won",
            "goal_condition_success_rate",
        ]
        if any(any("prompt" in step for step in traj.get("steps", [])) for traj in selected):
            step_columns.append("prompt")

        summary_rows = []
        step_rows = []
        for traj in selected:
            steps = traj.get("steps", [])
            last_step = steps[-1] if steps else {}
            summary_rows.append([
                traj.get("global_step"),
                traj.get("trajectory_index"),
                traj.get("task_type"),
                self._truncate_text(traj.get("task"), max_chars),
                traj.get("episode_length"),
                traj.get("episode_reward"),
                traj.get("valid_action_rate"),
                traj.get("success_rate"),
                self._truncate_text(last_step.get("observation"), max_chars),
                self._truncate_text(last_step.get("raw_response"), max_chars),
                self._truncate_text(last_step.get("think"), max_chars),
                self._truncate_text(last_step.get("action_text"), max_chars),
                self._truncate_text(last_step.get("model_output"), max_chars),
                self._truncate_text(last_step.get("env_action"), max_chars),
                self._truncate_text(json.dumps(traj, ensure_ascii=False), max_chars * 2),
            ])

            for step in steps[:max_steps]:
                row = [
                    traj.get("global_step"),
                    traj.get("trajectory_index"),
                    self._truncate_text(traj.get("task"), max_chars),
                    step.get("step"),
                    step.get("active"),
                    self._truncate_text(step.get("observation"), max_chars),
                    self._truncate_text(step.get("raw_response"), max_chars),
                    self._truncate_text(step.get("think"), max_chars),
                    self._truncate_text(step.get("action_text"), max_chars),
                    self._truncate_text(step.get("model_output"), max_chars),
                    self._truncate_text(step.get("env_action"), max_chars),
                    step.get("is_action_valid"),
                    step.get("reward"),
                    step.get("done"),
                    step.get("won"),
                    step.get("goal_condition_success_rate"),
                ]
                if "prompt" in step_columns:
                    row.append(self._truncate_text(step.get("prompt"), max_chars))
                step_rows.append(row)

        summary_attr = f"_wandb_{split}_trajectory_summary_rows"
        step_attr = f"_wandb_{split}_trajectory_step_rows"
        existing_summary_rows = getattr(self, summary_attr, [])
        existing_step_rows = getattr(self, step_attr, [])
        existing_summary_rows.extend(summary_rows)
        existing_step_rows.extend(step_rows)
        existing_summary_rows = existing_summary_rows[-max_summary_rows:]
        existing_step_rows = existing_step_rows[-max_step_rows:]
        setattr(self, summary_attr, existing_summary_rows)
        setattr(self, step_attr, existing_step_rows)

        summary_table = wandb.Table(columns=summary_columns, data=existing_summary_rows)
        step_table = wandb.Table(columns=step_columns, data=existing_step_rows)
        global_step = selected[0].get("global_step", None)
        wandb.log(
            {
                f"{split}/trajectory_summary": summary_table,
                f"{split}/trajectory_steps": step_table,
            },
            step=global_step,
        )

    def _log_contexts_to_wandb(self, trajectories, *, validate: bool):
        if not self._wandb_context_logging_enabled():
            return
        if not trajectories:
            return

        try:
            import wandb
        except Exception as exc:
            print(f"[TrajectoryCollector] Skip W&B context logging: {exc}")
            return

        split = "val" if validate else "train"
        max_contexts = max(1, int(self.config.trainer.get("wandb_context_samples_per_rollout", 5)))
        max_chars = max(64, int(self.config.trainer.get("wandb_max_text_chars", 400)))

        columns = [
            "global_step",
            "trajectory_index",
            "task",
            "context",
            "first_observation",
            "first_model_output",
            "first_env_action",
        ]
        rows = []
        for traj in trajectories[:max_contexts]:
            steps = traj.get("steps", [])
            first_step = steps[0] if steps else {}
            rows.append([
                traj.get("global_step"),
                traj.get("trajectory_index"),
                self._truncate_text(traj.get("task"), max_chars),
                self._truncate_text(traj.get("prompt_text"), max_chars * 3),
                self._truncate_text(first_step.get("observation"), max_chars),
                self._truncate_text(first_step.get("model_output"), max_chars),
                self._truncate_text(first_step.get("env_action"), max_chars),
            ])

        attr = f"_wandb_{split}_context_rows"
        existing_rows = getattr(self, attr, [])
        existing_rows.extend(rows)
        existing_rows = existing_rows[-max_contexts:]
        setattr(self, attr, existing_rows)

        table = wandb.Table(columns=columns, data=existing_rows)
        global_step = trajectories[0].get("global_step", None)
        wandb.log({f"{split}/context_samples": table}, step=global_step)

    def _dump_contexts(self, contexts):
        if not contexts:
            return

        log_path = self._context_log_path()
        if not log_path:
            return

        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        existing = []
        if os.path.exists(log_path) and os.path.getsize(log_path) > 0:
            with open(log_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if not isinstance(existing, list):
                raise ValueError(f"Context log must contain a JSON list: {log_path}")

        flattened = []
        for c in contexts:
            for step_ctx in c.get("contexts", []):
                flattened.append(
                    {
                        "global_step": c.get("global_step"),
                        "trajectory_index": c.get("trajectory_index"),
                        "task": c.get("task"),
                        "task_type": c.get("task_type"),
                        "step": step_ctx.get("step"),
                        "context": step_ctx.get("prompt_text"),
                    }
                )
        existing.extend(flattened)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
            f.write("\n")

    def _readable_trajectory_log_path(self):
        path = self.config.trainer.get("readable_trajectory_log_path", None)
        if path:
            return path
        log_dir = self.config.trainer.get("trajectory_log_dir", None)
        if not log_dir:
            return None
        return os.path.join(log_dir, "trajectories_readable.txt")

    @staticmethod
    def _fmt_num(value, digits: int = 2) -> str:
        if value is None:
            return "N/A"
        try:
            return f"{float(value):.{digits}f}"
        except (TypeError, ValueError):
            return str(value)

    def _format_trajectory_readable(self, trajectory, index: int) -> str:
        """
        Render one trajectory as a step-by-step, human-readable block for
        debugging (observation / prompt / model reasoning / raw output /
        action / env result per step), instead of a raw JSON dump.
        """
        steps = trajectory.get("steps", [])
        last_step = steps[-1] if steps else {}
        won = bool(last_step.get("won")) if last_step.get("won") is not None else bool(
            trajectory.get("success_rate", 0)
        )
        outcome = "SUCCESS" if won else "FAILURE"
        task_score = last_step.get("task_score")
        num_steps = trajectory.get("episode_length", len(steps))

        lines = [
            "=" * 80,
            f"Episode {index + 1} | outcome: {outcome} (won={won}, "
            f"task_score={self._fmt_num(task_score)}) | steps={num_steps}",
            f"Task: {trajectory.get('task')}",
            "=" * 80,
        ]

        for step in steps:
            lines.append("")
            lines.append(f"----- Step {step.get('step')} -----")
            lines.append("[OBSERVATION]")
            lines.append(str(step.get("observation")))
            if "prompt" in step:
                lines.append("")
                lines.append("[PROMPT SENT TO MODEL]")
                lines.append(str(step.get("prompt")))
            lines.append("")
            lines.append("[MODEL REASONING] (<think>...</think> content)")
            lines.append(str(step.get("think")))
            lines.append("")
            lines.append("[MODEL RAW OUTPUT]")
            lines.append(str(step.get("raw_response")))
            lines.append("")
            lines.append(
                f"[ACTION TAKEN] {step.get('action_text')}   (valid: {step.get('is_action_valid')})"
            )
            lines.append("")
            lines.append(
                f"[ENV RESULT] reward={self._fmt_num(step.get('reward'))}  "
                f"task_score={self._fmt_num(step.get('task_score'))}  "
                f"done={step.get('done')}"
            )

        lines.append("")
        return "\n".join(lines)

    def _dump_trajectories(self, trajectories, *, validate: bool = False):
        if not trajectories:
            return

        if self.config.trainer.get("print_trajectories", False):
            for i, trajectory in enumerate(trajectories):
                print(self._format_trajectory_readable(trajectory, i), flush=True)
                if "pick_and_place_correct_form" in trajectory:
                    print(trajectory["pick_and_place_correct_form"])

        log_path = self._trajectory_log_path()
        if log_path:
            log_dir = os.path.dirname(log_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            existing = []
            if os.path.exists(log_path) and os.path.getsize(log_path) > 0:
                with open(log_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                if not isinstance(existing, list):
                    raise ValueError(f"Trajectory log must contain a JSON list: {log_path}")
            existing.extend(trajectories)
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)
                f.write("\n")

        readable_log_path = self._readable_trajectory_log_path()
        if readable_log_path:
            readable_dir = os.path.dirname(readable_log_path)
            if readable_dir:
                os.makedirs(readable_dir, exist_ok=True)
            with open(readable_log_path, "a", encoding="utf-8") as f:
                for i, trajectory in enumerate(trajectories):
                    f.write(self._format_trajectory_readable(trajectory, i))
                    f.write("\n")

        self._log_trajectories_to_wandb(trajectories, validate=validate)
        self._log_contexts_to_wandb(trajectories, validate=validate)

    def preprocess_single_sample(
        self,
        item: int,
        gen_batch: DataProto,
        obs: Dict,
    ):
        """
        Process a single observation sample, organizing environment observations (text and/or images) 
        into a format processable by the model.
        
        Parameters:
            item (int): Sample index in the batch
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation, may contain 'text', 'image', 'anchor' keys
        
        Returns:
            dict: Contains processed input data such as input_ids, attention_mask, etc.
        """

        raw_prompt = gen_batch.non_tensor_batch['raw_prompt'][item]
        data_source = gen_batch.non_tensor_batch['data_source'][item]
        apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})
        
        # Get observation components
        obs_texts = obs.get('text', None)
        obs_images = obs.get('image', None)
        obs_anchors = obs.get('anchor', None)
        obs_text = obs_texts[item] if obs_texts is not None else None
        obs_image = obs_images[item] if obs_images is not None else None
        obs_anchor = obs_anchors[item] if obs_anchors is not None else None
        is_multi_modal = obs_image is not None

        _obs_anchor = torch_to_numpy(obs_anchor, is_object=True) if isinstance(obs_anchor, torch.Tensor) else obs_anchor

        # Build chat structure
        # obs_content = raw_prompt[0]['content']
        # if '<image>' in obs_content: 
        #     obs_content = obs_content.replace('<image>', '')

        # Build chat structure
        obs_content = ''
        if obs_text is not None:
            obs_content += obs_text
        else:
            print(f"Warning: No text observation found!")

        
        chat = np.array([{
            "content": obs_content,
            "role": "user",
        }])
        
        # Apply chat template
        prompt_with_chat_template = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False,
            **apply_chat_template_kwargs
        )
        
        # Initialize return dict
        row_dict = {}
        
        # Process multimodal data
        if is_multi_modal:
            # Replace image placeholder with vision tokens
            raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            row_dict['multi_modal_data'] = {'image': [process_image(obs_image)]}
            image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}
            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while '<image>' in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<image>',
                        '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                        '<|vision_end|>',
                        1,
                    )
                    index += 1

                prompt_with_chat_template = prompt_with_chat_template.replace('<|placeholder|>',
                                                                                self.processor.image_token)

        else:
            raw_prompt = prompt_with_chat_template
        
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                            tokenizer=self.tokenizer,
                                                                            max_length=self.config.data.max_prompt_length,
                                                                            pad_token_id=self.tokenizer.pad_token_id,
                                                                            left_pad=True,
                                                                            truncation=self.config.data.truncation,)
        
        

        if is_multi_modal:

            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            )  # (3, seq_length)
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.config.data.max_prompt_length:
            if self.config.data.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.config.data.max_prompt_length :]
            elif self.config.data.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.config.data.max_prompt_length]
            elif self.config.data.truncation == "middle":
                left_half = self.config.data.max_prompt_length // 2
                right_half = self.config.data.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.config.data.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.config.data.max_prompt_length}.")

        # Build final output dict
        row_dict.update({
            'input_ids': input_ids[0],
            'attention_mask': attention_mask[0],
            'position_ids': position_ids[0],
            'raw_prompt_ids': raw_prompt_ids,
            'anchor_obs': _obs_anchor,
            'index': item,
            'data_source': data_source
        })

        if self.config.trainer.get("wandb_log_contexts", False):
            row_dict['prompt_text'] = prompt_with_chat_template

        if self.config.data.get('return_raw_chat', False):
            row_dict['raw_prompt'] = chat.tolist()
        
        return row_dict

    def preprocess_batch(
        self,
        gen_batch: DataProto, 
        obs: Dict, 
    ) -> DataProto:
        """
        Process a batch of observation samples, converting environment observations into model-processable format.
        
        Parameters:
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation dictionary
                - 'text' (None or List[str]): Text observation data
                - 'image' (np.ndarray or torch.Tensor): Image observation data
                - 'anchor' (None or Any): Anchor observation without any histories or additional info. (for GiGPO only).
        
        Returns:
            DataProto: Contains processed batch data with preserved metadata
        """
        batch_size = len(gen_batch.batch['input_ids'])
        processed_samples = []
        
        # Process each sample in parallel
        for item in range(batch_size):
            # Extract per-sample observations
            processed = self.preprocess_single_sample(
                item=item,
                gen_batch=gen_batch,
                obs=obs,
            )
            processed_samples.append(processed)
        
        # Aggregate batch data
        batch = collate_fn(processed_samples)
        
        # Create DataProto with preserved metadata
        new_batch = DataProto.from_single_dict(
            data=batch,
            meta_info=gen_batch.meta_info
        )

        return new_batch


    def gather_rollout_data(
            self,
            total_batch_list: List[List[Dict]],
            episode_rewards: np.ndarray,
            episode_lengths: np.ndarray,
            success: Dict[str, np.ndarray],
            traj_uid: np.ndarray,
            tool_callings: np.ndarray,
            ) -> DataProto:
        """
        Collect and organize trajectory data, handling batch size adjustments to meet parallel training requirements.
        
        Parameters:
            total_batch_list (List[List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
            tool_callings (np.ndarray): Number of tool callings for each environment
        Returns:
            DataProto: Collected and organized trajectory data
        """
        batch_size = len(total_batch_list)

        success_rate = {}
        for key, value in success.items():
            success_rate[key] = np.mean(value)
        
        effective_batch = []
        for bs in range(batch_size):
            # sum the rewards for each data in total_batch_list[bs]
            for data in total_batch_list[bs]:
                assert traj_uid[bs] == data['traj_uid'], "data is not from the same trajectory"
                if data['active_masks']:
                    # episode_rewards
                    data['episode_rewards'] = episode_rewards[bs]
                    # episode_lengths
                    data['episode_lengths'] = episode_lengths[bs]
                    # tool_callings
                    data['tool_callings'] = tool_callings[bs]
                    # success_rate
                    for key, value in success_rate.items():
                        data[key] = value

                    effective_batch.append(data)
            
        # Convert trajectory data to DataProto format
        gen_batch_output = DataProto.from_single_dict(
            data=collate_fn(effective_batch)
        )
        return gen_batch_output

    def vanilla_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        """
        Collects trajectories through parallel agent-environment agent_loop.
        Parameters:
            gen_batch (DataProto): Initial batch with prompts to start the agent_loop
            actor_rollout_wg (WorkerGroup): Worker group containing the actor model for policy decisions
            envs (EnvironmentManagerBase): Environment manager containing parallel environment instances
        
        Returns:
            total_batch_list (List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
        """

        batch_size = len(gen_batch.batch)

        # Initial observations from the environment
        obs, infos = envs.reset(kwargs=gen_batch.non_tensor_batch.pop('env_kwargs', None))
        reset_infos = infos

        lenght_obs = len(obs['text']) if obs['text'] is not None else len(obs['image'])
        assert len(gen_batch.batch) == lenght_obs, f"gen_batch size {len(gen_batch.batch)} does not match obs size {lenght_obs}"
        
        if self.config.env.rollout.n > 0: # env grouping
            uid_batch = []
            for i in range(batch_size):
                if i % self.config.env.rollout.n == 0:
                    uid = str(uuid.uuid4())
                uid_batch.append(uid)
            uid_batch = np.array(uid_batch, dtype=object)
        else: # no env grouping, set all to the same uid
            uid = str(uuid.uuid4())
            uid_batch = np.array([uid for _ in range(len(gen_batch.batch))], dtype=object)
        is_done = np.zeros(batch_size, dtype=bool)
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        total_batch_list = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]
        episode_lengths = np.zeros(batch_size, dtype=np.float32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        tool_callings = np.zeros(batch_size, dtype=np.float32)
        valid_action_counts = np.zeros(batch_size, dtype=np.float32)
        active_action_counts = np.zeros(batch_size, dtype=np.float32)
        is_alfworld = "alfworld" in str(self.config.env.env_name).lower()
        _global_step = gen_batch.meta_info.get("global_step", None)
        _validate = bool(gen_batch.meta_info.get("validate", False))
        trajectory_records = [
            {
                "global_step": self._json_safe(_global_step),
                "trajectory_index": i,
                "uid": None,
                "traj_uid": traj_uid[i],
                "task": getattr(envs, "tasks", [None] * batch_size)[i],
                "gamefile": reset_infos[i].get("extra.gamefile") if (is_alfworld and i < len(reset_infos)) else None,
                "task_type": None,
                "prompt_text": None,
                "contexts": [],
                "steps": [],
            }
            for i in range(batch_size)
        ] if (
            self._trajectory_logging_enabled()
            and self._should_log_trajectories_this_call(_global_step, _validate)
        ) else None

        if trajectory_records is not None and is_alfworld:
            for i, record in enumerate(trajectory_records):
                record["pick_and_place_correct_form"] = PICK_AND_PLACE_CORRECT_FORM
                gamefile = record["gamefile"] or ""
                record["task_type"] = "pick_and_place" if (
                    "pick_and_place" in gamefile and "pick_two_obj_and_place" not in gamefile
                ) else "other"
        # Trajectory collection loop
        for _step in range(self.config.env.max_steps):
            active_masks = np.logical_not(is_done)
            if not active_masks.any():
                break

            batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs)
            prompt_texts = batch.non_tensor_batch.get("prompt_text", None)
            if trajectory_records is not None and prompt_texts is not None:
                for i in range(batch_size):
                    if trajectory_records[i].get("prompt_text") is None:
                        trajectory_records[i]["prompt_text"] = self._json_safe(prompt_texts[i])
                    if active_masks[i] and len(trajectory_records[i]["contexts"]) < 8:
                        trajectory_records[i]["contexts"].append(
                            {
                                "step": _step + 1,
                                "prompt_text": self._json_safe(prompt_texts[i]),
                            }
                        )

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            batch_input = batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            batch_input.meta_info = gen_batch.meta_info

            # pad to be divisible by dp_size
            batch_input_padded, pad_size = pad_dataproto_to_divisor(batch_input, actor_rollout_wg.world_size)
            batch_output_padded = actor_rollout_wg.generate_sequences(batch_input_padded)
            # # unpad
            batch_output = unpad_dataproto(batch_output_padded, pad_size=pad_size)

            batch.non_tensor_batch['uid'] = uid_batch
            batch.non_tensor_batch['traj_uid'] = traj_uid

            batch = batch.union(batch_output)
            
            text_actions = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)
            parsed_think_actions = [self._extract_think_action(t) for t in text_actions]
            
            next_obs, rewards, dones, infos = envs.step(text_actions)

            
            if len(rewards.shape) == 2:
                rewards = rewards.squeeze(1)
            if len(dones.shape) == 2:
                # dones is numpy, delete a dimension
                dones = dones.squeeze(1)

            if 'is_action_valid' in infos[0]:
                batch.non_tensor_batch['is_action_valid'] = np.array([info['is_action_valid'] for info in infos], dtype=bool)
            else:
                batch.non_tensor_batch['is_action_valid'] = np.ones(batch_size, dtype=bool)
            valid_action_counts[active_masks] += batch.non_tensor_batch['is_action_valid'][active_masks].astype(np.float32)
            active_action_counts[active_masks] += 1.0

            if 'tool_calling' in infos[0]:
                tool_callings[active_masks] += np.array([info['tool_calling'] for info in infos], dtype=np.float32)[active_masks]
            # Create reward tensor, only assign rewards for active environments
            # episode_rewards += torch_to_numpy(rewards) * torch_to_numpy(active_masks)
            episode_rewards[active_masks] += torch_to_numpy(rewards)[active_masks]
            episode_lengths[active_masks] += 1

            assert len(rewards) == batch_size, f"env should return rewards for all environments, got {len(rewards)} rewards for {batch_size} environments"
            batch.non_tensor_batch['rewards'] = torch_to_numpy(rewards, is_object=True)
            batch.non_tensor_batch['active_masks'] = torch_to_numpy(active_masks, is_object=True)

            if trajectory_records is not None:
                obs_anchor = obs.get("anchor", None)
                next_obs_anchor = next_obs.get("anchor", None)
                obs_text = obs.get("text", None) if self.config.trainer.get("trajectory_include_prompt", False) else None
                for i in range(batch_size):
                    if not active_masks[i]:
                        continue
                    info_i = infos[i]
                    step_record = {
                        "step": _step + 1,
                        "active": self._json_safe(active_masks[i]),
                        "observation": self._json_safe(obs_anchor[i] if obs_anchor is not None else None),
                        "raw_response": self._json_safe(text_actions[i]),
                        "think": self._json_safe(parsed_think_actions[i][0]),
                        "action_text": self._json_safe(parsed_think_actions[i][1]),
                        "model_output": self._json_safe(text_actions[i]),
                        "env_action": self._json_safe(info_i.get("env_action", None)),
                        "is_action_valid": self._json_safe(info_i.get("is_action_valid", None)),
                        "reward": self._json_safe(torch_to_numpy(rewards)[i]),
                        "task_score": self._json_safe(info_i.get("task_score", None)),
                        "done": self._json_safe(dones[i]),
                        "won": self._json_safe(info_i.get("won", None)),
                        "goal_condition_success_rate": self._json_safe(
                            info_i.get("goal_condition_success_rate", None)
                        ),
                        "next_observation": self._json_safe(
                            next_obs_anchor[i] if next_obs_anchor is not None else None
                        ),
                    }
                    if obs_text is not None:
                        step_record["prompt"] = self._json_safe(obs_text[i])
                    trajectory_records[i]["uid"] = uid_batch[i]
                    trajectory_records[i]["steps"].append(step_record)
            
            # Update episode lengths for active environments
            batch_list: list[dict] = to_list_of_dict(batch)

            for i in range(batch_size):
                total_batch_list[i].append(batch_list[i])
                total_infos[i].append(infos[i])

            # Update done states
            is_done = np.logical_or(is_done, dones)
                
            # Update observations for next step
            obs = next_obs

            # Break if all environments are done
            if is_done.all():
                break
        
        success: Dict[str, np.ndarray] = envs.success_evaluator(
                    total_infos=total_infos,
                    total_batch_list=total_batch_list,
                    episode_rewards=episode_rewards, 
                    episode_lengths=episode_lengths,
                    )
        valid_action_ratios = np.divide(
            valid_action_counts,
            np.maximum(active_action_counts, 1.0),
        )

        if trajectory_records is not None:
            success_values = success.get("success_rate", np.zeros(batch_size, dtype=np.float32))
            for i, record in enumerate(trajectory_records):
                record["episode_reward"] = self._json_safe(episode_rewards[i])
                record["episode_length"] = self._json_safe(episode_lengths[i])
                record["tool_callings"] = self._json_safe(tool_callings[i])
                record["valid_action_rate"] = self._json_safe(valid_action_ratios[i])
                record["success_rate"] = self._json_safe(success_values[i]) if i < len(success_values) else None
                if record["task_type"] == "pick_and_place":
                    record["pick_and_place_success_rate"] = record["success_rate"]
            self._dump_contexts(
                [
                    {
                        "global_step": r["global_step"],
                        "trajectory_index": r["trajectory_index"],
                        "task": r["task"],
                        "contexts": r.get("contexts", []),
                        "task_type": r.get("task_type"),
                    }
                    for r in trajectory_records[: max(1, int(self.config.trainer.get("wandb_context_samples_per_rollout", 5)))]
                ]
            )
            self._dump_trajectories(
                trajectory_records,
                validate=bool(gen_batch.meta_info.get("validate", False)),
            )

        self._print_compact_rollout_summary(
            gen_batch=gen_batch,
            episode_lengths=episode_lengths,
            success=success,
            valid_action_ratios=valid_action_ratios,
        )
        
        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings
    
    def dynamic_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        """
        Conduct dynamic rollouts until a target batch size is met. 
        Keeps sampling until the desired number of effective trajectories is collected.
        Adopted from DAPO (https://arxiv.org/abs/2503.14476)

        Args:
            gen_batch (DataProto): Initial batch for rollout.
            actor_rollout_wg: Actor model workers for generating responses.
            envs (EnvironmentManagerBase): Environment manager instance.

        Returns:
            total_batch_list (List[Dict]): Complete set of rollout steps.
            total_episode_rewards (np.ndarray): Accumulated rewards.
            total_episode_lengths (np.ndarray): Lengths per episode.
            total_success (Dict[str, np.ndarray]): Success metrics.
            total_traj_uid (np.ndarray): Trajectory IDs.
        """
        total_batch_list = []
        total_episode_rewards = []
        total_episode_lengths = []
        total_success = []
        total_traj_uid = []
        total_tool_callings = []
        try_count: int = 0
        max_try_count = self.config.algorithm.filter_groups.max_num_gen_batches

        while len(total_batch_list) < self.config.data.train_batch_size * self.config.env.rollout.n and try_count < max_try_count:

            if len(total_batch_list) > 0:
                print(f"valid num={len(total_batch_list)} < target num={self.config.data.train_batch_size * self.config.env.rollout.n}. Keep generating... ({try_count}/{max_try_count})")
            try_count += 1

            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = filter_group_data(batch_list=batch_list, 
                                                                                                episode_rewards=episode_rewards, 
                                                                                                episode_lengths=episode_lengths, 
                                                                                                success=success, 
                                                                                                traj_uid=traj_uid, 
                                                                                                tool_callings=tool_callings, 
                                                                                                config=self.config,
                                                                                                last_try=(try_count == max_try_count),
                                                                                                )
            
            total_batch_list += batch_list
            total_episode_rewards.append(episode_rewards)
            total_episode_lengths.append(episode_lengths)
            total_success.append(success)
            total_traj_uid.append(traj_uid)
            total_tool_callings.append(tool_callings)

        total_episode_rewards = np.concatenate(total_episode_rewards, axis=0)
        total_episode_lengths = np.concatenate(total_episode_lengths, axis=0)
        total_success = {key: np.concatenate([success[key] for success in total_success], axis=0) for key in total_success[0].keys()}
        total_traj_uid = np.concatenate(total_traj_uid, axis=0)
        total_tool_callings = np.concatenate(total_tool_callings, axis=0)

        return total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, total_tool_callings

    def multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            is_train: bool = True,
            ) -> DataProto:
        """
        Select and run the appropriate rollout loop (dynamic or vanilla).

        Args:
            gen_batch (DataProto): Initial prompt batch.
            actor_rollout_wg: Actor model workers.
            envs (EnvironmentManagerBase): Environment manager for interaction.
            is_train (bool): Whether in training mode (affects dynamic sampling).

        Returns:
            DataProto: Final collected trajectory data with metadata.
        """
        if is_train:
            gen_batch = gen_batch.repeat(repeat_times=self.config.env.rollout.n, interleave=True)
            
        # Initial observations from the environment
        if self.config.algorithm.filter_groups.enable and is_train:
            # Dynamic Sampling (for DAPO and Dynamic GiGPO)
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings = \
                self.dynamic_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
        else:
            # Vanilla Sampling   
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings = \
                self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
        assert len(total_batch_list) == len(total_episode_rewards)
        assert len(total_batch_list) == len(total_episode_lengths)
        assert len(total_batch_list) == len(total_traj_uid)
        assert len(total_batch_list) == len(totoal_tool_callings)
        

        # Create trajectory data
        gen_batch_output: DataProto = self.gather_rollout_data(
            total_batch_list=total_batch_list,
            episode_rewards=total_episode_rewards,
            episode_lengths=total_episode_lengths,
            success=total_success,
            traj_uid=total_traj_uid,
            tool_callings=totoal_tool_callings,
        )
        
        return gen_batch_output
