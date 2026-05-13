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

from typing import List, Optional
import re


def _normalize_action(text: str) -> str:
    text = text.strip().lower()
    text = text.strip("`'\" \n\t")
    text = re.sub(r"^\s*(action\s*:|next action\s*:)\s*", "", text)
    text = re.sub(r"^\s*(i will|i should|let'?s|we should)\s+", "", text)
    text = re.sub(r"^(now\s+)?(go|move|proceed)\s+to\s+", "go to ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" .。")
    return text


def _extract_tagged_action(text: str) -> Optional[str]:
    matches = re.findall(r"<action>\s*(.*?)\s*</action>", text, flags=re.IGNORECASE | re.DOTALL)
    if not matches:
        return None
    # Use the last action block. Small models sometimes draft invalid earlier
    # action blocks before finally emitting the intended command.
    return matches[-1]


def _match_admissible_action(candidate: str, action_pool: List[str]) -> Optional[str]:
    pool = [a for a in action_pool if a != "help"]
    normalized_to_original = {_normalize_action(a): a for a in pool}
    normalized_candidate = _normalize_action(candidate)

    if normalized_candidate in normalized_to_original:
        return normalized_to_original[normalized_candidate]

    # Recover common verbose generations, e.g. "I will go to drawer 1 now".
    # Prefer the longest command to avoid matching "look" inside longer text.
    for normalized_action, original_action in sorted(
        normalized_to_original.items(), key=lambda item: len(item[0]), reverse=True
    ):
        pattern = r"(?<!\w)" + re.escape(normalized_action) + r"(?!\w)"
        if re.search(pattern, normalized_candidate):
            return original_action

    return None


def alfworld_projection(actions: List[str], action_pools: List[List[str]]):
    """
    An function to process the actions
    actions: the list of actions to be processeed, it is a list of strings.
    action_pools: the list of action pools, each pool is a list of strings.
    """

    valids = [0] * len(actions)

    for i in range(len(actions)):
        original_str = actions[i]
        lower_output = original_str.lower()

        has_chinese = re.search(r'[\u4e00-\u9fff]', original_str) is not None

        candidate = _extract_tagged_action(original_str)
        has_action_tag = candidate is not None
        if candidate is None:
            # Fallback for partially formatted outputs. This keeps rollouts
            # moving while still marking the sample invalid for penalty.
            candidate = original_str

        matched_action = _match_admissible_action(candidate, action_pools[i])
        if matched_action is None:
            matched_action = _match_admissible_action(original_str, action_pools[i])

        if matched_action is not None:
            actions[i] = matched_action
            # If we can reliably recover an admissible action, treat it as valid.
            # Qwen thinking models may emit long reasoning and miss the final
            # action tag even when the decoded command itself is correct.
            valids[i] = int(not has_chinese)
        else:
            actions[i] = _normalize_action(candidate)[-80:]
            valids[i] = 0

    return actions, valids
