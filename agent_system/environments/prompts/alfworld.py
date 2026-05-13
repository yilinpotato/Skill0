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

# --------------------- ALFWorld --------------------- #
ALFWORLD_TEMPLATE_NO_HIS = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
Your assistant response may already begin inside an opened <think> block.
Keep the reasoning very short, then close it with </think> immediately.
Immediately after that, copy exactly one command from the admissible actions list and put only that command inside <action> </action> tags.
The final response format must end as </think><action>exact admissible command</action>.
Do not invent commands. Do not write explanations inside <action>. Do not output more than one action.
If you already know the command, close </think> right away and output the action immediately.
"""

ALFWORLD_TEMPLATE = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
Your assistant response may already begin inside an opened <think> block.
Keep the reasoning very short, then close it with </think> immediately.
Immediately after that, copy exactly one command from the admissible actions list and put only that command inside <action> </action> tags.
The final response format must end as </think><action>exact admissible command</action>.
Do not invent commands. Do not write explanations inside <action>. Do not output more than one action.
If you already know the command, close </think> right away and output the action immediately.
"""

ALFWORLD_TEMPLATE_WITH_MEMORY = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}

## Retrieved Relevant Experience

{retrieved_memories}

## Current Progress

Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
Your assistant response may already begin inside an opened <think> block.
Keep the reasoning very short, then close it with </think> immediately.
Immediately after that, copy exactly one command from the admissible actions list and put only that command inside <action> </action> tags.
The final response format must end as </think><action>exact admissible command</action>.
Do not invent commands. Do not write explanations inside <action>. Do not output more than one action.
If you already know the command, close </think> right away and output the action immediately.
"""
