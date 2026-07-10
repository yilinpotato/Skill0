"""
Hierarchical Skill Library: L0 (hot buffer) / L1 (warm validated) / L2 (cold internalization queue).

Responsibilities:
  - Provide hot skills (L0+L1) to the local executor as Context.
  - Provide cold skills (L2) to the RL trainer for parameter internalization.
  - Track per-skill stability via call success rate and modification frequency.
  - Promote/demote skills based on stability thresholds.
  - Persist runtime state inline with the existing skill JSON file.

Metadata fields (prefixed with `_` to keep skill content clean):
  _layer            "L0" | "L1" | "L2"
  _stability_score  float in [0, 1]
  _total_calls      int
  _successful_calls int
  _update_count     int   (how many times cloud has modified this skill)
  _stable_cycles    int   (cycles since last modification)
"""

import json
import os
from dataclasses import dataclass
from typing import Optional

from .skills_only_memory import SkillsOnlyMemory


@dataclass
class PromotionThresholds:
    l0_to_l1_score: float = 0.6
    l0_to_l1_stable_cycles: int = 3
    l1_to_l2_score: float = 0.9
    l1_to_l2_stable_cycles: int = 5


# ---------------------------------------------------------------------------
# Stability score
# ---------------------------------------------------------------------------

def compute_stability(skill: dict) -> float:
    total = skill.get('_total_calls', 0)
    success = skill.get('_successful_calls', 0)
    stable = skill.get('_stable_cycles', 0)
    updates = skill.get('_update_count', 0)

    success_rate = success / total if total > 0 else 0.0
    stability_part = stable / max(stable + updates, 1)
    return round(0.7 * success_rate + 0.3 * stability_part, 4)


# ---------------------------------------------------------------------------
# HierarchicalSkillLibrary
# ---------------------------------------------------------------------------

class HierarchicalSkillLibrary:
    METADATA_FIELDS = ('_layer', '_stability_score', '_total_calls',
                       '_successful_calls', '_update_count', '_stable_cycles')

    def __init__(
        self,
        skills_json_path: str,
        retrieval_mode: str = 'template',
        thresholds: Optional[PromotionThresholds] = None,
        **memory_kwargs,
    ):
        self.skills_json_path = skills_json_path
        self.thresholds = thresholds or PromotionThresholds()
        self.memory = SkillsOnlyMemory(
            skills_json_path=skills_json_path,
            retrieval_mode=retrieval_mode,
            **memory_kwargs,
        )
        self._init_metadata()

    # ------------------------------------------------------------------
    # Metadata initialisation & iteration
    # ------------------------------------------------------------------

    def _all_skills(self):
        """Yield (kind, task_type_or_None, skill_dict) for every skill."""
        for s in self.memory.skills.get('general_skills', []):
            yield ('general', None, s)
        for task_type, skills in self.memory.skills.get('task_specific_skills', {}).items():
            for s in skills:
                yield ('task_specific', task_type, s)

    def _init_metadata(self):
        for _, _, skill in self._all_skills():
            skill.setdefault('_layer', 'L0')
            skill.setdefault('_stability_score', 0.0)
            skill.setdefault('_total_calls', 0)
            skill.setdefault('_successful_calls', 0)
            skill.setdefault('_update_count', 0)
            skill.setdefault('_stable_cycles', 0)

    def _find_skill(self, skill_id: str) -> Optional[dict]:
        for _, _, s in self._all_skills():
            if s.get('skill_id') == skill_id:
                return s
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_patch(self, patch: dict, scope: str = 'general', task_type: Optional[str] = None) -> str:
        """
        Add a new Skill Patch (from cloud distillation) into L0.
        Returns the skill_id used.
        """
        skill_id = patch.get('skill_id') or self._next_skill_id(scope, task_type)
        skill = {
            'skill_id': skill_id,
            'title': patch.get('title', ''),
            'principle': patch.get('principle') or patch.get('action_flow', ''),
            'when_to_apply': patch.get('when_to_apply') or patch.get('trigger_condition', ''),
            '_layer': 'L0',
            '_stability_score': 0.0,
            '_total_calls': 0,
            '_successful_calls': 0,
            '_update_count': 0,
            '_stable_cycles': 0,
        }

        existing = self._find_skill(skill_id)
        if existing is not None:
            # Modification: bump update_count, reset stable_cycles, push back to L0
            for k in ('title', 'principle', 'when_to_apply'):
                if k in patch:
                    existing[k] = patch[k]
            existing['_update_count'] = existing.get('_update_count', 0) + 1
            existing['_stable_cycles'] = 0
            existing['_layer'] = 'L0'
            existing['_stability_score'] = compute_stability(existing)
            return skill_id

        if scope == 'general':
            self.memory.add_skills([skill], category='general')
        else:
            self.memory.add_skills([skill], category=task_type or scope)
        return skill_id

    def _next_skill_id(self, scope: str, task_type: Optional[str]) -> str:
        prefix = 'gen' if scope == 'general' else (task_type or 'task')[:3].lower()
        existing = {s.get('skill_id', '') for _, _, s in self._all_skills()}
        i = 1
        while f"{prefix}_{i:03d}" in existing:
            i += 1
        return f"{prefix}_{i:03d}"

    def get_hot_context(self, task_description: str, current_observation: str = '',
                        top_k: int = 6) -> dict:
        """Retrieve L0+L1 skills via the underlying memory, gated by active layer ids."""
        active_global, active_task = self._collect_layer_ids({'L0', 'L1'})
        # Apply filters via SkillsOnlyMemory's allow-lists
        self.memory.set_global_skill_active_ids(active_global)
        for task_type, ids in active_task.items():
            self.memory.set_task_specific_skill_active_ids(task_type, ids)

        return self.memory.retrieve(
            task_description=task_description,
            top_k=top_k,
            current_observation=current_observation,
        )

    def get_cold_skills_for_rl(self) -> list:
        """Return all L2 skills, ready to be internalized via RL."""
        return [s for _, _, s in self._all_skills() if s.get('_layer') == 'L2']

    def update_stability(self, skill_ids, success: bool) -> None:
        """Bump call stats for a set of skill_ids that were used in one episode."""
        if isinstance(skill_ids, str):
            skill_ids = [skill_ids]
        for sid in skill_ids:
            s = self._find_skill(sid)
            if s is None:
                continue
            s['_total_calls'] = s.get('_total_calls', 0) + 1
            if success:
                s['_successful_calls'] = s.get('_successful_calls', 0) + 1
            s['_stability_score'] = compute_stability(s)

    def tick_cycle(self) -> None:
        """Advance a cloud-update cycle: increment stable_cycles for unmodified skills,
        then evaluate promotions. Should be called once per cloud distillation cycle."""
        for _, _, s in self._all_skills():
            s['_stable_cycles'] = s.get('_stable_cycles', 0) + 1
            s['_stability_score'] = compute_stability(s)
        self._evaluate_promotions()

    def _evaluate_promotions(self) -> list:
        """Promote skills that meet thresholds. Returns list of (skill_id, from, to)."""
        moved = []
        t = self.thresholds
        for kind, _, s in self._all_skills():
            layer = s.get('_layer', 'L0')
            score = s.get('_stability_score', 0.0)
            cycles = s.get('_stable_cycles', 0)
            if layer == 'L0' and score >= t.l0_to_l1_score and cycles >= t.l0_to_l1_stable_cycles:
                s['_layer'] = 'L1'
                moved.append((s.get('skill_id'), 'L0', 'L1'))
            elif (layer == 'L1' and kind == 'general'
                  and score >= t.l1_to_l2_score and cycles >= t.l1_to_l2_stable_cycles):
                s['_layer'] = 'L2'
                moved.append((s.get('skill_id'), 'L1', 'L2'))
        if moved:
            print(f"[HierarchicalSkillLibrary] Promotions: {moved}")
        return moved

    def promote(self, skill_id: str) -> bool:
        """Manually promote a skill one layer (L0->L1->L2)."""
        s = self._find_skill(skill_id)
        if s is None:
            return False
        order = ['L0', 'L1', 'L2']
        cur = s.get('_layer', 'L0')
        if cur not in order or cur == 'L2':
            return False
        s['_layer'] = order[order.index(cur) + 1]
        return True

    def mark_internalized(self, skill_id: str) -> bool:
        """Remove a skill from the library after RL has internalized it into params."""
        if self._find_skill(skill_id) is None:
            return False
        return self.memory.remove_skill(skill_id)

    # ------------------------------------------------------------------
    # Layer filters
    # ------------------------------------------------------------------

    def _collect_layer_ids(self, layers: set) -> tuple[list, dict]:
        """Return (global_active_ids, {task_type: [ids]}) restricted to given layers."""
        general = []
        task_map: dict[str, list] = {}
        for kind, task_type, s in self._all_skills():
            if s.get('_layer') in layers:
                sid = s.get('skill_id')
                if kind == 'general':
                    general.append(sid)
                else:
                    task_map.setdefault(task_type, []).append(sid)
        return general, task_map

    # ------------------------------------------------------------------
    # SkillsOnlyMemory interface delegation (for trainer compatibility)
    # ------------------------------------------------------------------

    def retrieve(self, task_description: str, top_k: int = 6,
                 current_observation: Optional[str] = None, **kwargs):
        """Delegate to underlying memory, but filter by L0+L1 layers."""
        return self.get_hot_context(task_description, current_observation or '', top_k)

    def set_global_skill_top_k(self, top_k: Optional[int]):
        self.memory.set_global_skill_top_k(top_k)

    def get_global_skill_top_k(self) -> Optional[int]:
        return self.memory.get_global_skill_top_k()

    def set_global_skill_active_ids(self, skill_ids: Optional[list]):
        self.memory.set_global_skill_active_ids(skill_ids)

    def get_global_skill_active_ids(self) -> Optional[list]:
        return self.memory.get_global_skill_active_ids()

    def set_task_specific_skill_active_ids(self, task_type: str, skill_ids: Optional[list]):
        self.memory.set_task_specific_skill_active_ids(task_type, skill_ids)

    def get_task_specific_skill_active_ids(self, task_type: str) -> Optional[list]:
        return self.memory.get_task_specific_skill_active_ids(task_type)

    def remove_global_skill_ids(self, skill_ids: list) -> list:
        return self.memory.remove_global_skill_ids(skill_ids)

    def remove_task_specific_skill_ids(self, task_type: str, skill_ids: list) -> list:
        return self.memory.remove_task_specific_skill_ids(task_type, skill_ids)

    def add_skills(self, new_skills: list, category: str = 'general') -> int:
        return self.memory.add_skills(new_skills, category)

    def remove_skill(self, skill_id: str) -> bool:
        return self.memory.remove_skill(skill_id)

    def format_for_prompt(self, retrieved_memories: dict) -> str:
        return self.memory.format_for_prompt(retrieved_memories)

    def get_all_global_skill_ids(self) -> list:
        return self.memory.get_all_global_skill_ids()

    def get_all_task_specific_skill_ids(self, task_type: str) -> list:
        return self.memory.get_all_task_specific_skill_ids(task_type)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None) -> None:
        """Persist skills (with metadata fields) back to JSON."""
        target = path or self.skills_json_path
        os.makedirs(os.path.dirname(target) or '.', exist_ok=True)
        with open(target, 'w') as f:
            json.dump(self.memory.skills, f, indent=2, ensure_ascii=False)
        print(f"[HierarchicalSkillLibrary] Saved to {target}")

    def stats(self) -> dict:
        counts = {'L0': 0, 'L1': 0, 'L2': 0}
        for _, _, s in self._all_skills():
            counts[s.get('_layer', 'L0')] = counts.get(s.get('_layer', 'L0'), 0) + 1
        return counts
