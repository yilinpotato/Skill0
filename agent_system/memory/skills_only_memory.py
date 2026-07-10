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

"""
Lightweight skills-only memory system.

This is a simplified version of RetrievalMemory that only uses Claude-style skills
without the overhead of loading and indexing trajectory memories.

Supports two retrieval modes:
  - "template": keyword-based task type detection + return all task-specific skills
    (original behaviour, zero latency, no GPU needed)
  - "embedding": encode the task description with Qwen3-Embedding-0.6B and rank
    both general and task-specific skills by cosine similarity, so only the
    top-k most relevant ones are injected into the prompt
"""

import json
import os
from typing import Dict, Any, List, Optional
from .base import BaseMemory


class SkillsOnlyMemory(BaseMemory):
    """
    Lightweight memory system that only uses Claude-style skills.

    Retrieval mode is controlled by the ``retrieval_mode`` constructor argument:

    * ``"template"`` (default) – keyword matching selects the task category;
      skills are ranked by when_to_apply overlap with current observation.

    * ``"embedding"`` – semantic ranking via SentenceTransformer embeddings.

    * ``"llm"`` – an LLM (OpenAI-compatible endpoint) reads the full skill menu
      and selects the most relevant skill IDs given the current task + observation.
      Defaults to the same model used for SFT (Qwen3-4B-Thinking-2507) served
      via local vLLM.  Configurable via env vars:
        SKILL_LLM_BASE_URL  – endpoint, default http://localhost:8000/v1
        SKILL_LLM_API_KEY   – API key, default "EMPTY" (vLLM ignores it)
        SKILL_LLM_MODEL     – model name, default "Qwen3-4B-Thinking-2507"
      Falls back to template mode on any API error.
    """

    # ------------------------------------------------------------------ #
    # Construction                                                         #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        skills_json_path: str,
        retrieval_mode: str = "template",
        embedding_model_path: Optional[str] = None,
        task_specific_top_k: Optional[int] = None,
        global_skill_top_k: Optional[int] = None,
        global_skill_active_ids: Optional[List[str]] = None,
        task_specific_skill_active_ids: Optional[Dict[str, List[str]]] = None,
        llm_model: Optional[str] = None,
    ):
        """
        Args:
            skills_json_path:     Path to Claude-style skills JSON file.
            retrieval_mode:       ``"template"`` or ``"embedding"``.
            embedding_model_path: Local path (or HF model ID) for the
                                  SentenceTransformer embedding model.  Only
                                  used when ``retrieval_mode="embedding"``.
                                  Defaults to ``"Qwen/Qwen3-Embedding-0.6B"``.
            task_specific_top_k:  Maximum number of task-specific skills to
                                  return.  ``None`` means *return all* in
                                  template mode and use ``top_k`` (general
                                  skills count) in embedding mode.
            global_skill_top_k:   Optional override for the number of global
                                  skills injected.  Used by skill
                                  internalization curricula to fade global
                                  skills out while leaving task-specific
                                  skills visible.
            global_skill_active_ids:
                                  Optional explicit allow-list of global skill
                                  IDs.  Learned internalization uses this to
                                  remove learned global skills without touching
                                  task-specific skills.
            task_specific_skill_active_ids:
                                  Optional explicit allow-list of task-specific
                                  skill IDs per task type.
        """
        if retrieval_mode not in ("template", "embedding", "llm"):
            raise ValueError(
                f"retrieval_mode must be 'template', 'embedding', or 'llm', got '{retrieval_mode}'"
            )

        if not os.path.exists(skills_json_path):
            raise FileNotFoundError(f"Skills file not found: {skills_json_path}")

        with open(skills_json_path, 'r') as f:
            self.skills = json.load(f)

        self.retrieval_mode = retrieval_mode
        self.embedding_model_path = embedding_model_path or "Qwen/Qwen3-Embedding-0.6B"
        self.task_specific_top_k = task_specific_top_k
        self.global_skill_top_k = global_skill_top_k
        self.global_skill_active_ids = list(global_skill_active_ids) if global_skill_active_ids is not None else None
        self.task_specific_skill_active_ids = (
            {task_type: list(skill_ids) for task_type, skill_ids in task_specific_skill_active_ids.items()}
            if task_specific_skill_active_ids is not None
            else {}
        )

        # Lazy-initialised embedding state (only used in embedding mode)
        self._embedding_model = None
        self._skill_embeddings_cache: Optional[Dict] = None

        # LLM client (only used in llm mode)
        self._llm_client = None
        self._llm_model = llm_model or os.environ.get("SKILL_LLM_MODEL", "Qwen3-4B-Thinking-2507")

        n_general = len(self.skills.get('general_skills', []))
        n_task = sum(len(v) for v in self.skills.get('task_specific_skills', {}).values())
        n_mistakes = len(self.skills.get('common_mistakes', []))
        print(
            f"[SkillsOnlyMemory] Loaded skills: {n_general} general, "
            f"{n_task} task-specific, {n_mistakes} mistakes  "
            f"| retrieval_mode={retrieval_mode} | global_skill_top_k={global_skill_top_k}"
        )

        if retrieval_mode == "embedding":
            self._compute_skill_embeddings()

    # ------------------------------------------------------------------ #
    # Task-type detection (template mode)                                  #
    # ------------------------------------------------------------------ #

    def _detect_task_type(self, task_description: str) -> str:
        """Infer task category from task_description using keyword rules."""
        task_specific = self.skills.get('task_specific_skills', {})
        goal = task_description.lower()

        # ---- ALFWorld categories ----------------------------------------
        if 'pick_and_place' in task_specific or 'clean' in task_specific:
            if 'look at' in goal and 'under' in goal:
                return 'look_at_obj_in_light'
            elif 'clean' in goal:
                return 'clean'
            elif 'heat' in goal:
                return 'heat'
            elif 'cool' in goal:
                return 'cool'
            elif 'examine' in goal or 'find' in goal:
                return 'examine'
            else:
                return 'pick_and_place'

        # ---- WebShop categories -----------------------------------------
        elif 'apparel' in task_specific or 'electronics' in task_specific:
            if any(kw in goal for kw in [
                'shirt', 'dress', 'jacket', 'pant', 'coat', 'sweater',
                'blouse', 'clothing', 'clothes', 't-shirt',
            ]):
                return 'apparel'
            elif any(kw in goal for kw in [
                'shoe', 'boot', 'sneaker', 'sandal', 'heel', 'slipper',
                'footwear',
            ]):
                return 'footwear'
            elif any(kw in goal for kw in [
                'laptop', 'phone', 'computer', 'tablet', 'charger',
                'cable', 'headphone', 'speaker', 'camera', 'electronic',
            ]):
                return 'electronics'
            elif any(kw in goal for kw in [
                'necklace', 'ring', 'bracelet', 'earring', 'watch',
                'jewelry', 'bag', 'purse', 'wallet',
            ]):
                return 'accessories'
            elif any(kw in goal for kw in [
                'furniture', 'lamp', 'curtain', 'pillow', 'bedding',
                'decor', 'candle', 'vase', 'rug',
            ]):
                return 'home_decor'
            elif any(kw in goal for kw in [
                'cream', 'lotion', 'shampoo', 'conditioner', 'moisturizer',
                'serum', 'makeup', 'beauty', 'vitamin', 'supplement',
            ]):
                return 'beauty_health'
            else:
                return 'other'

        # ---- Fallback: first key in task_specific_skills, or 'unknown' --
        else:
            return next(iter(task_specific), 'unknown')

    # ------------------------------------------------------------------ #
    # Keyword-overlap scoring for on-demand retrieval (template mode)     #
    # ------------------------------------------------------------------ #

    _STOPWORDS = frozenset({
        'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'on',
        'at', 'by', 'for', 'with', 'about', 'as', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'from', 'up', 'down', 'out',
        'off', 'over', 'under', 'again', 'then', 'once', 'and', 'or', 'but',
        'if', 'while', 'that', 'this', 'it', 'its', 'not', 'no', 'any',
        'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
        'such', 'same', 'so', 'than', 'too', 'very', 'just', 'when', 'where',
        'which', 'who', 'how', 'what', 'there', 'their', 'they', 'them',
    })

    @classmethod
    def _tokenize(cls, text: str) -> set:
        import re
        tokens = re.findall(r'[a-z]+', text.lower())
        return {t for t in tokens if t not in cls._STOPWORDS and len(t) > 2}

    def _score_skill(self, skill: dict, query_tokens: set) -> float:
        """Score a skill by keyword overlap between query and when_to_apply + title."""
        when = skill.get('when_to_apply', '')
        title = skill.get('title', '')
        skill_tokens = self._tokenize(when + ' ' + title)
        if not skill_tokens:
            return 0.0
        overlap = len(query_tokens & skill_tokens)
        # Jaccard-like: overlap / union, but weighted toward query coverage
        return overlap / (len(query_tokens) + len(skill_tokens) - overlap + 1e-9)

    def _on_demand_retrieve(
        self,
        context: str,
        candidates: list,
        top_k: int,
    ) -> list:
        """Return top_k skills from candidates ranked by overlap with context."""
        if not candidates or top_k <= 0:
            return []
        query_tokens = self._tokenize(context)
        if not query_tokens:
            return candidates[:top_k]
        scored = [(self._score_skill(s, query_tokens), i, s) for i, s in enumerate(candidates)]
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [s for _, _, s in scored[:top_k]]

    # ------------------------------------------------------------------ #
    # Embedding helpers                                                    #
    # ------------------------------------------------------------------ #

    def _get_embedding_model(self):
        """Lazy-load the SentenceTransformer model (thread-safe for single process)."""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for embedding retrieval. "
                    "Install with: pip install sentence-transformers"
                )
            print(f"[SkillsOnlyMemory] Loading embedding model: {self.embedding_model_path}")
            self._embedding_model = SentenceTransformer(self.embedding_model_path)
            print("[SkillsOnlyMemory] Embedding model ready.")
        return self._embedding_model

    @staticmethod
    def _skill_to_text(skill: Dict[str, Any]) -> str:
        """Concatenate the skill fields most useful for semantic matching."""
        parts = []
        for field in ('title', 'principle', 'when_to_apply'):
            val = skill.get(field, '').strip()
            if val:
                parts.append(val)
        return ". ".join(parts)

    def _compute_skill_embeddings(self) -> Dict:
        """
        Pre-compute and cache normalised embeddings for every skill.

        The cache holds:
          ``items``      – flat list of ``(kind, task_type, skill_dict)``
          ``embeddings`` – numpy array of shape ``(n_skills, dim)``
          ``n_general``  – how many of the first rows correspond to general skills
        """
        if self._skill_embeddings_cache is not None:
            return self._skill_embeddings_cache

        import numpy as np

        general_items = [
            ('general', None, s)
            for s in self.skills.get('general_skills', [])
        ]
        task_items = [
            ('task_specific', task_type, s)
            for task_type, skills in self.skills.get('task_specific_skills', {}).items()
            for s in skills
        ]
        all_items = general_items + task_items
        texts = [self._skill_to_text(item[2]) for item in all_items]

        model = self._get_embedding_model()
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        self._skill_embeddings_cache = {
            'items': all_items,
            'embeddings': embeddings,
            'n_general': len(general_items),
        }
        print(
            f"[SkillsOnlyMemory] Cached embeddings for {len(all_items)} skills "
            f"({len(general_items)} general + {len(task_items)} task-specific)"
        )
        return self._skill_embeddings_cache

    def _embedding_retrieve(
        self,
        task_description: str,
        top_k_general: int,
        top_k_task_specific: int,
    ):
        """
        Retrieve the most relevant general and task-specific skills using
        cosine similarity between the task description and all cached skill
        embeddings.

        Args:
            task_description:   Free-form task goal string.
            top_k_general:      Number of general skills to return.
            top_k_task_specific: Number of task-specific skills to return
                                 (searched across **all** categories).

        Returns:
            Tuple of (general_skills, task_specific_skills).
        """
        import numpy as np

        cache = self._compute_skill_embeddings()
        model = self._get_embedding_model()

        query_emb = model.encode(
            [task_description],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )[0]  # shape: (dim,)

        sims = cache['embeddings'] @ query_emb  # cosine similarity, shape: (n,)

        n_general = cache['n_general']
        general_sims = sims[:n_general]
        task_sims = sims[n_general:]

        # Top-k general skills
        general_idx = np.argsort(general_sims)[::-1][:top_k_general]
        general_skills = [cache['items'][int(i)][2] for i in general_idx]

        # Top-k task-specific skills (cross-category search)
        task_idx = np.argsort(task_sims)[::-1][:top_k_task_specific]
        task_skills = [cache['items'][n_general + int(i)][2] for i in task_idx]

        return general_skills, task_skills

    # ------------------------------------------------------------------ #
    # LLM-based retrieval                                                  #
    # ------------------------------------------------------------------ #

    def _get_llm_client(self):
        if self._llm_client is None:
            from openai import OpenAI
            base_url = os.environ.get(
                "SKILL_LLM_BASE_URL",
                "http://localhost:8000/v1",  # default: local vLLM serving the SFT model
            )
            api_key = os.environ.get("SKILL_LLM_API_KEY", "EMPTY")
            self._llm_client = OpenAI(api_key=api_key, base_url=base_url)
        return self._llm_client

    def _build_skill_menu(self) -> tuple[str, Dict[str, dict]]:
        """Return a compact skill menu string and an id->skill lookup dict."""
        lookup: Dict[str, dict] = {}
        lines = []
        for s in self.skills.get('general_skills', []):
            sid = s.get('skill_id', '')
            lookup[sid] = s
            lines.append(f"[{sid}] (general) {s.get('title','')} — {s.get('when_to_apply','')}")
        for task_type, skills in self.skills.get('task_specific_skills', {}).items():
            for s in skills:
                sid = s.get('skill_id', '')
                lookup[sid] = s
                lines.append(f"[{sid}] ({task_type}) {s.get('title','')} — {s.get('when_to_apply','')}")
        return "\n".join(lines), lookup

    def _llm_retrieve(
        self,
        task_description: str,
        current_observation: str,
        top_k_general: int,
        top_k_task_specific: int,
    ) -> tuple[list, list]:
        """Ask the LLM to select the most relevant skill IDs, return (general, task_specific)."""
        menu, lookup = self._build_skill_menu()
        prompt = (
            f"Task: {task_description}\n"
            f"Current observation: {current_observation}\n\n"
            f"Available skills (id — when to apply):\n{menu}\n\n"
            f"Select up to {top_k_general} general skills and up to {top_k_task_specific} "
            f"task-specific skills that are most relevant RIGHT NOW.\n"
            f"Reply with ONLY a JSON object: "
            f'{{\"general\": [\"id1\", ...], \"task_specific\": [\"id1\", ...]}}'
        )
        try:
            client = self._get_llm_client()
            resp = client.chat.completions.create(
                model=self._llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=0.0,
            )
            raw = resp.choices[0].message.content.strip()
            # extract JSON even if wrapped in markdown
            import re
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            selected = json.loads(m.group()) if m else {}
        except Exception as e:
            print(f"[SkillsOnlyMemory] LLM retrieval failed ({e}), falling back to template")
            return None, None  # caller will fall back

        general_skills = [lookup[sid] for sid in selected.get('general', []) if sid in lookup]
        task_skills = [lookup[sid] for sid in selected.get('task_specific', []) if sid in lookup]
        return general_skills, task_skills

    def set_global_skill_top_k(self, top_k: Optional[int]):
        """Update the active global-skill budget used by retrieve()."""
        if top_k is not None:
            top_k = max(0, int(top_k))
        if self.global_skill_top_k != top_k:
            print(f"[SkillsOnlyMemory] global_skill_top_k: {self.global_skill_top_k} -> {top_k}")
        self.global_skill_top_k = top_k

    def get_global_skill_top_k(self) -> Optional[int]:
        """Return the active global-skill budget override, if any."""
        return self.global_skill_top_k

    def _effective_global_top_k(self, top_k: int) -> int:
        if self.global_skill_top_k is None:
            return max(0, int(top_k))
        return max(0, int(self.global_skill_top_k))

    def get_all_global_skill_ids(self) -> List[str]:
        """Return global/general skill IDs in file order."""
        ids = []
        for idx, skill in enumerate(self.skills.get('general_skills', [])):
            ids.append(skill.get('skill_id') or f"__global_idx_{idx}")
        return ids

    def _task_specific_skill_id(self, task_type: str, idx: int, skill: Dict[str, Any]) -> str:
        return skill.get('skill_id') or f"__task_{task_type}_{idx}"

    def get_all_task_specific_skill_ids(self, task_type: str) -> List[str]:
        """Return task-specific skill IDs for a task type in file order."""
        ids = []
        for idx, skill in enumerate(self.skills.get('task_specific_skills', {}).get(task_type, [])):
            ids.append(self._task_specific_skill_id(task_type, idx, skill))
        return ids

    def set_global_skill_active_ids(self, skill_ids: Optional[List[str]]):
        """Set the explicit active global skill allow-list."""
        new_ids = list(skill_ids) if skill_ids is not None else None
        if self.global_skill_active_ids != new_ids:
            old_count = None if self.global_skill_active_ids is None else len(self.global_skill_active_ids)
            new_count = None if new_ids is None else len(new_ids)
            print(f"[SkillsOnlyMemory] global_skill_active_ids count: {old_count} -> {new_count}")
        self.global_skill_active_ids = new_ids

    def get_global_skill_active_ids(self) -> Optional[List[str]]:
        """Return a copy of the explicit active global skill allow-list."""
        if self.global_skill_active_ids is None:
            return None
        return list(self.global_skill_active_ids)

    def set_task_specific_skill_active_ids(self, task_type: str, skill_ids: Optional[List[str]]):
        """Set the explicit active task-specific skill allow-list for one task type."""
        if skill_ids is None:
            if task_type in self.task_specific_skill_active_ids:
                old_count = len(self.task_specific_skill_active_ids[task_type])
                print(f"[SkillsOnlyMemory] task_specific_skill_active_ids[{task_type}]: {old_count} -> None")
                self.task_specific_skill_active_ids.pop(task_type, None)
            return

        new_ids = list(skill_ids)
        old_ids = self.task_specific_skill_active_ids.get(task_type)
        if old_ids != new_ids:
            old_count = None if old_ids is None else len(old_ids)
            new_count = len(new_ids)
            print(f"[SkillsOnlyMemory] task_specific_skill_active_ids[{task_type}]: {old_count} -> {new_count}")
        self.task_specific_skill_active_ids[task_type] = new_ids

    def get_task_specific_skill_active_ids(self, task_type: str) -> Optional[List[str]]:
        """Return a copy of the explicit active task-specific allow-list for one task type."""
        skill_ids = self.task_specific_skill_active_ids.get(task_type)
        if skill_ids is None:
            return None
        return list(skill_ids)

    def remove_task_specific_skill_ids(self, task_type: str, skill_ids: List[str]) -> List[str]:
        """Deactivate task-specific skills by ID for one task type."""
        if not skill_ids:
            return []
        if task_type not in self.task_specific_skill_active_ids:
            self.task_specific_skill_active_ids[task_type] = self.get_all_task_specific_skill_ids(task_type)

        remove_set = set(skill_ids)
        old_ids = list(self.task_specific_skill_active_ids.get(task_type, []))
        self.task_specific_skill_active_ids[task_type] = [sid for sid in old_ids if sid not in remove_set]
        removed = [sid for sid in old_ids if sid in remove_set]
        if removed:
            print(f"[SkillsOnlyMemory] Deactivated task-specific skills for {task_type}: {removed}")
        return removed

    def remove_global_skill_ids(self, skill_ids: List[str]) -> List[str]:
        """Deactivate global skills by ID without deleting them from the skill bank."""
        if not skill_ids:
            return []
        if self.global_skill_active_ids is None:
            self.global_skill_active_ids = self.get_all_global_skill_ids()

        remove_set = set(skill_ids)
        old_ids = list(self.global_skill_active_ids)
        self.global_skill_active_ids = [sid for sid in old_ids if sid not in remove_set]
        removed = [sid for sid in old_ids if sid in remove_set]
        if removed:
            print(f"[SkillsOnlyMemory] Deactivated global skills: {removed}")
        return removed

    def _active_global_skill_set(self) -> Optional[set]:
        if self.global_skill_active_ids is None:
            return None
        return set(self.global_skill_active_ids)

    def _filter_active_global_skills(self, skills: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        active_ids = self._active_global_skill_set()
        if active_ids is None:
            return skills
        filtered = []
        for idx, skill in enumerate(skills):
            skill_id = skill.get('skill_id') or f"__global_idx_{idx}"
            if skill_id in active_ids:
                filtered.append(skill)
        return filtered

    def _filter_active_task_specific_skills(self, task_type: str, skills: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        active_ids = self.task_specific_skill_active_ids.get(task_type)
        if active_ids is None:
            return skills
        active_set = set(active_ids)
        filtered = []
        for idx, skill in enumerate(skills):
            skill_id = self._task_specific_skill_id(task_type, idx, skill)
            if skill_id in active_set:
                filtered.append(skill)
        return filtered

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def retrieve(
        self,
        task_description: str,
        top_k: int = 6,
        current_observation: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Retrieve skills for a given task description.

        Args:
            task_description:    Current task goal string.
            top_k:               Number of general skills to include.
            current_observation: Current env observation text (template mode only).
                                 When provided, skills are ranked by keyword overlap
                                 with ``when_to_apply`` against this context rather
                                 than returned in document order.

        Returns:
            Dictionary with keys:
              - ``general_skills``       – list of skill dicts
              - ``task_specific_skills`` – list of skill dicts
              - ``mistakes_to_avoid``    – list of common-mistake dicts
              - ``task_type``            – detected task type string
              - ``task_specific_examples`` – always ``[]`` (reserved)
              - ``retrieval_mode``       – which mode was used
        """
        effective_global_top_k = self._effective_global_top_k(top_k)
        common_mistakes = self.skills.get('common_mistakes', [])[:5] if effective_global_top_k > 0 else []

        # ----------------------------------------------------------------
        # Embedding mode: semantic ranking of all skills
        # ----------------------------------------------------------------
        if self.retrieval_mode == "embedding":
            ts_top_k = self.task_specific_top_k if self.task_specific_top_k is not None else top_k
            general_budget = effective_global_top_k
            if self.global_skill_active_ids is not None:
                general_budget = max(general_budget, len(self.global_skill_active_ids))
            general_skills, task_skills = self._embedding_retrieve(
                task_description=task_description,
                top_k_general=general_budget,
                top_k_task_specific=ts_top_k,
            )
            general_skills = self._filter_active_global_skills(general_skills)[:effective_global_top_k]
            task_type = self._detect_task_type(task_description)
            return {
                'general_skills': general_skills,
                'task_specific_skills': task_skills,
                'mistakes_to_avoid': common_mistakes,
                'task_type': task_type,
                'task_specific_examples': [],
                'retrieval_mode': 'embedding',
                'global_skill_top_k': effective_global_top_k,
            }

        # ----------------------------------------------------------------
        # LLM mode: ask the model to select skill IDs from the full menu
        # ----------------------------------------------------------------
        if self.retrieval_mode == "llm":
            ts_top_k = self.task_specific_top_k if self.task_specific_top_k is not None else top_k
            obs = current_observation or task_description
            general_skills, task_skills = self._llm_retrieve(
                task_description=task_description,
                current_observation=obs,
                top_k_general=effective_global_top_k,
                top_k_task_specific=ts_top_k,
            )
            if general_skills is not None:  # None means API error → fall through to template
                general_skills = self._filter_active_global_skills(general_skills)[:effective_global_top_k]
                task_type = self._detect_task_type(task_description)
                return {
                    'general_skills': general_skills,
                    'task_specific_skills': task_skills,
                    'mistakes_to_avoid': common_mistakes,
                    'task_type': task_type,
                    'task_specific_examples': [],
                    'retrieval_mode': 'llm',
                    'global_skill_top_k': effective_global_top_k,
                }

        # ----------------------------------------------------------------
        # Template mode: keyword detection + on-demand scoring by
        # when_to_apply overlap with current_observation (or task_description)
        # ----------------------------------------------------------------
        task_type = self._detect_task_type(task_description)

        # Context for on-demand scoring: prefer live observation, fall back to goal
        scoring_context = (current_observation or '') + ' ' + task_description

        # General skills: always include dynamic skills, then score the rest
        all_general = self.skills.get('general_skills', [])
        dynamic_skills = [s for s in all_general if s.get('skill_id', '').startswith('dyn_')]
        static_skills = [s for s in all_general if not s.get('skill_id', '').startswith('dyn_')]

        if effective_global_top_k <= 0:
            general_skills = []
        else:
            candidate_general = dynamic_skills + static_skills
            if self.global_skill_active_ids is not None:
                candidate_general = self._filter_active_global_skills(candidate_general)
            # On-demand: rank by when_to_apply overlap with current context
            general_skills = self._on_demand_retrieve(
                scoring_context, candidate_general, effective_global_top_k
            )

        # Task-specific skills: score all candidates for this task type
        all_task_skills = self.skills.get('task_specific_skills', {}).get(task_type, [])
        filtered_task_skills = self._filter_active_task_specific_skills(task_type, all_task_skills)
        ts_budget = self.task_specific_top_k if self.task_specific_top_k is not None else len(filtered_task_skills)
        task_skills = self._on_demand_retrieve(scoring_context, filtered_task_skills, ts_budget)

        return {
            'general_skills': general_skills,
            'task_specific_skills': task_skills,
            'mistakes_to_avoid': common_mistakes,
            'task_type': task_type,
            'task_specific_examples': [],
            'retrieval_mode': 'template',
            'global_skill_top_k': effective_global_top_k,
        }

    def format_for_prompt(self, retrieved_memories: Dict[str, Any]) -> str:
        """
        Format retrieved skills into a string suitable for prompt injection.

        Args:
            retrieved_memories: Dict returned by :meth:`retrieve`.

        Returns:
            Formatted multi-section string to insert into the agent prompt.
        """
        sections = []
        task_type = retrieved_memories.get('task_type', 'unknown')
        mode = retrieved_memories.get('retrieval_mode', 'template')

        # General skills
        general_skills = retrieved_memories.get('general_skills', [])
        if general_skills:
            lines = ["### General Principles"]
            for skill in general_skills:
                title = skill.get('title', '')
                principle = skill.get('principle', '')
                lines.append(f"- **{title}**: {principle}")
            sections.append("\n".join(lines))

        # Task-specific skills
        task_skills = retrieved_memories.get('task_specific_skills', [])
        if task_skills:
            if mode == "embedding":
                section_title = "### Task-Relevant Skills"
            else:
                task_name = task_type.replace('_', ' ').title()
                section_title = f"### {task_name} Skills"
            lines = [section_title]
            for skill in task_skills:
                title = skill.get('title', '')
                principle = skill.get('principle', '')
                when = skill.get('when_to_apply', '')
                lines.append(f"- **{title}**: {principle}")
                if when:
                    lines.append(f"  _Apply when: {when}_")
            sections.append("\n".join(lines))

        # Common mistakes
        mistakes = retrieved_memories.get('mistakes_to_avoid', [])
        if mistakes:
            lines = ["### Mistakes to Avoid"]
            for mistake in mistakes:
                desc = mistake.get('description', '')
                fix = mistake.get('how_to_avoid', '')
                if desc:
                    lines.append(f"- **Don't**: {desc}")
                    if fix:
                        lines.append(f"  **Instead**: {fix}")
            sections.append("\n".join(lines))

        return "\n\n".join(sections) if sections else "No relevant skills found for this task."

    # ------------------------------------------------------------------ #
    # BaseMemory interface (not used in skills-only memory)               #
    # ------------------------------------------------------------------ #

    def reset(self, batch_size: int):
        pass

    def store(self, record: Dict[str, List[Any]]):
        pass

    def fetch(self, step: int):
        pass

    def __len__(self):
        return (
            len(self.skills.get('general_skills', [])) +
            sum(len(v) for v in self.skills.get('task_specific_skills', {}).values()) +
            len(self.skills.get('common_mistakes', []))
        )

    def __getitem__(self, idx: int):
        return self.skills

    # ------------------------------------------------------------------ #
    # Dynamic update methods                                               #
    # ------------------------------------------------------------------ #

    def add_skills(self, new_skills: List[Dict], category: str = 'general') -> int:
        """
        Add new skills to the bank and invalidate the embedding cache.

        Args:
            new_skills: List of skill dicts to add.
            category:   ``'general'`` or a task-type key (e.g. ``'clean'``).

        Returns:
            Number of skills actually added (duplicates are skipped).
        """
        added = 0
        existing_ids = self._get_all_skill_ids()

        for skill in new_skills:
            skill_id = skill.get('skill_id')
            if skill_id in existing_ids:
                print(f"[SkillsOnlyMemory] Skipping duplicate skill: {skill_id}")
                continue

            if category == 'general':
                self.skills.setdefault('general_skills', []).append(skill)
            else:
                self.skills.setdefault('task_specific_skills', {}).setdefault(category, []).append(skill)
            added += 1
            print(f"[SkillsOnlyMemory] Added skill: {skill_id} - {skill.get('title', 'N/A')}")

        if added > 0:
            # Invalidate embedding cache so it is recomputed on next retrieve
            self._skill_embeddings_cache = None

        return added

    def remove_skill(self, skill_id: str) -> bool:
        """Remove a skill by ID and invalidate the embedding cache."""
        removed = False

        original_len = len(self.skills.get('general_skills', []))
        self.skills['general_skills'] = [
            s for s in self.skills.get('general_skills', [])
            if s.get('skill_id') != skill_id
        ]
        if len(self.skills.get('general_skills', [])) < original_len:
            removed = True

        for task_type in self.skills.get('task_specific_skills', {}):
            original_len = len(self.skills['task_specific_skills'][task_type])
            self.skills['task_specific_skills'][task_type] = [
                s for s in self.skills['task_specific_skills'][task_type]
                if s.get('skill_id') != skill_id
            ]
            if len(self.skills['task_specific_skills'][task_type]) < original_len:
                removed = True

        if removed:
            self._skill_embeddings_cache = None
            print(f"[SkillsOnlyMemory] Removed skill: {skill_id}")
        return removed

    def save_skills(self, path: str):
        """Persist the current skill bank to a JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.skills, f, indent=2)
        print(f"[SkillsOnlyMemory] Saved {len(self)} skills to {path}")

    def _get_all_skill_ids(self) -> set:
        ids = set()
        for s in self.skills.get('general_skills', []):
            if s.get('skill_id'):
                ids.add(s['skill_id'])
        for task_skills in self.skills.get('task_specific_skills', {}).values():
            for s in task_skills:
                if s.get('skill_id'):
                    ids.add(s['skill_id'])
        return ids

    def get_skill_count(self) -> Dict[str, int]:
        return {
            'general': len(self.skills.get('general_skills', [])),
            'task_specific': sum(len(v) for v in self.skills.get('task_specific_skills', {}).values()),
            'common_mistakes': len(self.skills.get('common_mistakes', [])),
            'total': len(self),
        }
