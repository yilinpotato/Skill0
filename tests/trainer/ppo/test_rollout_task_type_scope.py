"""Exercise the pure labeling function without loading Ray/GPU dependencies."""
import ast
from pathlib import Path


def load_labeler():
    source = Path(__file__).resolve().parents[3] / 'agent_system/multi_turn_rollout/rollout_loop.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    method = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == 'infer_rollout_task_type')
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace[method.name]


def test_search_questions_do_not_become_shopping_categories():
    label = load_labeler()
    for question in ('Who delivered the Gettysburg Address?', 'Who invented the telephone?'):
        assert label({}, question, env_name='searchqa_live') == 'searchqa_live'


def test_native_domains_keep_their_own_labels():
    label = load_labeler()
    assert label({'extra.gamefile': 'pick_clean_then_place_in_recep/trial'}, 'shirt',
                 env_name='alfworld/AlfredTWEnv') == 'clean'
    assert label({}, 'buy a shirt', env_name='webshop') == 'apparel'
    assert label({}, 'shirt', env_name='alfworld/AlfredTWEnv') == 'other'
    assert label({}, 'heat a lamp', env_name='scienceworld') == 'scienceworld'
    assert label({}, 'shirt', env_name='unrecognized') == 'unknown'
