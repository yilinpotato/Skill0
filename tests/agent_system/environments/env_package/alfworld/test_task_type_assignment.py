from agent_system.environments.env_package.alfworld.envs import (
    normalize_task_types,
    task_type_for_worker,
)


def test_normalize_task_types_preserves_single_string():
    assert normalize_task_types("pick_and_place") == ["pick_and_place"]


def test_task_type_assignment_keeps_each_grpo_group_homogeneous():
    task_types = normalize_task_types(["clean", "heat", "cool"])
    assigned = [task_type_for_worker(task_types, index, group_n=6) for index in range(24)]

    assert assigned[:6] == ["clean"] * 6
    assert assigned[6:12] == ["heat"] * 6
    assert assigned[12:18] == ["cool"] * 6
    assert assigned[18:24] == ["clean"] * 6
