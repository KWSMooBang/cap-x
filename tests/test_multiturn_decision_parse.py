"""Multi-turn decision parsing: replies from a real run and edge cases."""

import ast
import json
from pathlib import Path

import pytest

from capx.utils.launch_utils import _extract_code, _parse_multi_turn_decision, parse_multi_turn_decision

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "multiturn_decision_replies.json").read_text())


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=[c["id"] for c in FIXTURE["cases"]])
def test_run_replies(case):
    parsed = parse_multi_turn_decision(case["reply"])
    assert (parsed.decision, parsed.source) == (case["decision"], case["source"])
    assert parsed.legacy_decision == "finish"  # the original parser finished all of these
    if parsed.decision == "regenerate":
        blocks = _extract_code(parsed.new_code)
        assert blocks and all(b.strip() for b in blocks)
        if "unclosed_fence" not in parsed.flags:  # a reply cut off mid-code cannot parse
            for b in blocks:
                ast.parse(b)


def test_fixture_mix():
    sources = [c["source"] for c in FIXTURE["cases"]]
    assert sources.count("finish_keyword") == 3
    assert sources.count("code_block") + sources.count("tool_call") == 5
    assert sources.count("empty") == 1


def test_keyword_regenerate_unchanged():
    reply = "The grasp missed.\nREGENERATE\n```python\nopen_gripper()\n```"
    parsed = parse_multi_turn_decision(reply)
    assert (parsed.decision, parsed.source, parsed.legacy_decision) == ("regenerate", "keyword", "regenerate")
    assert _extract_code(parsed.new_code) == ["open_gripper()"]
    assert _parse_multi_turn_decision(reply) == ("regenerate", reply.split("REGENERATE")[1].strip())


def test_finish_before_code_regenerates_and_flags():
    parsed = parse_multi_turn_decision("FINISH? Not yet:\n```python\nclose_gripper()\n```")
    assert (parsed.decision, parsed.source, parsed.flags) == ("regenerate", "code_block", ["finish_with_code"])


def test_quoted_code_then_finish_does_not_rerun():
    reply = "This code already ran and placed the bowl:\n```python\nopen_gripper()\ngoto_pose(p, q)\n```\nThe bowl is on the plate.\nFINISH"
    parsed = parse_multi_turn_decision(reply)
    assert (parsed.decision, parsed.source, parsed.flags) == ("finish", "finish_keyword", ["finish_after_code"])


def test_keyword_with_tool_call_is_normalized():
    reply = 'REGENERATE<|open|>tools<|sep|><|open|>call tool="code" index="1"<|sep|><|open|>argument key="code" type="string"<|sep|>open_gripper()\n<|close|>argument<|sep|><|close|>call<|sep|>'
    parsed = parse_multi_turn_decision(reply)
    assert (parsed.decision, parsed.source) == ("regenerate", "keyword")
    assert _extract_code(parsed.new_code) == ["open_gripper()"]


def test_keyword_with_unclosed_fence_flags():
    parsed = parse_multi_turn_decision("REGENERATE\n```python\nopen_gripper()\nx = (")
    assert parsed.flags == ["unclosed_fence"] and _extract_code(parsed.new_code)[0].startswith("open_gripper()")


def test_bare_fence_is_code():
    parsed = parse_multi_turn_decision("Try again:\n```\nx = 1\n```")
    assert parsed.decision == "regenerate"
    assert _extract_code(parsed.new_code) == ["x = 1"]


@pytest.mark.parametrize("reply,source", [(None, "empty"), ("   \n", "empty"), ("FINISH", "finish_keyword"), ("Looks done.", "fallback")])
def test_finish_paths(reply, source):
    parsed = parse_multi_turn_decision(reply)
    assert (parsed.decision, parsed.source) == ("finish", source)


def test_unclosed_fence_is_code():
    parsed = parse_multi_turn_decision("Next step:\n```python\nopen_gripper()\ngoto_pose(np.array([0.5,")
    assert (parsed.decision, parsed.source, parsed.flags) == ("regenerate", "code_block", ["unclosed_fence"])
    assert _extract_code(parsed.new_code)[0].startswith("open_gripper()")


def test_finish_between_fence_and_tool_call_regenerates():
    reply = ('```python\nopen_gripper()\n```\nFINISH? no, one more:\n'
             '<|open|>argument key="code" type="string"<|sep|>close_gripper()\n<|close|>argument')
    parsed = parse_multi_turn_decision(reply)
    assert parsed.decision == "regenerate" and "finish_with_code" in parsed.flags


# Unfenced code (bg_est2 10x5 misparsed_finish t01 tr5 / t02 tr5 / t03 tr1): declaration comments, then a program.
RAW_REPLY = """# Subtask: place(bottle_A, plate_A)
# Effect: bottle_A rests on plate_A
import numpy as np

def lift(pose, dz=0.1):
    p = np.array(pose[0]) + np.array([0.0, 0.0, dz])
    return p, pose[1]

grasp = get_grasp("bottle_A")
goto_pose(*lift(grasp))
close_gripper()
"""


def test_raw_code_regenerates():
    parsed = parse_multi_turn_decision(RAW_REPLY)
    assert (parsed.decision, parsed.source, parsed.flags) == ("regenerate", "raw_code", [])
    assert parsed.legacy_decision == "finish"
    assert _extract_code(parsed.new_code) == [RAW_REPLY.strip()]
    assert _parse_multi_turn_decision(RAW_REPLY) == ("regenerate", parsed.new_code)


def test_raw_code_after_prose_regenerates_and_flags():
    parsed = parse_multi_turn_decision("The bottle is not on the plate yet. Next:\n\n" + RAW_REPLY)
    assert (parsed.decision, parsed.source, parsed.flags) == ("regenerate", "raw_code", ["prose_before_code"])
    assert _extract_code(parsed.new_code) == [RAW_REPLY.strip()]


def test_markdown_heading_before_raw_code():
    parsed = parse_multi_turn_decision("# Analysis\nThe grasp slipped.\n\n" + RAW_REPLY)
    assert (parsed.source, parsed.flags) == ("raw_code", ["prose_before_code"])
    assert _extract_code(parsed.new_code) == [RAW_REPLY.strip()]


@pytest.mark.parametrize("reply", [
    "# Subtask: place(bottle_A, plate_A)\n# Effect: done\n",  # comments only
    "Looks done.",
    "Done",  # parses as a bare name
    '"""The bottle is on the plate."""',
    "The task is complete. The last call was:\nplace(bottle_A, plate_A)",  # quoted call after prose
    RAW_REPLY + "\nThis should finish the task.",  # trailing prose
    "pass",
])
def test_not_raw_code_falls_back(reply):
    parsed = parse_multi_turn_decision(reply)
    assert (parsed.decision, parsed.source) == ("finish", "fallback")


@pytest.mark.parametrize("reply", [RAW_REPLY + "\nFINISH", RAW_REPLY + "# FINISH\n", "FINISH\n" + RAW_REPLY])
def test_raw_code_with_finish_word_finishes(reply):
    parsed = parse_multi_turn_decision(reply)
    assert (parsed.decision, parsed.source, parsed.flags) == ("finish", "finish_keyword", [])


def test_raw_code_does_not_change_fenced_paths():
    reply = "Plan:\nx = 1\n```python\nopen_gripper()\n```"
    parsed = parse_multi_turn_decision(reply)
    assert (parsed.source, _extract_code(parsed.new_code)) == ("code_block", ["open_gripper()"])
