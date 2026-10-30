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


def test_finish_with_code_regenerates_and_flags():
    parsed = parse_multi_turn_decision("FINISH? Not yet:\n```python\nclose_gripper()\n```")
    assert (parsed.decision, parsed.source, parsed.flags) == ("regenerate", "code_block", ["finish_with_code"])


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
