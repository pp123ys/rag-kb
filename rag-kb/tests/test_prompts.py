from importlib import resources

import ragkb.prompts


def test_agent_prompt_contains_three_hard_rules():
    text = resources.files(ragkb.prompts).joinpath("agent_prompt.md").read_text(encoding="utf-8")
    assert "只依据检索内容回答" in text
    assert "没有找到" in text
    assert "来源" in text
