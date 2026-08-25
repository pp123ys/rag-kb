from importlib import resources

import ragkb.prompts


def test_agent_prompt_contains_three_hard_rules():
    text = resources.files(ragkb.prompts).joinpath("agent_prompt.md").read_text(encoding="utf-8")
    assert "只依据检索内容回答" in text
    assert "知识库中没有找到相关内容" in text          # 精确拒绝语
    assert "[来源：<source 字段值>]" in text            # 引用格式对齐真实 source
    assert "{context}" in text                          # 注入点存在
    assert "source" in text                             # 字段契约提示存在
