"""CPU checks of the replay tool: contexts, JS, weights, and an end-to-end run with a tiny random model."""
import importlib.util
import json
import re
import sqlite3
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / "verl_mrtpo"), str(ROOT / "tool_calling" / "scripts")]

import replay  # noqa: E402
from fetch_data import typed_label  # noqa: E402
from mrtpo.hir import HIRQuestionBuilder  # noqa: E402
from mrtpo.js import js_divergence_from_logits  # noqa: E402
from mrtpo.sql.constructions import FORMAT_BLOCK, execution_sql_block, p0_messages  # noqa: E402
from mrtpo.sql.runtime import execute, extract_sql  # noqa: E402

EXAMPLES = HERE / "examples"


def load_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rlla_replay = load_path("rlla_replay", ROOT / "tool_calling/vendor/verl/verl/utils/rlla_replay.py")
tc_mrtpo = load_path("tc_mrtpo", ROOT / "tool_calling/vendor/verl/verl/trainer/ppo/mrtpo.py")


def example(name):
    return json.loads((EXAMPLES / f"{name}.json").read_text())


# ---------------------------------------------------------------------------------------------- contexts

def test_blocks_appended_in_order_and_left_out_one_at_a_time():
    messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    rewards = [{"name": "a", "block": "A"}, {"name": "b", "block": ""}, {"name": "c", "block": "C"}]
    full, minus, texts = replay.build_contexts(messages, rewards)
    assert texts == ["A", "", "C"]
    assert full == [{"role": "system", "content": "S"}, {"role": "user", "content": "U\n\nA\n\nC"}]
    assert minus[0][1]["content"] == "U\n\nC"
    assert minus[1] is None
    assert minus[2][1]["content"] == "U\n\nA"
    assert all(m[0] == messages[0] for m in minus if m)
    assert messages[1]["content"] == "U"
    # a per-response block replaces the example's block; empty means not replayed
    full, minus, _ = replay.build_contexts(messages, rewards, {"b": "B", "c": ""})
    assert full[1]["content"] == "U\n\nA\n\nB" and minus[1][1]["content"] == "U\n\nA" and minus[2] is None
    with pytest.raises(ValueError):
        replay.build_contexts(messages, rewards, {"x": "X"})


def test_removed_constraint_contexts():
    messages = [{"role": "user", "content": "Q c1 c2"}]
    rewards = [{"name": "c1", "minus_prompt": "Q c2"}, {"name": "c2", "minus_prompt": [{"role": "user", "content": "Q c1"}]}]
    full, minus, _ = replay.build_contexts(messages, rewards)
    assert full == messages
    assert minus == [[{"role": "user", "content": "Q c2"}], [{"role": "user", "content": "Q c1"}]]
    with pytest.raises(ValueError):
        replay.build_contexts(messages, [{"name": "x", "block": "B", "minus_prompt": "Q"}])


def test_text_to_sql_example_matches_release_builders(tmp_path):
    ex = example("text_to_sql")
    content = ex["messages"][0]["content"]
    question = content.split("Question:\n", 1)[1].split("\n\nInstructions:", 1)[0]
    schema = content.split("Database Schema:\n", 1)[1].split("\nThis schema describes", 1)[0]
    assert p0_messages(question, schema) == ex["messages"]
    blocks = {r["name"]: r["block"] for r in ex["rewards"]}
    assert [r["name"] for r in ex["rewards"]] == ["format", "executable", "correct"]
    assert blocks["format"] == FORMAT_BLOCK
    gold = re.fullmatch(r"The reference query for this question is: `(.*)`\.", blocks["correct"]).group(1)
    assert blocks["correct"] == execution_sql_block(gold)
    # the executable block is the SQLite error of the response, as the reward writes it
    db = tmp_path / "head.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE head (head_ID number, name text, born_state text, age number)")
    first, second = ex["responses"]
    _, error = execute(extract_sql(first), db, timeout=30.0)
    assert blocks["executable"] == f"When this query was executed on this database, SQLite raised: `{error}`."
    assert execute(extract_sql(second["text"]), db, timeout=30.0)[1] is None
    assert second["blocks"] == {"executable": ""}
    # P_all is P_0 followed by every non-empty block, as in mrtpo/criteria.py
    full, minus, _ = replay.build_contexts(ex["messages"], ex["rewards"])
    assert full[0]["content"] == content + "\n\n" + "\n\n".join(blocks.values())
    assert minus[1][0]["content"] == content + "\n\n" + blocks["format"] + "\n\n" + blocks["correct"]


def test_tool_calling_example_matches_release_builders():
    ex = example("tool_calling")
    system, user = (m["content"] for m in ex["messages"])
    label = ex["responses"][0]
    row = {"instruction": system, "input": user, "output": label, "teacher_output": typed_label(label, system)}
    assert [r["name"] for r in ex["rewards"]] == ["correctness", "format"]
    full, minus, _ = replay.build_contexts(ex["messages"], ex["rewards"])
    assert full == rlla_replay.context_messages(row, "full")[0]
    assert minus[0] == rlla_replay.context_messages(row, "format")[0]       # correctness: P_FC vs P_F
    assert minus[1] == rlla_replay.context_messages(row, "correctness")[0]  # format: P_FC vs P_C
    assert "Parameter `number_of_questions`: 10." in ex["rewards"][0]["block"]


def test_instruction_following_example_matches_release_builders():
    ex = example("instruction_following")
    criteria = [r["name"] for r in ex["rewards"]]
    question = ex["messages"][0]["content"].split("\n\n", 1)[0]
    builder = HIRQuestionBuilder()
    assert builder._concat_question(question, criteria, "type4") == ex["messages"][0]["content"]
    for j, reward in enumerate(ex["rewards"]):
        assert reward["minus_prompt"] == builder._concat_question(question, criteria[:j] + criteria[j + 1:], "type4")


# ---------------------------------------------------------------------------------------------- JS and weights

def test_js_equals_both_training_stacks():
    torch.manual_seed(0)
    left, right = torch.randn(300, 1000) * 3, torch.randn(300, 1000) * 3
    ours = replay.chunked_distribution_js(left, right, chunk_size=128)
    assert torch.equal(ours, js_divergence_from_logits(left, right))
    assert torch.equal(ours, tc_mrtpo.chunked_js(left, right, 128))
    tempered = replay.chunked_distribution_js(left, right, chunk_size=128, temperature=1.1)
    assert torch.equal(tempered, js_divergence_from_logits(left / 1.1, right / 1.1))


def test_weights_have_mean_one_and_the_mrtpo_form():
    torch.manual_seed(0)
    js = torch.rand(57) * 0.69
    for beta in (0.25, 0.5, 0.75):
        q, w = replay.token_weights(js, beta)
        assert torch.allclose(q, torch.sqrt(js / replay.LOG_TWO))
        assert torch.allclose(w, (1 - beta) + beta * q / q.mean(), atol=1e-6)
        assert abs(w.mean().item() - 1.0) < 1e-6
    # identical to the tool-calling stack's weights
    _, w = replay.token_weights(js, 0.5)
    tc = tc_mrtpo.token_weights(js[None], torch.ones(1, 57), torch.ones(1, dtype=torch.bool), 0.5)[0]
    assert torch.allclose(w, tc, atol=1e-7)
    assert torch.equal(replay.token_weights(js, 0.0)[1], torch.ones(57))
    assert torch.equal(replay.token_weights(torch.zeros(9), 0.5)[1], torch.ones(9))


# ---------------------------------------------------------------------------------------------- end to end

def cached_tokenizer():
    from transformers import AutoTokenizer

    for name in ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen3-4B-Instruct-2507"):
        try:
            return AutoTokenizer.from_pretrained(name, local_files_only=True)
        except Exception:  # noqa: BLE001 - not in the local cache
            continue
    pytest.skip("no Qwen tokenizer in the local Hugging Face cache")


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    """A randomly initialised two-layer Qwen2 with a real Qwen tokenizer; no weights are downloaded."""
    from transformers import AutoModelForCausalLM, Qwen2Config

    tokenizer = cached_tokenizer()
    torch.manual_seed(0)
    config = Qwen2Config(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
                         tie_word_embeddings=True, eos_token_id=tokenizer.eos_token_id)
    model = AutoModelForCausalLM.from_config(config)
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    path = tmp_path_factory.mktemp("tiny-qwen2")
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path


def test_same_response_ids_under_every_context(tiny_model, monkeypatch):
    replayer = replay.Replayer(str(tiny_model), device="cpu", dtype="float32")
    ex = example("text_to_sql")
    calls = []
    original = replayer.response_logits

    def spy(prompt_ids, response_ids):
        calls.append((list(prompt_ids), list(response_ids)))
        return original(prompt_ids, response_ids)

    monkeypatch.setattr(replayer, "response_logits", spy)
    record = replay.run(ex, replayer)
    first = record["responses"][0]
    ids = replayer.response_ids(ex["responses"][0])
    assert first["response_ids"] == ids and ids[-1] == replayer.tokenizer.eos_token_id
    full, minus, _ = replay.build_contexts(ex["messages"], ex["rewards"])
    assert calls[:4] == [(replayer.prompt_ids(m), ids) for m in [full, *minus]]
    # second response: the executable block is empty, so 1 + 2 forward passes and uniform weights
    assert len(calls) == 4 + 3
    row = record["responses"][1]["rewards"][1]
    assert not row["replayed"] and row["w"] == [1.0] * len(record["responses"][1]["tokens"])


def test_replay_js_matches_an_independent_forward(tiny_model):
    replayer = replay.Replayer(str(tiny_model), device="cpu", dtype="float32")
    ex = example("tool_calling")
    record = replay.run(ex, replayer)
    response = record["responses"][0]
    ids = response["response_ids"]
    full, minus, _ = replay.build_contexts(ex["messages"], ex["rewards"])

    def logits(messages):
        prompt = replayer.prompt_ids(messages)
        out = replayer.model(torch.tensor([prompt + ids])).logits[0]
        return out[len(prompt) - 1:len(prompt) - 1 + len(ids)]

    for row, messages in zip(response["rewards"], minus):
        expected = js_divergence_from_logits(logits(full), logits(messages))
        assert torch.allclose(torch.tensor(row["js"]), expected, atol=1e-6)
        assert len(row["q"]) == len(row["w"]) == len(response["tokens"]) == len(ids)
        assert abs(sum(row["w"]) / len(row["w"]) - 1.0) < 1e-5


@pytest.mark.parametrize("name", ["text_to_sql", "tool_calling", "instruction_following"])
def test_cli_writes_page_and_json(tiny_model, tmp_path, name):
    replay.main(["--model", str(tiny_model), "--input", str(EXAMPLES / f"{name}.json"), "--out", str(tmp_path),
                 "--device", "cpu", "--dtype", "float32"])
    record = json.loads((tmp_path / f"{name}.json").read_text())
    page = (tmp_path / f"{name}.html").read_text()
    assert page.startswith("<!doctype html>") and "http" not in page.split("<body>", 1)[0]
    for response in record["responses"]:
        assert len(response["rewards"]) == len(example(name)["rewards"])
        assert page.count("title=\"token ") >= len(response["tokens"])


def test_cli_samples_responses_and_png(tiny_model, tmp_path):
    pytest.importorskip("matplotlib")
    replay.main(["--model", str(tiny_model), "--input", str(EXAMPLES / "tool_calling.json"), "--out", str(tmp_path),
                 "--device", "cpu", "--dtype", "float32", "--sample", "2", "--max-new-tokens", "12", "--png"])
    record = json.loads((tmp_path / "tool_calling.json").read_text())
    assert [r["source"] for r in record["responses"]] == ["sampled", "sampled"]
    assert all(1 <= len(r["response_ids"]) <= 12 for r in record["responses"])
    assert (tmp_path / "tool_calling_response1.png").stat().st_size > 0
    assert (tmp_path / "tool_calling_response2.png").stat().st_size > 0
