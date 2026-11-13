"""Stats page data: training logs, test results and spending are parsed; nothing secret is included."""
import json

from fastapi.testclient import TestClient

from proxmino import stats as ST
from proxmino.server import create_app


def test_stats_reads_training_logs_results_and_ledgers(engine, tmp_path):
    lab = tmp_path / "lab"
    (lab / "logs").mkdir(parents=True)
    (lab / "logs" / "proxmino-v1.log").write_text(
        "[08:06] training (1,380 conversations, one pass)\n"
        "\x1b[34m  iter   train_loss     tok/s     tokens\x1b[0m\n"
        "     1    val 2.672    4.45s\n    20    2.500 ▼        9      1.2k\n    40    1.900 ▼       12      2.4k\n"
        "   230    val 1.100    50.00s\n[heatwatch] cooling break (duty 75%): pausing 60s\n")
    (lab / "results").mkdir()
    (lab / "results" / "bench-proxmino-v1.json").write_text(json.dumps({"label": "proxmino:v1", "accuracy": 48.2, "wilson95": [42.4, 54.1], "n": 280}))
    (lab / "results" / "proxmino-v1-report.json").write_text(json.dumps({"promote": True, "gate": {}}))
    (lab / "distill").mkdir()
    (lab / "distill" / "v11_ledger.json").write_text(json.dumps({"model": "deepseek-flash", "tokens_in": 10, "tokens_out": 5, "dollars_estimate": 0.42, "budget": 2}))
    engine.cfg.data["model_lab"] = str(lab)
    run = ST.training_runs(lab)[0]
    assert run["steps_done"] == 40 and run["steps_total"] == 1380 and run["val_loss"] == [[1, 2.672], [230, 1.1]]
    assert run["train_loss"][-1] == [40, 1.9] and run["cooling_pauses"] == 1
    with TestClient(create_app(engine, run_scheduler=False), base_url="http://127.0.0.1") as c:
        s = c.get("/api/stats").json()
    assert s["training"][0]["name"] == "proxmino-v1"
    assert s["tests"]["benchmarks"][0]["accuracy"] == 48.2 and s["tests"]["reports"][0]["promote"] is True
    assert s["spending"][0]["dollars"] == 0.42
    assert s["app"]["chats"] >= 0 and s["machine"]["cores"] >= 1
    assert "key" not in json.dumps(s).lower() or all(k not in json.dumps(s) for k in ("sk-", "Bearer"))
