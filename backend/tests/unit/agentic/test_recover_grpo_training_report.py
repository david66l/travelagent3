from scripts.recover_grpo_training_report import _final_train_metrics


def test_recovery_reads_the_explicit_training_log(tmp_path):
    log_path = tmp_path / "explicit-training.log"
    log_path.write_text(
        "progress\n{'train_runtime': 12.5, 'train_loss': -0.01, 'epoch': 1}\n",
        encoding="utf-8",
    )

    assert _final_train_metrics(log_path) == {
        "train_runtime": 12.5,
        "train_loss": -0.01,
        "epoch": 1,
    }
