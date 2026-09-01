import json

from scripts.create_independent_blind_review_template import build_template


def test_template_reads_only_packet_ids_and_leaves_human_judgment_blank(tmp_path):
    packet = tmp_path / "blind_review_packet.jsonl"
    packet.write_text(
        "".join(
            json.dumps({"review_id": f"r-{index}", "candidate": {}}) + "\n"
            for index in range(3)
        ),
        encoding="utf-8",
    )
    output = tmp_path / "independent_blind_review.json"

    template = build_template(packet, output, "human-reviewer")

    assert template["status"] == "pending"
    assert template["review_protocol"] == "packet-only-no-key-no-source-label"
    assert template["reviewed_records"] == 3
    assert template["attestation"] == ""
    assert [row["disposition"] for row in template["record_dispositions"]] == ["", "", ""]
    assert json.loads(output.read_text(encoding="utf-8"))["packet_sha256"] == template[
        "packet_sha256"
    ]
