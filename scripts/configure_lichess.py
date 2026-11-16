"""Generate a token-free bridge config with no external move sources."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


BRIDGE_COMMIT = "06f4f10b5a5a70549b9fc5f8f759a532723b5dab"


def configuration(bridge: Path, root: Path) -> dict:
    cfg = yaml.safe_load((bridge / "config.yml.default").read_text())
    cfg["token"] = "SET_LICHESS_BOT_TOKEN_IN_ENVIRONMENT"
    engine = cfg["engine"]
    engine.update(dir=str(root / "deploy"), name="run-engine.sh", working_dir=str(root),
                  protocol="uci", ponder=False, uci_options={})
    engine["polyglot"]["enabled"] = False
    engine["draw_or_resign"].update(resign_enabled=False, offer_draw_enabled=False,
                                    resign_for_egtb_minus_two=False, offer_draw_for_egtb_zero=False)
    for group in ("online_moves", "lichess_bot_tbs"):
        for options in engine[group].values():
            if isinstance(options, dict):
                options["enabled"] = False
    cfg["challenge"].update(concurrency=1, variants=["standard"], time_controls=["blitz", "rapid"],
                            modes=["casual"], min_base=180, max_base=1800, max_increment=30,
                            max_simultaneous_games_per_user=1)
    cfg["matchmaking"]["allow_matchmaking"] = False
    cfg["fake_think_time"] = False
    cfg["max_takebacks_accepted"] = 0
    cfg["quit_after_all_games_finish"] = True
    for greeting in cfg.get("greeting", {}):
        cfg["greeting"][greeting] = ""
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    bridge = args.bridge.resolve()
    output = bridge / "config.yml"
    if output.exists():
        raise SystemExit(f"{output} already exists; refusing to overwrite it")
    cfg = configuration(bridge, root)
    output.write_text(yaml.safe_dump(cfg, sort_keys=False))
    (root / "deploy/run-engine.sh").chmod(0o755)
    print(f"Wrote {output}; set LICHESS_BOT_TOKEN in your environment before starting.")


if __name__ == "__main__":
    main()
