import argparse, hashlib, importlib.metadata, json, os, stat, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT, REPORTS = Path(__file__).resolve().parents[1], Path(__file__).resolve().parents[1] / "reports/validation_2026-07-30"
EXPERIMENT, PRODUCT = "EXP-SMC-3.0.2-016-DISPLACEMENT-PRODUCTION-IDENTITY-CUSTODY", "3.0.2"
PREREG_SHA = "6a092dcfbce1845bc4b0eff946a69fc7315bbbcfc7827642a2dceba7c3aaf301"
ATTEMPT_ROOT = "artifacts/v3_exp016_displacement_gate2_physical_001"
PATHS = {"preregistration": "reports/validation_2026-07-30/v3_exp016_displacement_custody_scope_reduction_preregistration.md",
         "ledger": "configs/experiments/ledger.jsonl", "wrapper": "scripts/run_exp016_displacement_custody_gate.py",
         "controller": "scripts/run_exp016_displacement_custody.py", "control": "configs/experiments/EXP-SMC-3.0.2-016-DISPLACEMENT-PRODUCTION-IDENTITY-CUSTODY.json", "materializer": "scripts/materialize_ohlcv_preholdout.py",
         "claim": ATTEMPT_ROOT + ".claim.json", "attempt_root": ATTEMPT_ROOT, "attempt": ATTEMPT_ROOT + "/ATTEMPT.json",
         "core_stdout": ATTEMPT_ROOT + "/CORE_STDOUT.log", "core_stderr": ATTEMPT_ROOT + "/CORE_STDERR.log",
         "core_result": ATTEMPT_ROOT + "/CORE_RESULT.json", "terminal": ATTEMPT_ROOT + "/TERMINAL.json",
         "output": "data/processed/nq_1m_previous_session_front_v3_exp016_discovery_2017_2021.parquet",
         "source": "data/processed/nq_1m_previous_session_front_v2_3_pre_holdout_2017_20260331.parquet", "explicit_manifest": "data/processed/nq_1m_previous_session_front_v2_3_pre_holdout_2017_20260331.parquet.manifest.json", "derived_optional_manifest": "data/processed/nq_1m_previous_session_front_v2_3_pre_holdout_2017_20260331.manifest.json", "real_semantic_root": "outputs/v3_exp016_displacement/discovery-2017-2021-attempt001"}
PATHS.update(output_manifest=PATHS["output"] + ".manifest.json", output_temporary=PATHS["output"] + ".tmp")
GATE_EVIDENCE = ("artifacts/v3_exp016_displacement_gate1_synthetic_002/TERMINAL.json", "reports/validation_2026-07-30/v3_exp016_gate2_physical_pretest.md", "reports/validation_2026-07-30/v3_exp016_gate2_causal_data_review.md", "reports/validation_2026-07-30/v3_exp016_gate2_governance_review.md", "reports/validation_2026-07-30/v3_exp016_gate2_consensus.md")
TRUE_AUTHORITY = {"gate2_authorized", "physical_schema_read_authorized", "ohlcv_authorized", "custody_materialization_authorized"}
FALSE_AUTHORITY = {"additional_attempt_authorized", "gate3_authorized", "mbo_authorized", "sealed_data_authorized", "future_reveal_authorized", "action_authorized", "economic_evaluation_authorized", "production_release_authorized", "real_semantic_replay_authorized"}
CANDIDATES, CORE_ARGV = tuple(PATHS[key] for key in ("wrapper", "controller", "control", "materializer")), [sys.executable, "-I", "-B", PATHS["controller"]]
CLEAN_ENV = {"LC_ALL": "C", "PATH": "/usr/bin:/bin"}
COMMAND_KEYS = {"format_version", "experiment_id", "product_version", "preregistration_sha256", "gate", "attempt_id", "cwd", "interpreter", "controller_argv", "paths", "candidate_sha256", "candidate_bundle_sha256", "gate_consensus_sha256", "gate_evidence_sha256", "interpreter_sha256", "package_versions", "authorization_ledger_head", "authority_status"}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode() + b"\n"


def parent_chain(relative):
    current = ROOT
    for part in Path(relative).parts[:-1]:
        current /= part
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            break
        require(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode), f"unsafe ancestor: {relative}")


def leaf(relative, limit=2_000_000, digest=False):
    parent_chain(relative)
    fd = os.open(ROOT / relative, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(fd)
        require(stat.S_ISREG(info.st_mode) and info.st_size <= limit, f"invalid leaf: {relative}")
        payload = hashlib.file_digest(stream, "sha256").hexdigest() if digest else stream.read(limit + 1)
        require(digest or len(payload) == info.st_size, f"unstable leaf: {relative}")
        return payload


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    os.fsync(fd)
    os.close(fd)


def create_json(relative, value):
    payload, path = canonical(value), ROOT / relative
    parent_chain(relative)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    fsync_dir(path.parent)
    return hashlib.sha256(payload).hexdigest()


def parse_json(relative, limit=2_000_000):
    payload = leaf(relative, limit)
    value = json.loads(payload)
    require(canonical(value) == payload, f"noncanonical JSON: {relative}")
    return value, payload


def preflight(command_path, command_sha):
    absolute = Path(os.path.abspath(Path(command_path) if Path(command_path).is_absolute() else ROOT / command_path))
    require(absolute.resolve(strict=True) == absolute and absolute.is_relative_to(REPORTS),
            "command manifest must be a real report-directory leaf")
    relative = absolute.relative_to(ROOT).as_posix()
    manifest, payload = parse_json(relative, 65_536)
    require(hashlib.sha256(payload).hexdigest() == command_sha, "command manifest SHA differs")
    require(set(manifest) == COMMAND_KEYS and tuple(manifest[key] for key in ("format_version", "experiment_id", "product_version", "preregistration_sha256", "gate", "attempt_id", "cwd", "interpreter", "controller_argv", "paths", "authority_status"))
            == (1, EXPERIMENT, PRODUCT, PREREG_SHA, 2, "physical_001", str(ROOT), sys.executable, CORE_ARGV, dict(PATHS, command_manifest=relative), "gate2_physical_attempt_authorized"), "command identity differs")
    require(leaf(PATHS["preregistration"], 65_536, True) == PREREG_SHA, "preregistration identity differs")
    evidence = manifest["gate_evidence_sha256"]
    require(set(evidence) == set(GATE_EVIDENCE) and all(leaf(path, 1_000_000, True) == evidence[path] for path in GATE_EVIDENCE), "Gate evidence differs")
    require(manifest["gate_consensus_sha256"] == evidence[GATE_EVIDENCE[-1]], "Gate-2 consensus differs")
    require(manifest["interpreter_sha256"] == leaf(sys.executable, 100_000_000, True), "interpreter identity differs")
    require(manifest["package_versions"] == {name: importlib.metadata.version(name) for name in ("pyarrow", "pandas")}, "package versions differ")
    candidate = manifest["candidate_sha256"]
    require(set(candidate) == set(CANDIDATES) and all(leaf(path, 2_000_000, True) == candidate[path] for path in CANDIDATES)
            and hashlib.sha256(canonical(candidate)).hexdigest() == manifest["candidate_bundle_sha256"], "candidate bundle differs")
    ledger = leaf(PATHS["ledger"])
    require(hashlib.sha256(ledger).hexdigest() == manifest["authorization_ledger_head"] and ledger.endswith(b"\n"),
            "authorization ledger head differs")
    last_line = ledger.splitlines()[-1]
    last = json.loads(last_line)
    authority, bindings = last.get("authority", {}), last.get("bindings", {})
    require(canonical(last).rstrip(b"\n") == last_line and last.get("experiment_id") == EXPERIMENT and last.get("product_version") == PRODUCT
            and last.get("status") == manifest["authority_status"] and set(authority) == TRUE_AUTHORITY | FALSE_AUTHORITY
            and all(authority[key] is True for key in TRUE_AUTHORITY) and all(authority[key] is False for key in FALSE_AUTHORITY)
            and bindings.get("preregistration_sha256") == PREREG_SHA and bindings.get("gate_evidence_sha256") == evidence
            and bindings.get("gate_consensus_sha256") == manifest["gate_consensus_sha256"]
            and bindings.get("candidate_bundle_sha256") == manifest["candidate_bundle_sha256"]
            and bindings.get("interpreter_sha256") == manifest["interpreter_sha256"]
            and bindings.get("package_versions") == manifest["package_versions"], "Gate-2 authority record differs")
    for target in (PATHS["claim"], PATHS["attempt_root"], PATHS["output"], PATHS["output_manifest"], PATHS["output_temporary"], PATHS["derived_optional_manifest"], PATHS["real_semantic_root"]):
        parent_chain(target)
        require(not (ROOT / target).exists() and not (ROOT / target).is_symlink(), f"target exists: {target}")
    return manifest, relative, ledger


def append_setup(manifest, command_path, command_sha, ledger, claim_sha):
    record = {"format_version": 1, "experiment_id": EXPERIMENT, "product_version": PRODUCT, "status": "gate2_physical_attempt_001_setup_started_authority_consumed",
              "recorded_at": datetime.now(timezone.utc).isoformat(), "parent_ledger_sha256": manifest["authorization_ledger_head"],
              "bindings": {"attempt_id": "physical_001", "claim_path": PATHS["claim"], "claim_sha256": claim_sha, "command_manifest_path": command_path,
                           "command_manifest_sha256": command_sha, "candidate_bundle_sha256": manifest["candidate_bundle_sha256"]},
              "authority": {"additional_attempt_authorized": False, "gate2_authorized": False, "physical_schema_read_authorized": False, "ohlcv_authorized": False, "custody_materialization_authorized": False}}
    payload = canonical(record)
    require(leaf(PATHS["ledger"]) == ledger, "ledger changed before setup append")
    fd = os.open(ROOT / PATHS["ledger"], os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW)
    with os.fdopen(fd, "ab") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    fsync_dir((ROOT / PATHS["ledger"]).parent)
    return hashlib.sha256(ledger + payload).hexdigest()


def validate_core():
    result, payload = parse_json(PATHS["core_result"], 65_536)
    require((result.get("status"), result.get("classification")) == ("complete_exact_projection", "CUSTODY_CORE_EXACT_PROJECTION_VERIFIED"),
            "core did not verify")
    require((result.get("experiment_id"), result.get("product_version"), result.get("preregistration_sha256"), result.get("gate"), result.get("attempt_id")) == (EXPERIMENT, PRODUCT, PREREG_SHA, "gate2_physical_custody", "physical_001"), "core identity differs")
    source, output = result.get("audits", {}).get("source"), result.get("audits", {}).get("output")
    equal = ("field_order", "arrow_schema_hex", "rows", "first_timestamp_ns", "last_timestamp_ns", "typed_row_commitment")
    equality, denials = result.get("audits", {}).get("equality"), result.get("authority_denials")
    require(isinstance(source, dict) and isinstance(output, dict) and all(source.get(key) is not None and source.get(key) == output.get(key) for key in equal)
            and isinstance(equality, dict) and set(equality) == set(equal) and all(value is True for value in equality.values())
            and result.get("fixed_contract", {}).get("ordered_fields") == ["open", "high", "low", "close", "volume", "symbol", "instrument_id", "ts"]
            and (result.get("fixed_contract", {}).get("batch_rows"), result.get("fixed_contract", {}).get("row_group_rows")) == (4096, 4096)
            and result.get("child", {}).get("invocations") == 1 and result.get("child", {}).get("returncode") == 0
            and result.get("resources", {}).get("full_frame_retained") is False and tuple(result.get(key) for key in ("mbo_accesses", "current_session_volume_uses", "action_or_future_fields")) == (0, 0, 0), "core execution/schema evidence differs")
    denial_keys = {"semantic_runner_authorized", "mbo_authorized", "sealed_data_authorized", "future_or_action_authorized", "economic_evaluation_authorized", "gate3_authorized", "production_release_authorized"}
    require(isinstance(denials, dict) and set(denials) == denial_keys and all(value is False for value in denials.values()), "core authority denial differs")
    output_identity, manifest_identity = leaf(PATHS["output"], 64_000_000, True), leaf(PATHS["output_manifest"], 1_000_000, True)
    require((output_identity, manifest_identity) == (result.get("identities", {}).get("output_sha256"), result.get("identities", {}).get("output_manifest_sha256")), "published output identity differs")
    return hashlib.sha256(payload).hexdigest(), output_identity, manifest_identity


def execute(context):
    rc, reason, attempt_sha, core_sha, output_sha, output_manifest_sha, descriptors = None, None, None, None, None, None, []
    parent_chain(ATTEMPT_ROOT)
    (ROOT / ATTEMPT_ROOT).mkdir()
    try:
        fsync_dir((ROOT / ATTEMPT_ROOT).parent)
        try:
            attempt_sha = create_json(PATHS["attempt"], dict(context, status="STARTED"))
            descriptors.append(os.open(ROOT / PATHS["core_stdout"], os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444))
            descriptors.append(os.open(ROOT / PATHS["core_stderr"], os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444))
            fsync_dir(ROOT / ATTEMPT_ROOT)
            rc = subprocess.run(CORE_ARGV, cwd=ROOT, env=CLEAN_ENV, stdout=descriptors[0], stderr=descriptors[1], check=False).returncode
        finally:
            for descriptor in descriptors:
                os.fsync(descriptor)
                os.close(descriptor)
        require(rc == 0, f"core returned {rc}")
        core_sha, output_sha, output_manifest_sha = validate_core()
    except BaseException as exc:
        reason = f"{type(exc).__name__}: {exc}"[:512]
    terminal = dict(context, attempt_sha256=attempt_sha, core_returncode=rc, core_result_sha256=core_sha, output_sha256=output_sha, output_manifest_sha256=output_manifest_sha, reason=reason,
                    status="CUSTODY_GO_GATE3_NOT_AUTHORIZED" if reason is None else "NOGO", classification="CUSTODY_GO_GATE3_NOT_AUTHORIZED" if reason is None else "NOGO")
    create_json(PATHS["terminal"], terminal)
    return 0 if reason is None else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--command-manifest", required=True)
    parser.add_argument("--command-manifest-sha256", required=True)
    args = parser.parse_args()
    manifest, command_path, ledger = preflight(args.command_manifest, args.command_manifest_sha256)
    base = {"format_version": 1, "experiment_id": EXPERIMENT, "product_version": PRODUCT, "gate": 2, "attempt_id": "physical_001",
            "command_manifest_path": command_path, "command_manifest_sha256": args.command_manifest_sha256,
            "authorization_ledger_head": manifest["authorization_ledger_head"],
            "candidate_bundle_sha256": manifest["candidate_bundle_sha256"]}
    claim_sha = create_json(PATHS["claim"], dict(base, status="CLAIMED"))
    setup_head = append_setup(manifest, command_path, args.command_manifest_sha256, ledger, claim_sha)
    context = dict(base, setup_start_ledger_head=setup_head, claim_path=PATHS["claim"], claim_sha256=claim_sha, controller_argv=CORE_ARGV)
    raise SystemExit(execute(context))


if __name__ == "__main__":
    require(sys.flags.isolated == 1 and sys.dont_write_bytecode, "wrapper requires -I -B")
    main()
