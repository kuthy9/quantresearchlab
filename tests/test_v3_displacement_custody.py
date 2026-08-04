from datetime import datetime, timedelta
import hashlib, importlib.util, json, os
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = ROOT / "scripts/run_exp016_displacement_custody.py"
GATE_PATH = ROOT / "scripts/run_exp016_displacement_custody_gate.py"
KNOWN_COMMITMENT = "b3fb12deae05c97d0799d1f3c2ebcd200ac2813628b74dbdf4affbc35c9efb42"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def custody(tmp_path, monkeypatch):
    core = _load("exp016_core_test", CORE_PATH)
    gate = _load("exp016_gate_test", GATE_PATH)
    monkeypatch.setattr(core, "ROOT", tmp_path)
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    monkeypatch.setattr(gate, "REPORTS", tmp_path / "reports/validation_2026-07-30")
    for relative in ("artifacts", "data/processed", "scripts", "configs/experiments", "reports/validation_2026-07-30"):
        (tmp_path / relative).mkdir(parents=True, exist_ok=True)
    return core, gate, tmp_path


def _schema(timestamp=None, metadata=None, replace=None, omit=(), append=()):
    types = {"open": pa.float64(), "high": pa.float64(), "low": pa.float64(), "close": pa.float64(), "volume": pa.float64(), "symbol": pa.string(), "instrument_id": pa.int64(), "ts": timestamp or pa.timestamp("ns", tz="America/New_York")}
    types.update(replace or {})
    fields = [pa.field(name, kind) for name, kind in types.items() if name not in omit]
    fields.extend(pa.field(name, kind) for name, kind in append)
    return pa.schema(fields, metadata=metadata)


def _stamps(count=2):
    return [datetime.fromisoformat("2017-01-03T18:00:00-05:00") + timedelta(minutes=index) for index in range(count)]


def _table(stamps=None, values=None, schema=None):
    stamps = stamps or _stamps()
    count = len(stamps)
    columns = {"open": [100.0] * count, "high": [101.0] * count, "low": [99.0] * count, "close": [100.5] * count, "volume": [10.0] * count, "symbol": ["NQH7"] * count, "instrument_id": [7] * count, "ts": stamps}
    if count >= 2:
        columns.update({"open": [100.0, 100.5] + [101.0] * (count - 2), "high": [101.0, 102.0] + [102.0] * (count - 2), "low": [99.0, 100.0] + [100.0] * (count - 2), "close": [100.5, 101.5] + [101.0] * (count - 2), "volume": [10.0, 20.0] + [30.0] * (count - 2)})
    columns.update(values or {})
    schema = schema or _schema()
    return pa.Table.from_arrays([pa.array(columns[field.name], type=field.type) for field in schema], schema=schema)


def _write(root, relative, table, row_group=4096):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, row_group_size=row_group)
    return path


def _explicit(core):
    return {"output": str(core.SOURCE), "output_sha256": core.SOURCE_SHA, "selection": core.SELECTION, "current_session_volume_used": False, "sealed_rows_written": False, "start": core.START, "end_exclusive": core.UPSTREAM_END}


def _child(core, output_sha, audit):
    return {"format_version": 1, "artifact": "causal_previous_session_front_pre_holdout", "selection": core.SELECTION, "current_session_volume_used": False, "source": str(core.SOURCE), "source_declared_sha256": core.SOURCE_SHA, "source_rehashed_during_materialization": False, "source_manifest": None, "source_manifest_sha256": None, "end_exclusive": core.CUTOFF, "sealed_rows_written": False, "rows": audit["rows"], "start": audit["first_timestamp"], "end": audit["last_timestamp"], "output": str(core.OUTPUT), "output_sha256": output_sha, "row_group_rows": core.ROW_GROUP_ROWS}


@pytest.mark.parametrize("case", ("fixed_offset", "naive", "wrong_unit", "missing", "extra", "forbidden", "order"))
def test_01_exact_named_zone_schema_and_order(custody, case):
    core, _, _ = custody
    core._validate_schema(_schema())
    bad = {"fixed_offset": _schema(pa.timestamp("ns", tz="-05:00")), "naive": _schema(pa.timestamp("ns")), "wrong_unit": _schema(pa.timestamp("us", tz="America/New_York")), "missing": _schema(omit=("volume",)), "extra": _schema(append=(("venue", pa.string()),)), "forbidden": _schema(append=(("future_return", pa.float64()),))}
    if case == "order":
        fields = list(_schema())
        fields[0], fields[1] = fields[1], fields[0]
        bad[case] = pa.schema(fields)
    with pytest.raises(RuntimeError):
        core._validate_schema(bad[case])


def test_02_inclusive_lower_exclusive_cutoff_and_count(custody, monkeypatch):
    core, _, root = custody
    path = _write(root, core.SOURCE, _table(_stamps() + [datetime.fromisoformat(core.CUTOFF)]))
    monkeypatch.setattr(core, "SOURCE_BYTES", path.stat().st_size)
    audit = core._audit(core.SOURCE, source=True)
    assert (audit["rows"], audit["first_timestamp"]) == (2, core.START) and audit["last_timestamp_ns"] < core._timestamp_ns(core.CUTOFF)
    relative = Path("data/processed/lower-missing.parquet")
    _write(root, relative, _table(_stamps()[1:]))
    with pytest.raises(RuntimeError, match="boundary"):
        core._audit(relative, source=False)


@pytest.mark.parametrize("row_group", (3, 2), ids=("same_row_group", "later_row_group_min"))
def test_02b_source_cutoff_cannot_hide_later_in_interval_row(custody, monkeypatch, row_group):
    core, _, root = custody
    start, cutoff = _stamps(1)[0], datetime.fromisoformat(core.CUTOFF)
    relative = Path(f"data/processed/cutoff-regression-{row_group}.parquet")
    path = _write(root, relative, _table([start, cutoff, start + timedelta(minutes=1)]), row_group)
    monkeypatch.setattr(core, "SOURCE_BYTES", path.stat().st_size)
    monkeypatch.setattr(core, "BATCH_ROWS", 2)
    with pytest.raises(RuntimeError, match="later row"):
        core._audit(relative, source=True)


def test_03_literal_commitment_batch_and_row_group_invariance(custody, monkeypatch):
    core, _, root = custody
    left, right = Path("data/processed/vector-a.parquet"), Path("data/processed/vector-b.parquet")
    _write(root, left, _table(), 1)
    _write(root, right, _table(), 2)
    monkeypatch.setattr(core, "BATCH_ROWS", 1)
    first = core._audit(left, source=False)
    monkeypatch.setattr(core, "BATCH_ROWS", 2)
    second = core._audit(right, source=False)
    assert first["typed_row_commitment"] == second["typed_row_commitment"] == KNOWN_COMMITMENT
    with pytest.raises(RuntimeError, match="projection"):
        core._projection_equality(first, core._audit(_write(root, Path("data/processed/changed.parquet"), _table(values={"close": [100.25, 101.5]})), source=False))


@pytest.mark.parametrize(("case", "offsets", "groups"), (("in_batch_disorder", (0, 2, 1), 3), ("cross_group_disorder", (0, 2, 1), 2), ("cross_group_duplicate", (0, 1, 1), 2)))
def test_04_global_disorder_and_duplicates(custody, case, offsets, groups):
    core, _, root = custody
    start = _stamps(1)[0]
    relative = Path(f"data/processed/{case}.parquet")
    _write(root, relative, _table([start + timedelta(minutes=value) for value in offsets]), groups)
    with pytest.raises(RuntimeError, match="order/uniqueness"):
        core._audit(relative, source=False)


@pytest.mark.parametrize(("case", "values"), (("null", {"open": [100.0, None]}), ("nonfinite", {"close": [100.5, float("nan")]}), ("geometry", {"high": [99.5, 102.0]}), ("negative_volume", {"volume": [10.0, -1.0]})))
def test_05_values_types_metadata_and_instrument_range(custody, case, values):
    core, _, root = custody
    relative = Path(f"data/processed/{case}.parquet")
    _write(root, relative, _table(values=values))
    with pytest.raises(RuntimeError):
        core._audit(relative, source=False)
    base = {"_schema": _schema(), "field_order": list(core.FIELDS), "arrow_schema_hex": "a", "rows": 2, "first_timestamp_ns": 1, "last_timestamp_ns": 2, "typed_row_commitment": "a" * 64}
    with pytest.raises(RuntimeError, match="schema/metadata"):
        core._projection_equality(base, dict(base, _schema=_schema(metadata={b"changed": b"yes"})))
    oversized = pa.scalar(1 << 63, type=pa.uint64())
    assert oversized.as_py() > (1 << 63) - 1
    with pytest.raises(RuntimeError, match="signed"):
        core._validate_schema(_schema(replace={"instrument_id": pa.uint64()}))


@pytest.mark.parametrize(("field", "value"), (("output_sha256", "0" * 64), ("selection", "current session"), ("current_session_volume_used", True), ("sealed_rows_written", True)))
def test_06_source_and_manifest_hash_contracts(custody, field, value):
    core, _, root = custody
    source = root / core.SOURCE
    source.write_bytes(b"synthetic-source")
    digest, size, _ = core._hash_file(core.SOURCE)
    assert (digest, size) == (hashlib.sha256(b"synthetic-source").hexdigest(), 16)
    assert digest != core.SOURCE_SHA
    manifest = _explicit(core)
    assert core._explicit_contract(manifest) == manifest
    manifest[field] = value
    with pytest.raises(RuntimeError, match="manifest"):
        core._explicit_contract(manifest)


def test_07_symlink_and_dataless_reject_before_open(custody, monkeypatch):
    core, gate, root = custody
    outside = root / "outside"
    outside.mkdir()
    (root / "redirect").symlink_to(outside, target_is_directory=True)
    for reject in (lambda: core._absent(Path("redirect/x")), lambda: gate.parent_chain("redirect/x")):
        with pytest.raises(RuntimeError, match="unsafe ancestor"):
            reject()
    leaf = root / "data/processed/local.parquet"
    leaf.write_bytes(b"x")
    link = Path("data/processed/link.parquet")
    (root / link).symlink_to(leaf)
    opened, real_open = [], core.os.open

    def forbidden_open(*args, **kwargs):
        opened.append(args[0])
        raise AssertionError("content opened")

    monkeypatch.setattr(core.os, "open", forbidden_open)
    with pytest.raises(RuntimeError, match="ordinary"):
        core._fenced(link).__enter__()
    monkeypatch.setattr(core.os, "open", real_open)
    real_lstat, info = core.os.lstat, core.os.lstat(leaf)
    names = ("st_dev", "st_ino", "st_mode", "st_size", "st_blocks", "st_mtime_ns", "st_ctime_ns")
    fake = SimpleNamespace(**{name: getattr(info, name) for name in names}, st_flags=core.SF_DATALESS)
    monkeypatch.setattr(core.os, "lstat", lambda path: fake if Path(path) == leaf else real_lstat(path))
    monkeypatch.setattr(core.os, "open", forbidden_open)
    with pytest.raises(RuntimeError, match="dataless"):
        core._fenced(Path("data/processed/local.parquet"), allocated=True).__enter__()
    assert not opened


@pytest.mark.parametrize("field", ("source_manifest", "source_manifest_sha256"))
def test_08_derived_manifest_and_child_provenance(custody, field):
    core, _, root = custody
    audit = {"rows": 2, "first_timestamp": core.START, "last_timestamp": "2017-01-03T18:01:00-05:00"}
    manifest = _child(core, "d" * 64, audit)
    assert core._child_contract(manifest, "d" * 64, audit) == manifest
    manifest[field] = "forbidden"
    with pytest.raises(RuntimeError, match="child manifest"):
        core._child_contract(manifest, "d" * 64, audit)
    (root / core.DERIVED).write_text("present")
    with pytest.raises(FileExistsError):
        core._absent(core.DERIVED)


def test_09_create_once_targets_never_clobber(custody):
    core, gate, root = custody
    for relative in (core.CORE_RESULT, core.OUTPUT, core.OUTPUT_MANIFEST, core.TEMPORARY):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"sentinel")
        with pytest.raises(FileExistsError):
            core._absent(relative)
        assert path.read_bytes() == b"sentinel"
    claim = root / gate.PATHS["claim"]
    claim.write_bytes(b"claim")
    with pytest.raises(FileExistsError):
        gate.create_json(gate.PATHS["claim"], {"replacement": True})
    attempt = root / gate.ATTEMPT_ROOT
    attempt.mkdir(exist_ok=True)
    with pytest.raises(FileExistsError):
        gate.execute({})


def _gate_bundle(gate, root):
    prereg = root / gate.PATHS["preregistration"]
    prereg.write_bytes(b"prereg\n")
    gate.PREREG_SHA = hashlib.sha256(prereg.read_bytes()).hexdigest()
    evidence = {}
    for relative in gate.GATE_EVIDENCE:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"evidence:{relative}\n".encode())
        evidence[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    candidate = {}
    for relative in gate.CANDIDATES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
        candidate[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    bundle, consensus_sha, status = hashlib.sha256(gate.canonical(candidate)).hexdigest(), evidence[gate.GATE_EVIDENCE[-1]], "gate2_physical_attempt_authorized"
    interpreter_sha = hashlib.sha256(Path(gate.sys.executable).read_bytes()).hexdigest()
    packages = {name: gate.importlib.metadata.version(name) for name in ("pyarrow", "pandas")}
    authority = {key: key in gate.TRUE_AUTHORITY for key in gate.TRUE_AUTHORITY | gate.FALSE_AUTHORITY}
    bindings = {"preregistration_sha256": gate.PREREG_SHA, "gate_evidence_sha256": evidence, "gate_consensus_sha256": consensus_sha, "candidate_bundle_sha256": bundle, "interpreter_sha256": interpreter_sha, "package_versions": packages}
    record = {"experiment_id": gate.EXPERIMENT, "product_version": gate.PRODUCT, "status": status, "bindings": bindings, "authority": authority}
    ledger = gate.canonical(record)
    (root / gate.PATHS["ledger"]).write_bytes(ledger)
    relative = "reports/validation_2026-07-30/gate2-command.json"
    manifest = {"format_version": 1, "experiment_id": gate.EXPERIMENT, "product_version": gate.PRODUCT, "preregistration_sha256": gate.PREREG_SHA, "gate": 2, "attempt_id": "physical_001", "cwd": str(root), "interpreter": gate.sys.executable, "controller_argv": gate.CORE_ARGV, "paths": dict(gate.PATHS, command_manifest=relative), "candidate_sha256": candidate, "candidate_bundle_sha256": bundle, "gate_consensus_sha256": consensus_sha, "gate_evidence_sha256": evidence, "interpreter_sha256": interpreter_sha, "package_versions": packages, "authorization_ledger_head": hashlib.sha256(ledger).hexdigest(), "authority_status": status}
    command = root / relative
    command.write_bytes(gate.canonical(manifest))
    return command, manifest


@pytest.mark.parametrize("case", ("pass", "command", "candidate", "evidence", "runtime", "authority", "sha"))
def test_10_wrapper_identity_authority_and_mutation_order(custody, monkeypatch, case):
    _, gate, root = custody
    interpreter = root / "synthetic-python"
    interpreter.write_bytes(b"synthetic interpreter")
    monkeypatch.setattr(gate.sys, "executable", str(interpreter))
    monkeypatch.setattr(gate, "CORE_ARGV", [str(interpreter), gate.PATHS["controller"]])
    command, manifest = _gate_bundle(gate, root)
    if case == "command":
        manifest["controller_argv"] = [gate.sys.executable, "wrong.py"]
    if case == "candidate":
        manifest["candidate_sha256"][gate.CANDIDATES[0]] = "0" * 64
        manifest["candidate_bundle_sha256"] = hashlib.sha256(gate.canonical(manifest["candidate_sha256"])).hexdigest()
    if case == "evidence":
        manifest["gate_evidence_sha256"][gate.GATE_EVIDENCE[0]] = "0" * 64
    if case == "runtime":
        manifest["interpreter_sha256"] = "0" * 64
    if case == "authority":
        record = json.loads((root / gate.PATHS["ledger"]).read_bytes())
        record["authority"]["ohlcv_authorized"] = False
        ledger = gate.canonical(record)
        (root / gate.PATHS["ledger"]).write_bytes(ledger)
        manifest["authorization_ledger_head"] = hashlib.sha256(ledger).hexdigest()
    command.write_bytes(gate.canonical(manifest))
    digest = "0" * 64 if case == "sha" else hashlib.sha256(command.read_bytes()).hexdigest()
    if case == "pass":
        checked, relative, _ = gate.preflight(command, digest)
        assert checked == manifest and set(manifest) == gate.COMMAND_KEYS and relative.endswith("gate2-command.json")
    else:
        with pytest.raises(RuntimeError):
            gate.preflight(command, digest)
    events = []
    monkeypatch.setattr(gate, "preflight", lambda *args: (events.append("preflight") or ({"authorization_ledger_head": "h", "candidate_bundle_sha256": "b"}, "command.json", b"ledger\n")))
    monkeypatch.setattr(gate, "create_json", lambda path, value: events.append(f"create:{path}") or "claim")
    monkeypatch.setattr(gate, "append_setup", lambda *args: events.append("setup") or "head")
    monkeypatch.setattr(gate, "execute", lambda context: events.append("execute") or 0)
    monkeypatch.setattr(gate.sys, "argv", ["gate", "--command-manifest", "command.json", "--command-manifest-sha256", "d" * 64])
    with pytest.raises(SystemExit):
        gate.main()
    assert events == ["preflight", f"create:{gate.PATHS['claim']}", "setup", "execute"]


def _core_harness(core, monkeypatch, root, mode="ok"):
    events, captured, seen = [], [], {}
    schema = _schema()
    last = "2017-01-03T18:01:00-05:00"
    output_sha = hashlib.sha256(b"output").hexdigest()
    manifest_sha = hashlib.sha256(b"manifest").hexdigest()
    audit = {"_schema": schema, "field_order": list(core.FIELDS), "arrow_schema_hex": schema.serialize().to_pybytes().hex(), "parquet_schema_rendering": "synthetic", "stable_stat": {}, "rows": 2, "first_timestamp_ns": core._timestamp_ns(core.START), "last_timestamp_ns": core._timestamp_ns(last), "first_timestamp": core.START, "last_timestamp": last, "typed_row_commitment": KNOWN_COMMITMENT, "row_groups_read": 1, "peak_batch_rows": 2, "maximum_batches_held": 1}
    output = dict(audit)
    if mode == "output_mismatch":
        output["typed_row_commitment"] = "0" * 64
    child = _child(core, output_sha, output)
    if mode == "bad_manifest":
        child["rows"] = 3

    def hash_file(relative, **kwargs):
        count = seen.get(relative, 0)
        seen[relative] = count + 1
        events.append(f"hash:{relative}")
        if relative == core.SOURCE:
            digest = "0" * 64 if mode == "source_pre_hash" and count == 0 else core.SOURCE_SHA
            digest = "1" * 64 if mode == "source_drift" and count else digest
            return digest, core.SOURCE_BYTES, {}
        return (core.MATERIALIZER_SHA, 100, {}) if relative == core.MATERIALIZER else (output_sha, 100, {})

    def json_file(relative):
        events.append(f"json:{relative}")
        if relative == core.EXPLICIT:
            value = _explicit(core)
            if mode == "explicit_field":
                value["selection"] = "future"
            return value, ("0" * 64 if mode == "explicit_hash" else core.EXPLICIT_SHA), 100, {}
        return child, manifest_sha, 100, {}

    def fsync_leaf(relative, **kwargs):
        events.append(f"fsync:{relative}")
        if mode == "missing_output" and relative == core.OUTPUT:
            raise FileNotFoundError(relative)
        return {}

    monkeypatch.chdir(root)
    monkeypatch.setattr(core.sys, "argv", ["core"])
    monkeypatch.setattr(core, "_directory", lambda path: None)
    monkeypatch.setattr(core, "_absent", lambda path: None)
    monkeypatch.setattr(core, "_hash_file", hash_file)
    monkeypatch.setattr(core, "_json_file", json_file)
    monkeypatch.setattr(core, "_schema_only", lambda: events.append("schema") or {"_schema": schema, "stable_stat": {}, "field_order": list(core.FIELDS)})
    monkeypatch.setattr(core, "_audit", lambda path, source: events.append("source_audit" if source else "output_audit") or (dict(audit) if source else dict(output)))
    monkeypatch.setattr(core, "_fsync_leaf", fsync_leaf)
    monkeypatch.setattr(core, "_fsync_dir", lambda path: events.append(f"fsync_dir:{path}"))
    monkeypatch.setattr(core, "_exclusive_result", lambda value: events.append("result") or captured.append(value))
    monkeypatch.setattr(core, "_progress", lambda *args, **kwargs: None)
    monkeypatch.setattr(core.subprocess, "run", lambda *args, **kwargs: events.append("child") or SimpleNamespace(returncode=9 if mode == "nonzero" else 0))
    return events, captured


@pytest.mark.parametrize("mode", ("nonzero", "missing_output", "bad_manifest", "source_drift", "output_mismatch", "source_pre_hash", "explicit_hash", "explicit_field"))
def test_11_core_failure_matrix_and_schema_before_child(custody, monkeypatch, mode):
    core, _, root = custody
    events, captured = _core_harness(core, monkeypatch, root, mode)
    with pytest.raises((RuntimeError, FileNotFoundError)):
        core.main()
    assert not captured and "result" not in events
    if "child" in events:
        assert events.index("schema") < events.index("child")


@pytest.mark.parametrize("returncode", (0, 7))
def test_12_fsync_core_terminal_hash_binding_and_no_failure_go(custody, monkeypatch, returncode):
    core, gate, root = custody
    events, captured = _core_harness(core, monkeypatch, root)
    core.main()
    result = captured[0]
    assert events.index("schema") < events.index("child")
    assert events.count("child") == 1 and events[-1] == "result"
    assert events.index(f"fsync:{core.OUTPUT}") < events.index(f"hash:{core.OUTPUT}")
    assert events.index(f"fsync:{core.OUTPUT_MANIFEST}") < events.index(f"hash:{core.OUTPUT}")
    assert events.index(f"fsync_dir:{core.OUTPUT.parent}") < events.index(f"hash:{core.OUTPUT}")
    assert result["pre_child"]["schema_validation_completed_before_child"] is True and result["resources"]["full_frame_retained"] is False and result["child"]["argv"][-4:] == ["--batch-rows", "4096", "--row-group-rows", "4096"]
    (root / gate.PATHS["output"]).write_bytes(b"output")
    (root / gate.PATHS["output_manifest"]).write_bytes(b"manifest")
    core_result = root / gate.PATHS["core_result"]
    core_result.parent.mkdir(parents=True)
    core_result.write_bytes(gate.canonical(result))
    assert gate.validate_core()[0] == hashlib.sha256(gate.canonical(result)).hexdigest()
    core_result.write_bytes(gate.canonical(dict(result, mbo_accesses=1)))
    with pytest.raises(RuntimeError, match="execution/schema"):
        gate.validate_core()
    core_result.write_bytes(gate.canonical(result))
    gate.ROOT = root / "terminal-execution"
    (gate.ROOT / "artifacts").mkdir(parents=True)
    terminal_events = []
    original = gate.create_json

    def create(path, value):
        terminal_events.append(path)
        return original(path, value)

    gate.create_json = create
    gate.subprocess.run = lambda *args, **kwargs: SimpleNamespace(returncode=returncode)
    gate.validate_core = lambda: ("r" * 64, "o" * 64, "m" * 64)
    status = gate.execute({"binding": "frozen"})
    terminal = json.loads((gate.ROOT / gate.PATHS["terminal"]).read_text())
    assert terminal_events == [gate.PATHS["attempt"], gate.PATHS["terminal"]]
    assert list((gate.ROOT / gate.ATTEMPT_ROOT).glob("*TERMINAL*")) == [gate.ROOT / gate.PATHS["terminal"]]
    if returncode == 0:
        assert status == 0 and terminal["core_result_sha256"] == "r" * 64
    else:
        assert status == 1 and terminal["status"] == "NOGO"


def test_13_fixed_4096_limits_and_no_full_frame(custody):
    core, gate, root = custody
    count, start = core.ROW_GROUP_ROWS + 1, _stamps(1)[0]
    stamps = [start + timedelta(minutes=index) for index in range(count)]
    relative = Path("data/processed/oversized-row-group.parquet")
    _write(root, relative, _table(stamps), count)
    with pytest.raises(RuntimeError, match="row-group limit"):
        core._audit(relative, source=False)
    source, gate_source = CORE_PATH.read_text(encoding="utf-8"), GATE_PATH.read_text(encoding="utf-8")
    assert core.BATCH_ROWS == core.ROW_GROUP_ROWS == 4096
    assert "iter_batches" in source and "to_pandas" not in source and gate.CORE_ARGV[1:3] == ["-I", "-B"] and '"-I", "-B"' in source and all("sys.flags.isolated" in text for text in (source, gate_source))
