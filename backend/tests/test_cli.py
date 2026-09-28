from actlens import __version__, cli


def test_parser_defaults():
    a = cli.build_parser().parse_args([])
    assert (a.model, a.device, a.dtype, a.host, a.open) == (None, "auto", "float32", "127.0.0.1", False)


def test_parser_options():
    a = cli.build_parser().parse_args(["-m", "Qwen/Qwen2.5-0.5B", "--dtype", "bfloat16", "-p", "9000", "--cache-mb", "512"])
    assert (a.model, a.dtype, a.port, a.cache_mb) == ("Qwen/Qwen2.5-0.5B", "bfloat16", 9000, 512)


def test_version_flag(capsys):
    import pytest

    with pytest.raises(SystemExit) as e:
        cli.main(["--version"])
    assert e.value.code == 0
    assert __version__ in capsys.readouterr().out
