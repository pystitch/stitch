import pytest


def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true",
                     help="run slow tests")


def pytest_runtest_setup(item):
    if 'slow' in item.keywords and not item.config.getoption("--run-slow"):
        pytest.skip("need --run-slow option to run")
