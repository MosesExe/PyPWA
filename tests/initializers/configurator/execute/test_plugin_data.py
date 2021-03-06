import pytest

from PyPWA.initializers.configurator.execute import _plugin_data


config = {
    "blank program module": {
        "option 1": 123,
        "optn 2": "A collection of words",
        "Optional 3": "prest 2"
    },
    "Builtin Parser": {
        "Enable Cache": False
    }

}


class FakeSettings(object):

    @property
    def loaded_settings(self):
        return config

    @property
    def plugin_ids(self):
        return list(config.keys())


@pytest.fixture
def setup_settings():
    return _plugin_data.SetupProgram(FakeSettings())


def test_full_configurator_execute_with_blank_module(setup_settings):
    setup_settings.setup()
    setup_settings.execute()
