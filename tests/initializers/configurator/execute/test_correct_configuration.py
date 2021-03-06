import pytest

from PyPWA.initializers.configurator.execute import _correct_configuration

template_1 = {
    "predetermined value": ["this", "that", "other"],
    "numbers": int,
    "exact value": float,
    "is true": bool,
    "a list": list
}

template_2 = {
    "general settings": {
        "number of threads": int,
        "debug": ["info", "debug", "warning"]
    },
    "main": {
        "settings": set,
        "data": str,
        "more nests": {
            "correct": bool,
            "settings": dict,
            "extra data": str
        }
    }
}

found_1 = {
    "predetermin value": "othr",
    "numbes": 5.001,
    "exactvalue": 2.345,
    "iss true": "tRue",
    "the list": ["list", "of", "values"]
}

found_2 = {
    "General settigns": {
        "nuM of threads": 5.2,
        "debg": "inf"
    },
    "MAIN": {
        "setings": ["limit_A1", "limit_A1"],
        "daTa": "/usr/local/this",
        "moR nests": {
            "extr dat": None,
            "CoRRct": "tru",
            "settngs": {
                "somedata": "That we don't know about for whatever reason."
            }
        }
    }
}


def temp1(self):
    return template_1


def temp2(self):
    return template_2


@pytest.fixture
def settings_aid_1(monkeypatch):
    monkeypatch.setattr(
        _correct_configuration._storage_data.Templates,
        "get_templates",
        temp1
    )
    aid = _correct_configuration.SettingsAid()
    return aid.correct_settings(found_1)


@pytest.fixture
def settings_aid_2(monkeypatch):
    monkeypatch.setattr(
        _correct_configuration._storage_data.Templates,
        "get_templates",
        temp2
    )
    aid = _correct_configuration.SettingsAid()
    return aid.correct_settings(found_2)


def test_1_predetermined_value(settings_aid_1):
    assert settings_aid_1["predetermined value"] == "other"


def test_1_numbers(settings_aid_1):
    assert settings_aid_1["numbers"] == 5


def test_1_exact_value(settings_aid_1):
    assert settings_aid_1["exact value"] == 2.345


def test_1_is_true(settings_aid_1):
    assert settings_aid_1["is true"] is True


def test_1_a_list(settings_aid_1):
    assert settings_aid_1["a list"] == ["list", "of", "values"]


def test_2_number_of_threads(settings_aid_2):
    assert settings_aid_2["general settings"]["number of threads"] == 5


def test_2_debug(settings_aid_2):
    assert settings_aid_2["general settings"]["debug"] == "info"


def test_2_settings(settings_aid_2):
    assert settings_aid_2["main"]["settings"] == {"limit_A1"}


def test_2_data(settings_aid_2):
    assert settings_aid_2["main"]["data"] == "/usr/local/this"


def test_2_more_nests(settings_aid_2):
    assert settings_aid_2["main"]["more nests"]["correct"] is True


def test_2_extra_data(settings_aid_2):
    assert isinstance(
        settings_aid_2["main"]["more nests"]["extra data"], type(None)
    )
