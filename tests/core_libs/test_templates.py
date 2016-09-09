import pytest

from PyPWA.core_libs import templates


def test_AllObjects_CallAbstractMethod_RaiseNotImplementedError():
    """
    Ensures that the objects will raise a NotImplementedError when called.
    """
    with pytest.raises(NotImplementedError):
        options = templates.OptionsTemplate()
        options.request_metadata("name")

    minimizer = templates.MinimizerTemplate({"this": "that"})
    with pytest.raises(NotImplementedError):
        minimizer.main_options("function")

    with pytest.raises(NotImplementedError):
        minimizer.start()

    processing = templates.KernelProcessingTemplate({"this": "that"})
    with pytest.raises(NotImplementedError):
        processing.main_options("more", "less", "something")

    with pytest.raises(NotImplementedError):
        processing.fetch_interface()

    data_reader = templates.DataReaderTemplate({"this": "that"})
    with pytest.raises(NotImplementedError):
        data_reader.return_reader("the file")

    with pytest.raises(NotImplementedError):
        data_reader.return_writer("the file", 1)

    data_parser = templates.DataParserTemplate({"this": "that"})
    with pytest.raises(NotImplementedError):
        data_parser.parse("the file")

    with pytest.raises(NotImplementedError):
        data_parser.write("the data", "the file")

    writer = templates.WriterTemplate("Something")
    with pytest.raises(NotImplementedError):
        writer.write(12)

    with pytest.raises(NotImplementedError):
        writer.close()

    with pytest.raises(NotImplementedError):
        with templates.WriterTemplate("Something") as stream:
            stream.write("else")

    reader = templates.ReaderTemplate("Something")
    with pytest.raises(NotImplementedError):
        reader.next_event

    with pytest.raises(NotImplementedError):
        reader.previous_event

    with pytest.raises(NotImplementedError):
        with templates.ReaderTemplate("Something") as stream:
            stream.reset()

    with pytest.raises(NotImplementedError):
        for event in reader:
            pass

    with pytest.raises(NotImplementedError):
        reader.reset()

    with pytest.raises(NotImplementedError):
        reader.close()

    interface = templates.InterfaceTemplate()
    with pytest.raises(NotImplementedError):
        interface.run()

    with pytest.raises(NotImplementedError):
        interface.previous_value

    with pytest.raises(NotImplementedError):
        interface.stop()

    with pytest.raises(NotImplementedError):
        interface.is_alive

    empty_shell = templates.ShellCoreTemplate()
    with pytest.raises(NotImplementedError):
        empty_shell.make_config({})

    with pytest.raises(NotImplementedError):
        empty_shell.run({})


def test_TemplateOptions_CreateMetaObject_HoldData():
    """
    Tests that the template object renders out its data correctly when
    supplied with usable information.
    """

    class TestObject(templates.OptionsTemplate):
        def _plugin_name(self):
            return "test"

        def _plugin_interface(self):
            return "nothing"

        def _plugin_type(self):
            return self._data_parser

        def _plugin_requires(self):

            function = """\
def function(this, that)
    return this * that"""

            return self._build_function("numpy", function)

        def _plugin_arguments(self):
            return False

        def _default_options(self):
            return {
                "this": 1,
                "that": 2,
                "other": 3
            }

        def _option_levels(self):
            return {
                "this": self._required,
                "that": self._optional,
                "other": self._advanced
            }

        def _option_types(self):
            return {
                "this": bool,
                "that": int,
                "other": int
            }

        def _main_comment(self):
            return "test comment"

        def _option_comments(self):
            return {
                "this": "this",
                "that": "that",
                "other": "or other"
            }

    options = TestObject()
    assert options.request_metadata("name") == "test"
    assert options.request_metadata("interface") == "nothing"
    assert options.request_metadata("provides") == "data parser"
    assert options.request_metadata("arguments") is False
    assert options.request_options("required")["test"]["this"] == 1
    assert options.request_options("optional")["test"]["this"] == 1
    assert options.request_options("optional")["test"]["that"] == 2
    assert options.request_options("advanced")["test"]["other"] == 3
    assert options.request_options("advanced")["test"]["this"] == 1
    assert options.request_options("advanced")["test"]["that"] == 2

    with pytest.raises(KeyError):
        options.request_options("required")["test"]["other"] == 3