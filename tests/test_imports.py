def test_package_imports():
    from jaxqsofit import (
        JAXQSOFit,
        make_custom_component,
        make_custom_line_component,
        make_template_component,
    )

    assert JAXQSOFit is not None
    assert callable(make_custom_component)
    assert callable(make_custom_line_component)
    assert callable(make_template_component)


def test_legacy_default_prior_builder_is_not_public():
    import pytest
    import jaxqsofit

    with pytest.raises(AttributeError):
        getattr(jaxqsofit, "build_default_prior_config")
