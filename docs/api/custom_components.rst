Custom Components Package
=========================

.. currentmodule:: jaxqsofit.custom_components

``SpectralComponentSpec`` is the preferred definition for every custom
spectral component. Set ``kind`` to ``continuum``, ``broad_line``, or
``narrow_line`` and place all definitions in ``cfg.lines.components``.

.. autoclass:: SpectralComponentSpec
   :members:
   :undoc-members:
   :special-members: __init__

Specialized component definitions remain available for model-internal and
advanced use:

.. autoclass:: CustomComponentSpec
   :members:
   :undoc-members:
   :special-members: __init__
   :show-inheritance:

.. autoclass:: CustomLineComponentSpec
   :members:
   :undoc-members:
   :special-members: __init__
   :show-inheritance:

.. autofunction:: make_custom_component

.. autofunction:: make_custom_line_component

.. autofunction:: make_template_component
