Defaults Package
================

.. currentmodule:: jaxqsofit.defaults

.. autofunction:: build_default_bal_components

.. data:: DEFAULT_LINE_CONFIG

   Default emission-line configuration used by
   :meth:`jaxqsofit.PriorConfig.from_spectrum`.

.. data:: DEFAULT_LINE_PRIOR_ROWS

   Default line-prior table rows used by
   :meth:`jaxqsofit.PriorConfig.from_spectrum`.

.. data:: DEFAULT_ELG_NARROW_LINE_PRIOR_ROWS

   Optional narrow-line table rows appended when
   ``include_elg_narrow_lines=True``.

.. data:: DEFAULT_HIGH_IONIZATION_LINE_PRIOR_ROWS

   Optional high-ionization/coronal narrow-line table rows appended when
   ``include_high_ionization_lines=True``.
