Why debug mode changes behavior
===============================

.. warning::

   Attack and aggregator wrappers choose between *checked* (validated) and
   *unchecked* (raw) versions based on the global ``__debug__`` flag.
   Running Python with ``-O`` or ``-OO`` removes validation and changes the
   error surface. Always test with validation enabled before deploying
   unchecked code.

When Python runs in debug mode (the default), every aggregator and attack
call is routed through a wrapper that validates arguments, checks tensor
aliasing, and ensures contract compliance. This adds a small runtime overhead
but catches mistakes early.

When you run with ``python -O`` or ``python -OO``, the ``__debug__`` flag
becomes ``False``. The wrappers then bypass validation and call the raw
implementations directly. This is useful for performance benchmarking, but it
also means that invalid inputs may produce silent corruption or cryptic
crashes instead of clear error messages.

Best practice: keep validation enabled during development and only disable it
for final production runs after the code has been thoroughly tested.
