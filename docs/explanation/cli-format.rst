CLI argument format
===================

Additional arguments such as ``--gar-args``, ``--attack-args``,
``--model-args``, ``--dataset-args``, ``--loss-args``, and
``--criterion-args`` are parsed by :func:`tools.parse_keyval`.

Format
------

Arguments are passed as a list of ``key:value`` strings::

    --gar-args lr:0.01 batch:32

The parser splits on the first colon. Keys and values are returned as a
dictionary. This format was chosen because it is unambiguous and easy to
type on the command line.

Documenting new options
-----------------------

When you add a new research option, document it in this ``key:value`` format
so users can pass it from the command line without ambiguity. Avoid using
spaces or commas inside values; if you need them, quote the entire argument.
