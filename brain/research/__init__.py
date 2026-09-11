"""Trading Brain research: trajectory datasets, the retrieval index, and the
measurements the design rests on.

Nothing here is a runtime authority.  These modules read the future by
construction — they pair each past bar with the sixty minutes that followed it —
and they never emit a market fact, a belief, a decision or an order.

The dependency runs one way: this package imports the geometry and the context
vector from ``brain/core/``, never the reverse, so no research surface can
become something the runtime needs.
"""
