Module Reference
================

API documentation for the ``precisionai.agriemb`` package — configuration, dataset,
embeddings, evaluation, and Dense Patch Token feature maps.

precisionai.agriemb.config
------------------------------

Environment-driven configuration (API key/base URL, dataset dir, model, batch size).

.. automodule:: precisionai.agriemb.config
   :members:
   :undoc-members:
   :show-inheritance:

precisionai.agriemb.dataset
------------------------------

Dataset download and cluster sampling, built on ``agristress-utils``.

.. automodule:: precisionai.agriemb.dataset
   :members:
   :undoc-members:
   :show-inheritance:

precisionai.agriemb.embeddings
------------------------------

Image embedding generation via the PAI embedding API.

.. automodule:: precisionai.agriemb.embeddings
   :members:
   :undoc-members:
   :show-inheritance:

precisionai.agriemb.evaluate
------------------------------

Image→Image, Plant→Image, and Plant→Plant retrieval evaluation via ``agrieval``.

.. automodule:: precisionai.agriemb.evaluate
   :members:
   :undoc-members:
   :show-inheritance:

precisionai.agriemb.dpt
------------------------------

Dense Patch Token feature maps: tile images and embed every tile via the PAI
embedding API, then evaluate via ``agrieval``.

.. automodule:: precisionai.agriemb.dpt
   :members:
   :undoc-members:
   :show-inheritance:
