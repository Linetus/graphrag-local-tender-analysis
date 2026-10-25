# Input documents

The pipeline reads PDFs from `data/pliegos/`. They are not included in this repository.

The thesis used public technical tender documents (*pliegos de prescripciones técnicas*) from Spain's public procurement platform, [Plataforma de Contratación del Sector Público](https://contrataciondelestado.es), plus public cybersecurity guides from [CCN-CERT](https://www.ccn-cert.cni.es) and [INCIBE](https://www.incibe.es) for the generalisation test.

To run it, download a few documents with a clear table of contents into `data/pliegos/` and run `python src/main.py` from the repository root.
