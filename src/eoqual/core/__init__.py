"""
eoqual.core — Briques numériques communes aux modules EOQual.

Fonctions pures sur tableaux, sans entrées/sorties ni visualisation,
dépendant seulement de numpy et scipy :

- :mod:`eoqual.core.fourier` : TFD sans artefacts de bord (décomposition
  périodique + lisse), translation sous-pixel, déquantification ;
- :mod:`eoqual.core.edge`    : profil de bord → ESF → LSF → MTF, MTF50,
  FWHM, RER ;
- :mod:`eoqual.core.bayer`   : motifs de Bayer, plans, masques ;
- :mod:`eoqual.core.psf`     : Airy, Strehl, profil radial, énergie
  encerclée ;
- :mod:`eoqual.core.detrend` : retrait d'un plan avant mesure ;
- :mod:`eoqual.core.tiling`  : découpage en tuiles.

Les sous-modules s'importent directement :

>>> from eoqual.core.fourier import perdecomp
>>> from eoqual.core.edge import frequency_at
"""

__version__ = "0.1.0"
__title__ = "eoqual.core"
__description__ = "Shared numerical building blocks for EOQual modules"
__url__ = "https://github.com/EOQual/eoqual-core"
__uri__ = __url__
__doc__ = __description__ + " <" + __uri__ + ">"
__author__ = "Olivier Amram"
__email__ = "olivier.amram@free.fr"
__license__ = "MIT"
__copyright__ = "Copyright (c) 2026 Olivier Amram"

from . import bayer, detrend, edge, fourier, psf, tiling  # noqa: F401

__all__ = ["bayer", "detrend", "edge", "fourier", "psf", "tiling"]
