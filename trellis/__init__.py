import os

from . import models
from . import modules

# Eval / VQVAE-only: skip pipelines & render stack (rembg, spconv-heavy paths, etc.)
if os.environ.get("SHAPELLM_EVAL_LIGHT", "") != "1":
    from . import pipelines
    from . import renderers
    from . import representations
    from . import utils
