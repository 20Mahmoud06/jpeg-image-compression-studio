# GUI

The PyQt6 GUI is not implemented yet. This directory is reserved for the GUI
phase.

The backend is available through the stable pipeline package:

```python
from jpeg_studio.pipeline import (
    compress_image,
    decompress_image,
    compress_file,
    decompress_file,
)
```

See `docs/BACKEND_INTEGRATION.md` for the full integration contract.
