import sys
import traceback
sys.path.append('e:/AI/ComfyUI_windows_portable/ComfyUI')
try:
    from nodes import a2sb_loader
    print("Import successful!")
except Exception as e:
    print("Import Failed!")
    traceback.print_exc()
