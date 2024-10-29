import sys
import pathlib

# import from submodule
ext_path = pathlib.Path(__file__).parent / "../ext"
sys.path.extend([str(p.absolute()) for p in ext_path.glob("*")])
