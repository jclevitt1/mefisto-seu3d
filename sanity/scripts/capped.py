import subprocess, sys
try: sys.exit(subprocess.run(sys.argv[2:], timeout=float(sys.argv[1])).returncode)
except subprocess.TimeoutExpired: print('CAPPED: killed after', sys.argv[1], 's', flush=True); sys.exit(124)
