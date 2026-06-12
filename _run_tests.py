import subprocess
import sys

result = subprocess.run(
    [sys.executable, '-m', 'pytest', 'tests/test_feature_validation.py', '--tb=line', '-p', 'no:cacheprovider'],
    capture_output=True, text=True, cwd=r'd:\SOLO-2\AI_solo_coder_task_B_056'
)
with open(r'd:\SOLO-2\AI_solo_coder_task_B_056\t7.txt', 'w', encoding='utf-8') as f:
    f.write(result.stdout)
    f.write('\n=== STDERR ===\n')
    f.write(result.stderr)
    f.write(f'\n=== RETURN CODE: {result.returncode} ===\n')
print("DONE")
