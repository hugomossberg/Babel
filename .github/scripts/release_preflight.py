"""Read-only preflight for public releases; never creates refs or publishes."""
import ast
import os
from pathlib import Path
import re
import subprocess
import sys


def _default_version(path):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='VERSION' for t in node.targets):
            value=node.value.values[-1] if isinstance(node.value,ast.BoolOp) else node.value
            if isinstance(value,ast.Constant) and isinstance(value.value,str):
                return value.value
    raise ValueError('Release version is not declared in app/config.py')


def validate_release(repository,tag,sha,root,resolve_ref):
    if repository!='hugomossberg/Babel':
        raise ValueError('Only the public Babel repository may publish')
    number=r'(?:0|[1-9][0-9]*)'
    if not re.fullmatch(rf'v{number}\.{number}\.{number}(?:-beta)?',tag):
        raise ValueError('Unsupported release tag; expected vX.Y.Z or vX.Y.Z-beta')
    if not re.fullmatch(r'[0-9a-f]{40}',sha):
        raise ValueError('Invalid release commit SHA')
    version=tag.removeprefix('v')
    if _default_version(root/'app/config.py')!=version:
        raise ValueError('Release tag and source version do not match')
    notes=root/'docs/releases'/f'{tag}.md'
    if not notes.is_file() or not notes.read_text(encoding='utf-8').strip():
        raise ValueError('Reviewed release notes are missing or empty')
    if resolve_ref(f'{sha}^{{commit}}')!=resolve_ref('refs/remotes/origin/main^{commit}'):
        raise ValueError('Release commit must be the current public main tip')
    return version


def main():
    root=Path(__file__).resolve().parents[2]
    def resolve(ref):
        return subprocess.check_output(['git','rev-parse','--verify',ref],cwd=root,text=True).strip()
    try:
        version=validate_release(os.environ.get('GITHUB_REPOSITORY',''),os.environ.get('GITHUB_REF_NAME',''),os.environ.get('GITHUB_SHA',''),root,resolve)
    except (ValueError,OSError,SyntaxError,subprocess.CalledProcessError) as error:
        print(f'Release preflight failed: {error}',file=sys.stderr)
        return 1
    print(f'Validated public release {version}; no publication performed by preflight')
    return 0


if __name__=='__main__':
    sys.exit(main())
