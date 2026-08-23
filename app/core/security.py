import os
from pathlib import Path
from fastapi import HTTPException
from app.core.db import get_setting

def is_safe_path(target_path: str) -> bool:
    try:
        if not target_path:
            return False
            
        # Check for traversal pattern
        norm = os.path.normpath(target_path)
        if norm.startswith("..") or "/../" in target_path:
            return False

        target = Path(target_path).resolve(strict=False)
        
        series_dir = get_setting("media_series_path", "/tv")
        movies_dir = get_setting("media_movies_path", "/movies")
        local_prefixes = get_setting("local_path_prefix", "").split(",")
        remote_prefixes = get_setting("remote_path_prefix", "").split(",")
        
        allowed_roots = []
        if series_dir:
            for p in series_dir.split(","):
                p = p.strip()
                if p:
                    allowed_roots.append(Path(p).resolve(strict=False))
        if movies_dir:
            for p in movies_dir.split(","):
                p = p.strip()
                if p:
                    allowed_roots.append(Path(p).resolve(strict=False))
        for p in local_prefixes + remote_prefixes:
            p = p.strip()
            if p:
                allowed_roots.append(Path(p).resolve(strict=False))

        # Standard mounts
        for default_root in ["/tv", "/movies", "/media", "/data", "/downloads"]:
            if os.path.exists(default_root):
                allowed_roots.append(Path(default_root).resolve(strict=False))
            
        for root in allowed_roots:
            try:
                target.relative_to(root)
                return True
            except ValueError:
                continue
                
        return False
    except Exception:
        return False

def validate_media_path(target_path: str) -> str:
    if not is_safe_path(target_path):
        raise HTTPException(status_code=403, detail="Path traversal detected or path not in allowed media roots")
    return target_path
