import os
from pathlib import Path
from fastapi import HTTPException
from app.core.db import get_setting

def is_safe_path(target_path: str) -> bool:
    try:
        target = Path(target_path).resolve(strict=False)
        
        series_dir = get_setting("media_series_path", "/tv")
        movies_dir = get_setting("media_movies_path", "/movies")
        
        allowed_roots = []
        if series_dir:
            allowed_roots.append(Path(series_dir).resolve(strict=False))
        if movies_dir:
            allowed_roots.append(Path(movies_dir).resolve(strict=False))
            
        for root in allowed_roots:
            try:
                # Check if target is relative to root
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
