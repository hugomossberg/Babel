import subprocess
import json
import os
from typing import Dict, List, Optional

def inspect_mkv_tracks(video_path: str) -> Dict[str, List[Dict]]:
    """
    Uses mkvmerge / ffprobe to quickly extract information about embedded audio and subtitle tracks.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    tracks = {"subtitles": [], "audio": []}
    try:
        cmd = ["mkvmerge", "-J", video_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=60)
        data = json.loads(result.stdout)
        
        for track in data.get("tracks", []):
            track_type = track.get("type")
            properties = track.get("properties", {})
            lang = properties.get("language", "und")
            track_id = track.get("id")
            codec = track.get("codec")
            forced = properties.get("forced_track", False)
            default = properties.get("default_track", False)
            title = properties.get("track_name", "")

            track_info = {
                "id": track_id,
                "language": lang,
                "codec": codec,
                "forced": forced,
                "default": default,
                "title": title
            }

            if track_type == "subtitles":
                tracks["subtitles"].append(track_info)
            elif track_type == "audio":
                tracks["audio"].append(track_info)

        return tracks
    except Exception:
        # Fallback to ffprobe
        try:
            cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", video_path]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=60)
            data = json.loads(result.stdout)

            for stream in data.get("streams", []):
                codec_type = stream.get("codec_type")
                tags = stream.get("tags", {})
                lang = tags.get("language", "und")
                track_id = stream.get("index")
                codec = stream.get("codec_name", "")
                
                disposition = stream.get("disposition", {})
                forced = bool(disposition.get("forced", 0))
                default = bool(disposition.get("default", 0))
                title = tags.get("title", "")

                track_info = {
                    "id": track_id,
                    "language": lang,
                    "codec": codec,
                    "forced": forced,
                    "default": default,
                    "title": title
                }

                if codec_type == "subtitle":
                    tracks["subtitles"].append(track_info)
                elif codec_type == "audio":
                    tracks["audio"].append(track_info)

            return tracks
        except Exception as e2:
            return {"subtitles": [], "audio": [], "error": str(e2)}

def extract_embedded_srt(video_path: str, output_srt_path: str, preferred_lang: str = "eng") -> bool:
    """
    Extracts the best matching embedded subtitle track to an SRT file using mkvextract or ffmpeg.
    Prefers non-forced SRT / SubRip / text subtitles.
    """
    tracks_info = inspect_mkv_tracks(video_path)
    sub_tracks = tracks_info.get("subtitles", [])
    
    selected_track_id = None
    selected_sub_index = None
    # Look for matching language codes (e.g. 'eng', 'en', 'swe', 'sv')
    lang_prefixes = [preferred_lang.lower()]
    if preferred_lang.lower() in ["eng", "en"]:
        lang_prefixes = ["eng", "en"]
    elif preferred_lang.lower() in ["swe", "sv"]:
        lang_prefixes = ["swe", "sv"]

    TEXT_CODECS = {"SubRip/SRT", "S_TEXT/UTF8", "S_TEXT/ASS", "S_TEXT/SSA", "S_TEXT/WEBVTT", "SubStationAlpha", "WebVTT"}

    for i, track in enumerate(sub_tracks):
        lang = track.get("language", "").lower()
        forced = track.get("forced", False)
        codec = track.get("codec", "")
        title = track.get("title", "").lower() if track.get("title") else ""
        
        # Skip commentary or description tracks
        if any(bad in title for bad in ["commentary", "director", "description", "audio description"]):
            continue

        is_text_codec = any(tc.lower() in codec.lower() for tc in TEXT_CODECS) or "srt" in codec.lower() or "text" in codec.lower() or "ass" in codec.lower() or "utf" in codec.lower()
        if any(lp == lang or lang.startswith(lp) for lp in lang_prefixes) and not forced and is_text_codec:
            selected_track_id = track.get("id")
            selected_sub_index = i
            break

    if selected_track_id is None:
        return False

    # Try mkvextract first (fastest and cleanest for MKV)
    try:
        selected_codec = ""
        for track in sub_tracks:
            if track.get("id") == selected_track_id:
                selected_codec = track.get("codec", "").lower()
                break
                
        cmd = ["mkvextract", "tracks", video_path, f"{selected_track_id}:{output_srt_path}"]
        subprocess.run(cmd, capture_output=True, check=True, timeout=120)
        
        if os.path.exists(output_srt_path) and os.path.getsize(output_srt_path) > 100:
            if "ass" in selected_codec or "ssa" in selected_codec:
                # Convert ASS to clean SRT using ffmpeg
                temp_ass = output_srt_path + ".ass"
                os.rename(output_srt_path, temp_ass)
                ffmpeg_cmd = ["ffmpeg", "-y", "-i", temp_ass, "-c:s", "srt", output_srt_path]
                res = subprocess.run(ffmpeg_cmd, capture_output=True, timeout=60)
                try: os.remove(temp_ass)
                except: pass
                if res.returncode != 0 or not os.path.exists(output_srt_path) or os.path.getsize(output_srt_path) == 0:
                    return False
            return True
    except Exception:
        pass

    # Fallback to ffmpeg
    try:
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-map", f"0:s:{selected_sub_index}",
            "-c:s", "srt",
            output_srt_path
        ]
        subprocess.run(cmd, capture_output=True, check=True, timeout=120)
        return os.path.exists(output_srt_path) and os.path.getsize(output_srt_path) > 100
    except Exception:
        return False
