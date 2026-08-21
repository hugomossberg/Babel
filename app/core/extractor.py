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
        
        duration = 0.0
        container = data.get("container", {}).get("properties", {})
        if "duration" in container:
            duration = float(container["duration"]) / 1e9 # mkvmerge returns nanoseconds
        
        tracks["duration"] = duration
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
            cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", video_path]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=60)
            data = json.loads(result.stdout)
            
            duration = 0.0
            if "format" in data and "duration" in data["format"]:
                duration = float(data["format"]["duration"])
            tracks["duration"] = duration

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
    from app.core.languages import get_language
    lang_obj = get_language(preferred_lang)
    lang_prefixes = lang_obj.aliases if lang_obj else [preferred_lang.lower()]
    TEXT_CODECS = {"SubRip/SRT", "S_TEXT/UTF8", "S_TEXT/ASS", "S_TEXT/SSA", "S_TEXT/WEBVTT", "SubStationAlpha", "WebVTT"}

    candidates = []
    for i, track in enumerate(sub_tracks):
        lang = track.get("language", "").lower()
        forced = track.get("forced", False)
        codec = track.get("codec", "")
        title = track.get("title", "").lower() if track.get("title") else ""
        
        # Skip outright bad tracks
        if any(bad in title for bad in ["commentary", "director", "description", "audio description"]):
            continue

        is_text_codec = any(tc.lower() in codec.lower() for tc in TEXT_CODECS) or "srt" in codec.lower() or "text" in codec.lower() or "ass" in codec.lower() or "utf" in codec.lower()
        
        if any(lp == lang or lang.startswith(lp) for lp in lang_prefixes) and is_text_codec:
            score = 100
            if forced or any(kw in title for kw in ["forced", "signs", "songs", "foreign", "parts", "descriptive"]):
                continue  # Skip forced tracks entirely for full translation
            
            if any(kw in title for kw in ["full", "sdh", "normal", "dialogue"]):
                score += 20
                
            if track.get("default"):
                score += 10
                
            candidates.append({"score": score, "id": track.get("id"), "index": i, "codec": codec})

    candidates.sort(key=lambda x: x["score"], reverse=True)
    if not candidates:
        return False

    duration = tracks_info.get("duration", 0.0)
    import srt

    for cand in candidates:
        selected_track_id = cand["id"]
        selected_sub_index = cand["index"]
        selected_codec = cand["codec"].lower()

        success = False
        # Try mkvextract first
        try:
            cmd = ["mkvextract", "tracks", video_path, f"{selected_track_id}:{output_srt_path}"]
            subprocess.run(cmd, capture_output=True, check=True, timeout=120)
            
            if os.path.exists(output_srt_path) and os.path.getsize(output_srt_path) > 100:
                if any(x in selected_codec for x in ["ass", "ssa", "vtt", "webvtt"]):
                    temp_file = output_srt_path + ".tmp"
                    os.rename(output_srt_path, temp_file)
                    ffmpeg_cmd = ["ffmpeg", "-y", "-i", temp_file, "-c:s", "srt", output_srt_path]
                    subprocess.run(ffmpeg_cmd, capture_output=True, timeout=60)
                    try: os.remove(temp_file)
                    except: pass
                
                if os.path.exists(output_srt_path) and os.path.getsize(output_srt_path) > 0:
                    success = True
        except Exception:
            pass

        if not success:
            # Fallback to ffmpeg
            try:
                cmd = [
                    "ffmpeg", "-y", "-i", video_path,
                    "-map", f"0:s:{selected_sub_index}",
                    "-c:s", "srt",
                    output_srt_path
                ]
                subprocess.run(cmd, capture_output=True, check=True, timeout=120)
                if os.path.exists(output_srt_path) and os.path.getsize(output_srt_path) > 100:
                    success = True
            except Exception:
                pass

        if success:
            # Sanity check for partial sources
            try:
                with open(output_srt_path, "r", encoding="utf-8-sig") as f:
                    content = f.read()
                
                subs = list(srt.parse(content))
                if not subs:
                    continue
                
                last_end = subs[-1].end.total_seconds()
                
                # Conservative logic to detect obvious partial sources
                if duration > 1200: # 20 mins
                    if len(subs) < 100 and last_end < (duration * 0.25):
                        # Rejected as partial source
                        continue
                
                return True
            except Exception:
                # If parsing fails or anything weird, we still extracted it, just return True
                return True

    return False
