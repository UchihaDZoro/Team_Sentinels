"""
Stream Resolver Service
Resolves live streams, YouTube URLs, HLS/m3u8 manifests, RTSP, and direct video links
into playable stream URLs for OpenCV VideoCapture.
"""
import re
import logging
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger("stream_resolver")

YOUTUBE_REGEX = re.compile(
    r'(https?://)?(www\.|m\.)?(youtube\.com/(watch\?.*v=|live/|embed/|shorts/)|youtu\.be/)([\w\-]+)',
    re.IGNORECASE
)
URL_REGEX = re.compile(r'^(https?|rtsp|rtmp)://', re.IGNORECASE)


def is_network_stream(source: str) -> bool:
    """Check if the source is a remote network stream/URL."""
    if not isinstance(source, str):
        return False
    return bool(URL_REGEX.match(source.strip()))


def is_youtube_url(source: str) -> bool:
    """Check if the source is a YouTube video or live stream URL."""
    if not isinstance(source, str):
        return False
    return bool(YOUTUBE_REGEX.search(source.strip()))


def resolve_stream_source(source: str) -> Tuple[str, Dict[str, Any]]:
    """
    Resolves a stream source (local file, YouTube URL, RTSP, or direct HTTP/HLS stream).
    Returns (playable_url_or_path, metadata_dict).
    """
    source_clean = source.strip()
    
    # If not a URL, return as local source
    if not is_network_stream(source_clean):
        return source_clean, {"type": "local", "is_live": False}
    
    # Check if it's YouTube or requires yt-dlp extraction
    if is_youtube_url(source_clean) or ("twitch.tv" in source_clean) or ("vimeo.com" in source_clean):
        return _resolve_with_ytdlp(source_clean)
    
    # Direct RTSP or HLS/MP4 link
    is_live = source_clean.startswith("rtsp://") or (".m3u8" in source_clean)
    return source_clean, {
        "type": "network_direct",
        "is_live": is_live,
        "title": source_clean.split("/")[-1] or "Live Stream"
    }


def _resolve_with_ytdlp(url: str) -> Tuple[str, Dict[str, Any]]:
    """Uses yt-dlp to extract the direct playable video or HLS manifest URL."""
    try:
        import yt_dlp
    except ImportError:
        logger.error("yt-dlp is not installed. Install with `pip install yt-dlp`")
        return url, {"type": "error", "error": "yt-dlp not installed"}

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "socket_timeout": 10,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if not info:
                logger.warning(f"Could not extract info from URL: {url}")
                return url, {"type": "error", "error": "Extract info returned empty"}

            title = info.get("title", "Online Stream")
            is_live = bool(info.get("is_live") or info.get("live_status") == "is_live")

            # Check formats
            formats = [
                f for f in info.get("formats", [])
                if f.get("url") and f.get("vcodec") != "none"
            ]

            stream_url = None

            if is_live:
                # For live streams, pick 720p or 480p for high-speed AI processing
                if formats:
                    # Sort by closeness to 720p
                    formats.sort(key=lambda x: abs((x.get("height") or 0) - 720))
                    stream_url = formats[0].get("url")
            else:
                # For recorded videos: prefer MP4 or webm <= 1080p
                if formats:
                    suitable = [f for f in formats if (f.get("height") or 0) <= 1080]
                    if suitable:
                        # Prefer 720p or 1080p
                        suitable.sort(key=lambda x: x.get("height") or 0, reverse=True)
                        stream_url = suitable[0].get("url")
                    else:
                        stream_url = formats[-1].get("url")

            # Fallback to direct url or requested_formats if not found
            if not stream_url:
                if info.get("url"):
                    stream_url = info["url"]
                elif info.get("requested_formats"):
                    for rf in info["requested_formats"]:
                        if rf.get("url") and rf.get("vcodec") != "none":
                            stream_url = rf["url"]
                            break

            if not stream_url:
                logger.warning(f"No playable video format found for {url}")
                return url, {"type": "error", "error": "No playable format"}

            metadata = {
                "type": "youtube" if is_youtube_url(url) else "web_stream",
                "title": title,
                "is_live": is_live,
                "original_url": url,
                "resolved_url": stream_url,
            }
            logger.info(f"Successfully resolved {url} -> {title} (live={is_live})")
            return stream_url, metadata

    except Exception as e:
        logger.error(f"Error resolving stream with yt-dlp: {e}", exc_info=True)
        return url, {"type": "error", "error": str(e)}
