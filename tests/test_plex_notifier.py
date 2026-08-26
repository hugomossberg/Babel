import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from app.services.plex_notifier import (
    notify_plex_library_refresh,
    _map_to_plex_path,
    _match_section,
)

SECTIONS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<MediaContainer size="3">
  <Directory key="1" type="movie" title="Movies">
    <Location id="1" path="/mnt/data/media/movies" />
  </Directory>
  <Directory key="2" type="show" title="TV Shows">
    <Location id="2" path="/mnt/data/media/tv" />
  </Directory>
  <Directory key="6" type="show" title="TV Shows 4K">
    <Location id="6" path="/mnt/data/media/tv4k" />
  </Directory>
</MediaContainer>"""

SECTIONS = [("1", ["/mnt/data/media/movies"]),
            ("2", ["/mnt/data/media/tv"]),
            ("6", ["/mnt/data/media/tv4k"])]


def _settings(**overrides):
    base = {"plex_url": "http://plex:32400", "plex_token": "tok",
            "plex_path_babel_prefix": "", "plex_path_plex_prefix": ""}
    base.update(overrides)
    return lambda key, default="": base.get(key, default)


def test_match_section_picks_correct_library():
    assert _match_section(SECTIONS, "/mnt/data/media/tv/Show/S01E01.sr.srt") == "2"
    assert _match_section(SECTIONS, "/mnt/data/media/movies/Film/Film.sr.srt") == "1"


def test_match_section_prefers_longest_match():
    """tv4k must not be swallowed by a shorter, earlier-listed location."""
    nested = [("2", ["/mnt/data/media"]), ("6", ["/mnt/data/media/tv4k"])]
    assert _match_section(nested, "/mnt/data/media/tv4k/Show/ep.sr.srt") == "6"


def test_match_section_returns_none_when_outside_all_libraries():
    assert _match_section(SECTIONS, "/somewhere/else/file.sr.srt") is None


def test_match_section_does_not_match_sibling_prefix():
    """'/media/tv' must not match '/media/tv4k' by bare string prefix."""
    only_tv = [("2", ["/mnt/data/media/tv"])]
    assert _match_section(only_tv, "/mnt/data/media/tv4k/Show/ep.sr.srt") is None


def test_path_mapping_translates_babel_path_to_plex_path():
    with patch("app.services.plex_notifier.get_setting",
               side_effect=_settings(plex_path_babel_prefix="/data/media",
                                     plex_path_plex_prefix="/mnt/data/media")):
        assert _map_to_plex_path("/data/media/tv/S.srt") == "/mnt/data/media/tv/S.srt"


def test_path_mapping_is_identity_when_unconfigured():
    with patch("app.services.plex_notifier.get_setting", side_effect=_settings()):
        assert _map_to_plex_path("/mnt/data/media/tv/S.srt") == "/mnt/data/media/tv/S.srt"


@pytest.mark.asyncio
async def test_no_request_when_url_or_token_missing():
    with patch("app.services.plex_notifier.get_setting",
               side_effect=_settings(plex_url="", plex_token="")):
        with patch("app.services.plex_notifier.httpx.AsyncClient") as client:
            await notify_plex_library_refresh("/mnt/data/media/tv/x.sr.srt")
            client.assert_not_called()


def _mock_client(get_mock):
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.get = get_mock
    return client


@pytest.mark.asyncio
async def test_refreshes_only_the_matching_section():
    sections_res = MagicMock(status_code=200, text=SECTIONS_XML)
    refresh_res = MagicMock(status_code=200, text="")
    get_mock = AsyncMock(side_effect=[sections_res, refresh_res])

    with patch("app.services.plex_notifier.get_setting", side_effect=_settings()):
        with patch("app.services.plex_notifier.httpx.AsyncClient",
                   return_value=_mock_client(get_mock)):
            await notify_plex_library_refresh("/mnt/data/media/tv/Show/ep.sr.srt")

    urls = [c.args[0] for c in get_mock.call_args_list]
    assert urls[0].endswith("/library/sections")
    assert urls[1].endswith("/library/sections/2/refresh")
    assert len(urls) == 2, "should not refresh sections that did not change"


@pytest.mark.asyncio
async def test_falls_back_to_all_sections_when_path_unmatched():
    sections_res = MagicMock(status_code=200, text=SECTIONS_XML)
    get_mock = AsyncMock(side_effect=[sections_res] + [MagicMock(status_code=200, text="")] * 3)

    with patch("app.services.plex_notifier.get_setting", side_effect=_settings()):
        with patch("app.services.plex_notifier.httpx.AsyncClient",
                   return_value=_mock_client(get_mock)):
            await notify_plex_library_refresh("/unmapped/path/ep.sr.srt")

    refreshed = [c.args[0] for c in get_mock.call_args_list if "refresh" in c.args[0]]
    assert len(refreshed) == 3


@pytest.mark.asyncio
async def test_network_failure_is_swallowed():
    """A media server being down must never fail a job whose subtitle already published."""
    get_mock = AsyncMock(side_effect=Exception("connection refused"))
    with patch("app.services.plex_notifier.get_setting", side_effect=_settings()):
        with patch("app.services.plex_notifier.httpx.AsyncClient",
                   return_value=_mock_client(get_mock)):
            await notify_plex_library_refresh("/mnt/data/media/tv/Show/ep.sr.srt")
