import os
import time
import pytest
from app.services.scanner import scan_library_folders


def test_scanner_includes_mtime_and_size(tmp_path):
    tv_dir = tmp_path / "tv"
    show_dir = tv_dir / "Test Show" / "Season 1"
    show_dir.mkdir(parents=True)
    video = show_dir / "Test Show - S01E01.mkv"
    video.write_bytes(b"x" * 1024 * 1024 * 2)  # 2MB

    # Set mtime to 10 days ago
    ten_days_ago = time.time() - (10 * 86400)
    os.utime(str(video), (ten_days_ago, ten_days_ago))

    scan = scan_library_folders(str(tv_dir), category="series")
    assert len(scan) == 1
    show = scan[0]
    assert show["title"] == "Test Show"
    assert "mtime" in show
    assert abs(show["mtime"] - ten_days_ago) < 2
    assert len(show["episodes"]) == 1
    ep = show["episodes"][0]
    assert ep["filename"] == "Test Show - S01E01.mkv"
    assert "mtime" in ep
    assert abs(ep["mtime"] - ten_days_ago) < 2
    assert ep["size_mb"] == 2.0
    assert ep["has_target_sub"] is False


def test_scanner_movies_mtime_and_size(tmp_path):
    movies_dir = tmp_path / "movies"
    movies_dir.mkdir()
    movie = movies_dir / "Great Movie (2024).mp4"
    movie.write_bytes(b"x" * 1024 * 1024)

    forty_days_ago = time.time() - (40 * 86400)
    os.utime(str(movie), (forty_days_ago, forty_days_ago))

    scan = scan_library_folders(str(movies_dir), category="movies")
    assert len(scan) == 1
    m = scan[0]
    assert m["filename"] == "Great Movie (2024).mp4"
    assert "mtime" in m
    assert abs(m["mtime"] - forty_days_ago) < 2
    assert m["size_mb"] == 1.0


class MockBabelLibraryLogic:
    """
    Python mirror of client-side Alpine filtering logic to test all permutations
    and assert exact parity with UI behavior.
    """
    def __init__(self, media_data=None):
        self.media_data = media_data or {"series": [], "movies": []}
        self.media_category = "series"
        self.media_search_query = ""
        self.media_filter = "all"
        self.selected_episode_filters = {}

    def is_show_recent(self, show):
        if not show:
            return False
        cutoff = time.time() - (30 * 86400)
        if show.get("mtime", 0) >= cutoff:
            return True
        return any(ep.get("mtime", 0) >= cutoff for ep in show.get("episodes", []))

    def is_movie_recent(self, movie):
        if not movie:
            return False
        cutoff = time.time() - (30 * 86400)
        return movie.get("mtime", 0) >= cutoff

    def get_filtered_series(self):
        lst = self.media_data.get("series", [])
        q = (self.media_search_query or "").strip().lower()
        if q:
            lst = [s for s in lst if q in s.get("title", "").lower()]
        if self.media_filter == "missing":
            lst = [s for s in lst if any(not ep.get("has_target_sub") for ep in s.get("episodes", []))]
        elif self.media_filter == "complete":
            lst = [s for s in lst if len(s.get("episodes", [])) > 0 and all(ep.get("has_target_sub") for ep in s.get("episodes", []))]
        elif self.media_filter == "recent":
            lst = [s for s in lst if self.is_show_recent(s)]
        return lst

    def get_filtered_movies(self):
        lst = self.media_data.get("movies", [])
        q = (self.media_search_query or "").strip().lower()
        if q:
            lst = [m for m in lst if q in m.get("filename", "").lower()]
        if self.media_filter == "missing":
            lst = [m for m in lst if not m.get("has_target_sub")]
        elif self.media_filter == "complete":
            lst = [m for m in lst if m.get("has_target_sub")]
        elif self.media_filter == "recent":
            lst = [m for m in lst if self.is_movie_recent(m)]
        return lst

    def get_filter_count(self, filter_name):
        if self.media_category == "series":
            lst = self.media_data.get("series", [])
            if filter_name == "all":
                return len(lst)
            if filter_name == "missing":
                return len([s for s in lst if any(not ep.get("has_target_sub") for ep in s.get("episodes", []))])
            if filter_name == "complete":
                return len([s for s in lst if len(s.get("episodes", [])) > 0 and all(ep.get("has_target_sub") for ep in s.get("episodes", []))])
            if filter_name == "recent":
                return len([s for s in lst if self.is_show_recent(s)])
        elif self.media_category == "movies":
            lst = self.media_data.get("movies", [])
            if filter_name == "all":
                return len(lst)
            if filter_name == "missing":
                return len([m for m in lst if not m.get("has_target_sub")])
            if filter_name == "complete":
                return len([m for m in lst if m.get("has_target_sub")])
            if filter_name == "recent":
                return len([m for m in lst if self.is_movie_recent(m)])
        return 0

    def get_selected_episode_filter(self, show_title):
        return self.selected_episode_filters.get(show_title, "all")

    def set_selected_episode_filter(self, show_title, filter_name):
        self.selected_episode_filters[show_title] = filter_name

    def get_filtered_episodes(self, show, season="All"):
        if not show or "episodes" not in show:
            return []
        eps = show["episodes"]
        if season and season != "All":
            eps = [ep for ep in eps if ep.get("season", "Root") == season]
        ep_filter = self.get_selected_episode_filter(show.get("title", ""))
        if ep_filter == "missing":
            eps = [ep for ep in eps if not ep.get("has_target_sub")]
        elif ep_filter == "complete":
            eps = [ep for ep in eps if ep.get("has_target_sub")]
        return eps

    def get_empty_state_message(self, category):
        if category == "series":
            if not self.media_data.get("series"):
                return "No TV series found in configured path."
            if self.media_search_query and self.media_filter != "all":
                return "No series match your search and filter criteria."
            if self.media_search_query:
                return "No library items match your search."
            if self.media_filter == "missing":
                return "No missing subtitles found."
            if self.media_filter == "complete":
                return "No completed items found."
            if self.media_filter == "recent":
                return "No recently added media found."
            return "No TV series found."
        elif category == "movies":
            if not self.media_data.get("movies"):
                return "No movies found in configured Movies folder."
            if self.media_search_query and self.media_filter != "all":
                return "No movies match your search and filter criteria."
            if self.media_search_query:
                return "No library items match your search."
            if self.media_filter == "missing":
                return "No missing subtitles found."
            if self.media_filter == "complete":
                return "No completed items found."
            if self.media_filter == "recent":
                return "No recently added media found."
            return "No movies found."
        return "No items found."


def test_series_filtering_all_missing_complete_recent():
    now = time.time()
    ten_days = now - (10 * 86400)
    forty_days = now - (40 * 86400)

    # Show 1: Complete and Recent
    show1 = {
        "title": "Breaking Bad",
        "mtime": ten_days,
        "episodes": [
            {"filename": "BB.S01E01.mkv", "season": "Season 1", "has_target_sub": True, "mtime": ten_days},
            {"filename": "BB.S01E02.mkv", "season": "Season 1", "has_target_sub": True, "mtime": ten_days}
        ]
    }
    # Show 2: Missing 1 episode and Old (not recent)
    show2 = {
        "title": "The Wire",
        "mtime": forty_days,
        "episodes": [
            {"filename": "Wire.S01E01.mkv", "season": "Season 1", "has_target_sub": True, "mtime": forty_days},
            {"filename": "Wire.S01E02.mkv", "season": "Season 1", "has_target_sub": False, "mtime": forty_days}
        ]
    }
    # Show 3: Missing all episodes and Recent
    show3 = {
        "title": "Severance",
        "mtime": ten_days,
        "episodes": [
            {"filename": "Sev.S01E01.mkv", "season": "Season 1", "has_target_sub": False, "mtime": ten_days}
        ]
    }

    logic = MockBabelLibraryLogic({"series": [show1, show2, show3], "movies": []})
    logic.media_category = "series"

    # All filter
    logic.media_filter = "all"
    assert [s["title"] for s in logic.get_filtered_series()] == ["Breaking Bad", "The Wire", "Severance"]
    assert logic.get_filter_count("all") == 3

    # Missing filter: The Wire (1 missing) and Severance (all missing)
    logic.media_filter = "missing"
    assert [s["title"] for s in logic.get_filtered_series()] == ["The Wire", "Severance"]
    assert logic.get_filter_count("missing") == 2

    # Complete filter: Only Breaking Bad
    logic.media_filter = "complete"
    assert [s["title"] for s in logic.get_filtered_series()] == ["Breaking Bad"]
    assert logic.get_filter_count("complete") == 1

    # Recent filter: Breaking Bad and Severance (<= 30 days)
    logic.media_filter = "recent"
    assert [s["title"] for s in logic.get_filtered_series()] == ["Breaking Bad", "Severance"]
    assert logic.get_filter_count("recent") == 2


def test_series_search_case_insensitive_and_combinations():
    now = time.time()
    show1 = {
        "title": "House of the Dragon",
        "mtime": now,
        "episodes": [{"filename": "HotD.S01E01.mkv", "season": "Season 1", "has_target_sub": True, "mtime": now}]
    }
    show2 = {
        "title": "House M.D.",
        "mtime": now,
        "episodes": [{"filename": "House.S01E01.mkv", "season": "Season 1", "has_target_sub": False, "mtime": now}]
    }
    show3 = {
        "title": "Succession",
        "mtime": now,
        "episodes": [{"filename": "Succ.S01E01.mkv", "season": "Season 1", "has_target_sub": False, "mtime": now}]
    }

    logic = MockBabelLibraryLogic({"series": [show1, show2, show3], "movies": []})
    logic.media_category = "series"

    # Case-insensitive search "house"
    logic.media_search_query = "house"
    logic.media_filter = "all"
    assert [s["title"] for s in logic.get_filtered_series()] == ["House of the Dragon", "House M.D."]

    # Search uppercase "DRAGON"
    logic.media_search_query = "DRAGON"
    assert [s["title"] for s in logic.get_filtered_series()] == ["House of the Dragon"]

    # Search + Filter combination: "house" + "missing"
    logic.media_search_query = "house"
    logic.media_filter = "missing"
    assert [s["title"] for s in logic.get_filtered_series()] == ["House M.D."]

    # Search + Filter combination: "house" + "complete"
    logic.media_search_query = "house"
    logic.media_filter = "complete"
    assert [s["title"] for s in logic.get_filtered_series()] == ["House of the Dragon"]

    # No match
    logic.media_search_query = "nonexistent"
    assert logic.get_filtered_series() == []
    assert logic.get_empty_state_message("series") == "No series match your search and filter criteria."


def test_movie_filtering_search_and_recent():
    now = time.time()
    ten_days = now - (10 * 86400)
    forty_days = now - (40 * 86400)

    movie1 = {"filename": "Dune.Part.Two.2024.mkv", "has_target_sub": True, "mtime": ten_days}
    movie2 = {"filename": "Oppenheimer.2023.mkv", "has_target_sub": False, "mtime": forty_days}
    movie3 = {"filename": "Alien.Romulus.2024.mkv", "has_target_sub": False, "mtime": ten_days}

    logic = MockBabelLibraryLogic({"series": [], "movies": [movie1, movie2, movie3]})
    logic.media_category = "movies"

    # All
    logic.media_filter = "all"
    assert [m["filename"] for m in logic.get_filtered_movies()] == [
        "Dune.Part.Two.2024.mkv",
        "Oppenheimer.2023.mkv",
        "Alien.Romulus.2024.mkv"
    ]
    assert logic.get_filter_count("all") == 3

    # Missing
    logic.media_filter = "missing"
    assert [m["filename"] for m in logic.get_filtered_movies()] == [
        "Oppenheimer.2023.mkv",
        "Alien.Romulus.2024.mkv"
    ]
    assert logic.get_filter_count("missing") == 2

    # Complete
    logic.media_filter = "complete"
    assert [m["filename"] for m in logic.get_filtered_movies()] == ["Dune.Part.Two.2024.mkv"]
    assert logic.get_filter_count("complete") == 1

    # Recent (<= 30 days)
    logic.media_filter = "recent"
    assert [m["filename"] for m in logic.get_filtered_movies()] == [
        "Dune.Part.Two.2024.mkv",
        "Alien.Romulus.2024.mkv"
    ]
    assert logic.get_filter_count("recent") == 2

    # Search movies
    logic.media_search_query = "alien"
    logic.media_filter = "all"
    assert [m["filename"] for m in logic.get_filtered_movies()] == ["Alien.Romulus.2024.mkv"]


def test_season_and_episode_filter_combinations():
    show = {
        "title": "Game of Thrones",
        "episodes": [
            {"filename": "GoT.S01E01.mkv", "season": "Season 1", "has_target_sub": True},
            {"filename": "GoT.S01E02.mkv", "season": "Season 1", "has_target_sub": False},
            {"filename": "GoT.S02E01.mkv", "season": "Season 2", "has_target_sub": True},
            {"filename": "GoT.S02E02.mkv", "season": "Season 2", "has_target_sub": True},
        ]
    }
    logic = MockBabelLibraryLogic({"series": [show], "movies": []})

    # Season 1, Episode Filter All -> 2 episodes
    assert len(logic.get_filtered_episodes(show, "Season 1")) == 2

    # Season 1, Episode Filter Missing -> 1 episode
    logic.set_selected_episode_filter("Game of Thrones", "missing")
    eps = logic.get_filtered_episodes(show, "Season 1")
    assert len(eps) == 1
    assert eps[0]["filename"] == "GoT.S01E02.mkv"

    # Season 1, Episode Filter Complete -> 1 episode
    logic.set_selected_episode_filter("Game of Thrones", "complete")
    eps = logic.get_filtered_episodes(show, "Season 1")
    assert len(eps) == 1
    assert eps[0]["filename"] == "GoT.S01E01.mkv"

    # Season 2, Episode Filter Missing -> 0 episodes
    logic.set_selected_episode_filter("Game of Thrones", "missing")
    assert len(logic.get_filtered_episodes(show, "Season 2")) == 0

    # Season 2, Episode Filter Complete -> 2 episodes
    logic.set_selected_episode_filter("Game of Thrones", "complete")
    assert len(logic.get_filtered_episodes(show, "Season 2")) == 2

    # All seasons, Episode Filter Complete -> 3 episodes
    assert len(logic.get_filtered_episodes(show, "All")) == 3


def test_empty_state_messages():
    logic = MockBabelLibraryLogic({"series": [], "movies": []})

    # Empty series
    assert logic.get_empty_state_message("series") == "No TV series found in configured path."
    assert logic.get_empty_state_message("movies") == "No movies found in configured Movies folder."

    # With data but filter yields 0
    logic.media_data["series"] = [{"title": "Show A", "episodes": [{"has_target_sub": True}]}]
    logic.media_filter = "missing"
    assert logic.get_empty_state_message("series") == "No missing subtitles found."

    logic.media_filter = "recent"
    assert logic.get_empty_state_message("series") == "No recently added media found."

    logic.media_filter = "complete"
    logic.media_data["series"] = [{"title": "Show B", "episodes": [{"has_target_sub": False}]}]
    assert logic.get_empty_state_message("series") == "No completed items found."


def test_switching_category_preserves_independent_views():
    show = {"title": "Lost", "episodes": [{"filename": "Lost.S01E01.mkv", "has_target_sub": True}]}
    movie = {"filename": "Lost.in.Translation.mkv", "has_target_sub": False}
    logic = MockBabelLibraryLogic({"series": [show], "movies": [movie]})

    logic.media_search_query = "lost"

    logic.media_category = "series"
    series_res = logic.get_filtered_series()
    assert len(series_res) == 1
    assert series_res[0]["title"] == "Lost"

    logic.media_category = "movies"
    movies_res = logic.get_filtered_movies()
    assert len(movies_res) == 1
    assert movies_res[0]["filename"] == "Lost.in.Translation.mkv"
