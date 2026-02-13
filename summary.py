"""Shared summary calculations used by Summary & Planning and Outputs pages."""

from math import ceil

PALLET_CUBE = 50


def compute_summary(df, fluff, trailer_capacity):
    """Compute summary metrics from the cube data.

    Returns a dict with keys:
        total_cube, total_cube_with_fluff, pallet_count,
        trailer_goal, probable_trailers
    """
    total_cube = df["TOTAL"].sum()
    total_cube_with_fluff = total_cube + fluff
    pallet_count = ceil(total_cube / PALLET_CUBE) if total_cube else 0
    trailer_goal = total_cube_with_fluff / trailer_capacity if trailer_capacity else 0
    probable_trailers = ceil(trailer_goal * 1.1) if trailer_goal else 0

    return {
        "total_cube": total_cube,
        "total_cube_with_fluff": total_cube_with_fluff,
        "pallet_count": pallet_count,
        "trailer_goal": trailer_goal,
        "probable_trailers": probable_trailers,
    }
