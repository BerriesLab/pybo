import matplotlib.colors as mcolors


def diverging_norm(z_vals, center: float | None):
    """Normalize for a diverging colormap (coolwarm) over `z_vals`.

    Plain min/max scaling when no center is given - the colormap's midpoint then
    falls wherever the data's range happens to average to, which is fine when the
    user has no meaningful reference value in mind. With a center, TwoSlopeNorm
    pins that value to the colormap's neutral midpoint; vmin/vmax are widened to
    include it since TwoSlopeNorm requires vmin <= vcenter <= vmax.
    """
    vmin, vmax = min(z_vals), max(z_vals)
    if center is None:
        return mcolors.Normalize(vmin=vmin, vmax=vmax)
    vmin = min(vmin, center)
    vmax = max(vmax, center)
    if vmin == vmax:
        # A single-valued (or all-equal) range has nothing either side of the
        # center to diverge across; fall back to a degenerate linear norm rather
        # than let TwoSlopeNorm reject vmin == vcenter == vmax.
        return mcolors.Normalize(vmin=vmin, vmax=vmax)
    # TwoSlopeNorm needs strict vmin < vcenter < vmax. All the data sitting on one
    # side of the center is exactly when the widened vmin or vmax lands exactly on
    # it - nudge that one edge out by a hair rather than pass the boundary through,
    # which still puts every point on the correct (single) side of the colormap.
    eps = (vmax - vmin) * 1e-6
    if vmin == center:
        vmin -= eps
    if vmax == center:
        vmax += eps
    return mcolors.TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)


def mark_center(cbar, center: float | None) -> None:
    """Draw a line across `cbar` at `center`, if there is one to mark.

    A colorbar's long axis is in the norm's own data units - vmin to vmax - so a
    plain axhline at the value lands in the right place with no extra mapping.
    Without this the centred colormap's neutral band reads as "no change from
    some value", with nothing on the bar itself saying which.
    """
    if center is None:
        return
    cbar.ax.axhline(center, color="black", linewidth=1.0, zorder=5)