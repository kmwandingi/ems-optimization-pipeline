# ─────────────────────────────────────────────────────────────────────────────
# draggable_window.py — draggable confirmation overlay (aligned & pre-positioned)
# ─────────────────────────────────────────────────────────────────────────────
"""
This component draws a rounded, draggable window that sits directly ON TOP of
the schedule cells and streams its start hour back to Streamlit.

Key guarantees
• The iframe containing the component is given `position:relative; z-index:999`,
  so it always stacks above the schedule row that precedes it.
• `box-sizing:border-box` on the window means its 2 px border is INCLUDED in
  the width we programmatically set, so the window’s edges line up exactly with
  the outer borders of the hour-cells (which have a 1 px border each side).
"""

from streamlit.components.v1 import html as components_html


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def render_draggable_window(
    device_name: str,
    duration: int,
    initial_hour: int,
):
    """
    Parameters
    ----------
    device_name   : str   unique tag (used for DOM ids)
    duration      : int   number of contiguous hours the window covers
    initial_hour  : int   left-hand start hour when first rendered (0-based)

    Returns
    -------
    int | None    : latest start hour chosen by the user, or None
    """
    dom = device_name.replace(" ", "_")
    html_code = f"""
    <style>
      /* ——— Iframe itself (raised above schedule row) ——— */
      :host, :root {{
        position: relative;
        z-index: 999;
      }}

      /* ——— Wrapper pulls iframe up so it overlaps the buttons ——— */
      #wrap-{dom} {{
        margin-top: -46px;      /* 44 px row height + 2 px gap            */
        height: 44px;
        pointer-events: none;   /* transparent by default                 */
      }}
      #wrap-{dom} * {{ pointer-events: auto; }}

      /* ——— Invisible 24-column grid keeps geometry in sync ——— */
      #grid-{dom} {{
        display: grid;
        grid-template-columns: repeat(24, 1fr);
        width: 100%;
        height: 44px;
      }}

      /* ——— Draggable window ——— */
      #win-{dom} {{
        position: absolute; top: 0;
        height: 44px; border-radius: 8px;
        background: rgba(6, 182, 212, .25);
        border: 2px solid var(--secondary-color);
        box-sizing: border-box;         /* include border in programmed width */
        cursor: grab;
        display: flex; justify-content: space-between;
      }}
      #win-{dom} .grip {{
        width: 6px; height: 100%;
        background: var(--secondary-color);
        cursor: ew-resize;
      }}
    </style>

    <div id="wrap-{dom}">
      <div id="grid-{dom}"></div>
      <div id="win-{dom}">
        <div class="grip"></div><div class="grip"></div>
      </div>
    </div>

    <script>
      (function() {{
        /* constants */
        const DUR  = {duration};
        const COLS = 24;
        const MAX  = COLS - DUR;

        /* handles */
        const grid = document.getElementById("grid-{dom}");
        const win  = document.getElementById("win-{dom}");

        /* geometry helpers */
        let idx   = {initial_hour};
        let cellW = 0;
        const recalc = () => {{
          cellW = grid.getBoundingClientRect().width / COLS;
          win.style.width = (DUR * cellW) + "px";
          win.style.left  = (idx * cellW) + "px";
        }};
        addEventListener("load",   recalc);
        addEventListener("resize", recalc);

        /* drag */
        let down = false, sx = 0, sl = 0;
        win.addEventListener("pointerdown", e => {{
          down = true; sx = e.clientX; sl = idx * cellW;
          win.setPointerCapture(e.pointerId);
        }});
        const finish = e => {{
          if (!down) return;
          down = false; win.releasePointerCapture(e.pointerId);
          Streamlit.setComponentValue(idx);   /* report new start-hour   */
        }};
        win.addEventListener("pointerup",     finish);
        win.addEventListener("pointercancel", finish);
        win.addEventListener("pointermove", e => {{
          if (!down) return;
          const ni = Math.round((sl + (e.clientX - sx)) / cellW);
          idx      = Math.max(0, Math.min(MAX, ni));
          win.style.left = (idx * cellW) + "px";
        }});

        /* set iframe height once */
        Streamlit.setFrameHeight(44);
      }})();
    </script>
    """

    return components_html(html_code, height=60)
