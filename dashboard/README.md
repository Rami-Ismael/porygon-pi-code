# Mask Lab

Selected design: X-ray's mask slider and compact controls, Orbit's glowing opponent stars, and Bench's legality and performance metrics. The dashboard is a self-contained static HTML artifact built from recorded experiments. It makes no model requests.

Run `python3 code/dashboard/build.py`, then `python3 -m http.server 8767 --bind 127.0.0.1 --directory code/dashboard` from the repository root. Open `http://127.0.0.1:8767/`.

Use the family, starting-team and mask-size controls to choose a real trial. Switch between Ling and diffusion to inspect fills. All four battle scores remain visible when available. Unsupported inputs and rejected outputs remain explicit. Stars select opponent records. Expand the cost notebook to compare measured usage, cold-cache and maximum-output scenarios.

Headline gains give each starting team equal weight, conditional on legal Ling completions. They are not overall success probabilities. The builder uses the completed full-regulation v2 comparison when available, otherwise the original seven-supported-task run. Three starting teams are too few to establish an overall winner.

Motion: frequent model/mask changes render immediately. Revealed masks use a 200ms opacity/transform transition; reduced motion removes displacement. Stars glow statically to encode wins, without perpetual decorative animation. Keyboard focus is preserved on rerender. The grid scrolls horizontally on narrow screens.

Source data lives in the gitignored `code/results` directories. `build.py` verifies both methods used identical battle configurations and checks cost reconstruction against all 63 usage records. Reports: `../docs/llm-cost-audit.md` and `../docs/diffusion-baseline-comparison.md`.
