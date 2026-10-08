# v0 prompt — frontend for the team sheet generator

Paste the fenced block below into v0.dev as one message, unedited. Everything outside the fence is context for you, not for v0.

```
Build a single-page React app, "Team Sheet Generator", for a competitive Pokemon VGC research tool (Champions VGC 2026 Regulation Set M-B). The user pins one Pokemon, a local model builds full six-Pokemon teams around it, and the user then battles each team to measure how it actually performs.

THE RULE THAT OUTRANKS EVERY OTHER INSTRUCTION HERE
The finding this tool exists to communicate is that generated teams came out legal, novel and on-style while their win rate was bad. The interface must never imply a team is good before it has been measured.
- Each build method carries its measured baseline win rate inline, in the control itself, not in a tooltip: graft 0.420, invent 0.093. Show "a real untouched tournament team = 0.475" as the number to beat.
- One shared measurement axis, drawn once above the results, spanning 0.00 to 0.60, tick-marked at those three baselines. Every team's win-rate bar sits on that same axis, so a team nobody has battled is visibly empty on it next to one that has been.
- Never render a bare win rate. Always the number, its interval and its battle count together: "0.438 +/- 0.061 over 96 battles", where the interval is 1.96 * se from the API.
- "Not measured" is an explicit worded state, not a zero, a dash or a skeleton: "Not battled yet - this team has no win rate until you measure it."

API. This backend already exists and serves the app from the same origin. Do not mock it and do not invent fields.
GET /api/meta -> { species: [{key, name, count}], styles: string[], baselines: [{key, label, value}], corpus: number, opponents: number }
POST /api/build { pin: species key, style: string, method: "graft" | "invent", n: number } -> { teams: [{ paste: string, slots: [{species, item, ability, nature, moves: string[], evs: string, pinned: boolean}] }] } or { error: string }
POST /api/measure { paste: string, opponents: number, battles: number } -> { win_rate: number, se: number, battles: number } or { error: string }. It runs real battles and takes about 8 seconds for 96 battles (opponents 12 x battles 8). Keep the rest of the UI usable while it runs and let several teams measure at once.

Fetch /api/meta once on mount. Every Pokemon name, every playstyle option, the baselines and the counts come from that response. Hardcode no Pokemon name and no species count anywhere: a name the API did not return must never appear on screen, placeholders included.

Layout. Sticky header with the title and a readout built from /api/meta: "Reg M-B - {corpus} teams - {opponents} opponents". Under it a two-column grid, a sticky ~320px control column and a results column, collapsing to one column below 860px.

Controls, left column:
1. Pin a candidate. A searchable combobox over meta.species, each row showing its count as "in N of {corpus} teams", in the order the API returns (most common first). A typed name that matches nothing gets an inline note under the field, not a toast; Build stays disabled until the text resolves to a key. Send the key, display the name.
2. Build method. Two radio cards. "Graft onto a real team" - swaps your pick into a real tournament team, measured baseline 0.420. "Invent all six" - the model writes the whole sheet, measured baseline 0.093.
3. Playstyle. A select over meta.styles: none, trickroom, rain, sun, sand, snow, tailwind. Render "none" as "any style".
4. Number of teams, 1 to 6, default 4.
5. Build button.

States, each distinct and each worded: idle (nothing built, one line saying what will happen), building (sampling teams and checking each one against Showdown's validator, several seconds), built-unmeasured, measuring (per team, never a global spinner), measured, error (show the server's error string verbatim with a retry control).

Team card. Numbered "Sheet 01". Six rows, one per Pokemon: "Species @ Item" on the first line, then ability and nature, then the four moves, then the stat spread. Mark the pinned Pokemon's row so the user can see the one they asked for. Two regulation facts you must respect: this format has NO Tera type, so never render a Tera field; and the spread is called Stat Points, not EVs (max 32 per stat, 66 total). The API sends the spread pre-formatted, like "8 HP / 10 Atk / 32 Spe" - print it as given, never parse it or re-total it. A "Copy paste" button copies team.paste verbatim to the clipboard with a transient confirmation.

Measurement rail, on each card: the win rate, its interval and battle count, the bar positioned on the shared axis, and a Measure button posting {paste, opponents: 12, battles: 8}. Compare the result against the graft baseline of 0.420 and mark at-or-above versus below with a word or a symbol as well as colour.

Stack and constraints. React with Tailwind and shadcn/ui. No chart library - the axis and the bars are plain divs. One page, no routing. Dark and light themes, dark by default, both fully legible. Full keyboard access: arrow-navigable combobox, visible focus ring on every control, and the measured result announced through an aria-live region so it is heard, not only seen. Every number on screen originates in an API response or in the three baselines above; invent no data, no sample teams and no fake win rates for empty states.
```

## What the backend already does (so you can tell v0's output from the working parts)

Already built and running, at `/tmp/vgc-pilot/gui/server.py` (mirror: `gui/server.py` in this repo):

- Serves all three endpoints above on `http://localhost:8770` and serves the current hand-written UI at `/`.
- `/api/meta` is live and returns 189 species entries, 692 corpus teams, 50 opponents, the 7 styles, and the 3 baselines. The species list is the corpus's, not the dex's — take the count from the response, never from this file.
- `graft` swaps a real corpus set for the pinned species into a real tournament team; `invent` samples all six slots from the diffusion checkpoint with the pin held fixed.
- Every emitted team is checked by the real Showdown `TeamValidator` (a persistent node subprocess) before it is returned, so anything the frontend shows is already legal.
- `/api/measure` plays the team against the first N of the top-50 meta teams with the behaviour-cloning policy on both sides and returns wins/battles.

What v0 is being asked for is the frontend only. The existing `gui/static/index.html` is the reference implementation of the same contract — keep it as the fallback until v0's output measures a team end to end.
