# Plant Simulation model: control logic

Plant Simulation 16.1 (German object library), frame `.Modelle.Modell` of the board-game model `Gameboard.spp`
(not included: the model file belongs to the course material). One file per Method in `methods/`; the file name is
the Method name. SimTalk 2.0, one statement per line. The Python side is `src/pnpssp/plantsim_com.py`.

Changes after the version used in the course (marked *new*): `GetCurrentState` also writes the games going around
by language (`loop_DE`, `loop_EN`, `loop_ES`), which `reset` adds as columns 15–17 of `CurrentState`; the model
with returning games (`src/pnpssp/recirculation.py`) needs them. Games that have already merged into
`MainConveyor` on their way back are not counted, as for `len_loop`.

## A. Clean-up (delete from the frame)
Objects: `Quelle` (the orphan without name, interval 0).
Tables: `States`, `Actions`.
Methods: `robotControl`, `CalcCurrentState`, `Output_Control`, `resetCurrentState`, `resetActionControl`,
`FlushWhenFull_Buf1`, `FlushWhenFull_Buf2`, `FeedSenke`, `Throughputtime_Station1`, `Throughputtime_Station2`.
Keep `ExportModel` and `ModelDump` (useful for checking).

## B. Dialog settings
| Object | Tab | Setting |
|---|---|---|
| `Paintingstation` | Zeiten | Bearbeitungszeit **20 s** (was 1:40) |
| `Paintingstation` | Steuerungen | Eingang and Ausgang **empty** |
| `BindingStation` | Zeiten | Bearbeitungszeit **20 s** (was 50/75/100 by language; to keep a language dependence, register `BindingTime` as Eingang) |
| `BindingStation` | Steuerungen | Eingang: `BindingTime` (optional) / Ausgang empty |
| `Montagestation` | Attribute | **Hauptteil-Vorgänger: 2** (Förderstrecke1 = manuals). The manual becomes the main MU and carries the language in `Name`; the board is mounted and consumed |
| `Montagestation` | Steuerungen | Ausgang: `assembleGame`, **vor Aktionen** ✔ |
| `PickAndPlace` | Steuerungen | Eingang **empty**; Ausgang: `pnpDecision`, **vor Aktionen** ✔ |
| `bufferID1`, `bufferID2` | Steuerungen | Eingang empty; Ausgang: `holdInBuffer`, vor Aktionen ✔ |
| `bufferID1`, `bufferID2` | Attribute | Kapazität 10, any buffer type (Queue recommended) |
| `Return_to_mainCon` | Steuerungen | Ausgang **empty**; connect with a Kante to `MainConveyor` |
| `Förderstrecke8`, `Förderstrecke9` | Steuerungen | Ausgang **empty**; connect each with a Kante to `MainConveyor` |
| `Senke` | Steuerungen | Eingang **empty** (counting is done by the statistic `StatAnzahlEin`) |
| `Ereignisverwalter` | Steuerungen | Ende: 0 (no limit); Echtzeit: off |
| `SystemLoad` (table) | contents | Häufigkeit: DE 0.4, EN 0.35, ES 0.25 |

The three new connectors (Return_to_mainCon, Förderstrecke8, Förderstrecke9 → MainConveyor) replace
`feedToConveyor` and `FeedSenke`: Plant Simulation handles the blocking at the merge, without code.

## C. Tables
`CurrentState`: TableFile with **Spaltenindex activated**, 1 row; `reset` writes the headers.
`ActionControl`: TableFile with Spaltenindex activated, column `action` (string), 1 row; `reset` initialises it.

## D. Variables (Variable objects in the frame)
`GoalGames` : integer = 1000 — finished games that end the episode.
`SeedRL`    : integer = 1    — seed written by Python before `reset`.

## E. Methods
| Method | Registered at | Role |
|---|---|---|
| `reset` | — (called by Python) | resets simulation, tables, seed; runs to the first decision |
| `GetCurrentState` | — (called by `pnpDecision`; optionally by Python) | writes the state row, including the loop contents by language (*new*) |
| `pnpDecision` | PickAndPlace / Ausgang, vor Aktionen | decision point: state, event_id + 1, stops the event controller |
| `AIControl` | — (called by Python) | applies A / B / LOOP, eviction, flush to the Senke, resumes |
| `assembleGame` | Montagestation / Ausgang, vor Aktionen | checks the language (`Name`) and colours the game |
| `holdInBuffer` | bufferID1 and bufferID2 / Ausgang, vor Aktionen | empty: keeps the games until the flush |
| `BindingTime` | BindingStation / Eingang (optional) | binding time by language |

## F. Protocol (as in the docstring of `src/pnpssp/plantsim_com.py`)
1. `pnpDecision`: a game is ready to leave the robot → writes `CurrentState`, `event_id += 1`,
   `Ereignisverwalter.stop`. The game waits in the robot.
2. Python reads the row, writes `ActionControl["action",1]` and executes `AIControl`.
3. `AIControl`: clears the flags, applies the action to `PickAndPlace.BE(1)`, writes
   `did_loop / n_evicted / completed_batch`, `Ereignisverwalter.start`.
4. The simulation runs until the next game reaches the robot → step 1.
`goal_state = 1` when `Senke.StatAnzahlEin >= GoalGames`.

## G. Test without Python
Run `reset` by hand: the simulation runs and stops with a game in the robot and `event_id = 1` in `CurrentState`.
Write "A" into `ActionControl["action",1]` and run `AIControl`: the game goes to `bufferID1` and the simulation
continues to the next game. Repeat with "B" and "LOOP". After 10 equal games in a buffer,
`Senke.StatAnzahlEin` rises by 10 and `completed_batch = 1`.
