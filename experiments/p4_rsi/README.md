# p4_rsi: recursive tool improvement (claim 3)
- **Claim tested:** 3. Tools the agent builds make its later edits better.
- **Design:** 3 rounds of C2 on rotating tasks. Each round's saved tools and SAEs are merged into the library for the next round. Then the library snapshots L0..L3 are evaluated on held-out tasks, with nothing saved during held-out runs merged back.
- **Primary metric:** held-out task score as a function of library round, at fixed budget.
- **Success threshold:** L3 beats L0 on both held-out tasks.
- **Training-signal stretch (4.3):** run `python -m wsp.export` on successful trajectories to get an SFT set for a small open model. This step is gated on budget.
