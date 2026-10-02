# 🖨️ Hackathon Cheat Sheet (Software)

<!-- markdownlint-disable MD013 -->

> One page to print or keep open. Full map: [`WORKFLOW.md`](WORKFLOW.md). Datathon version: [`08-datathon-handbook/00-start-here/cheatsheet.md`](08-datathon-handbook/00-start-here/cheatsheet.md).

## The 8 phases (24 h event; double only the Build phase for 48 h)

| # | Phase | When | Done when… |
| - | --- | --- | --- |
| 0 | Prepare | Days before | Everyone ran hello-world; accounts, keys and credits ready |
| 1 | Rules | H+0 to 0:30 | AI-use, old-code, team and submission rules written down |
| 2 | Frame and choose | to H+2:30 | One-sentence problem, named user, 3 non-goals, sponsor track picked |
| 3 | Set up and deploy | to H+4 | **Live URL** a teammate opened on another network |
| 4 | Build | to ~H+17 | Core flow works end to end on the live URL |
| 5 | Freeze | ~H+17 to 20 | No new features; seeded data; **backup video recorded** |
| 6 | Pitch and submit | H+20 to 23 | Timed run under the limit; README, video, form, repo access done |
| 7 | Learn | After | Keys rotated, retro written, sponsors followed up |

## The 10 rules

1. **Read the rules first**: AI disclosure, pre-existing code, team size, submission format.
2. **Deploy in the first hours**, even if it only says hello.
3. **One user, one problem, one core flow** of four steps or fewer.
4. **Secrets go in `.env`**, never in the repo; proxy API keys through your backend.
5. **Keep an AI-use log** (tool, purpose, files) and put it in the README.
6. **Use a sponsor's tech deeply in one feature**, not lightly in ten.
7. **Every external call gets a timeout, a loading state and a fallback.**
8. **Freeze features about 3 hours before the deadline.** Fix bugs only after that.
9. **Record the backup video as soon as the demo first works.**
10. **Sleep.** Rested teams make fewer bugs and better pitches.

## The 3-minute pitch

| Time | Say / show |
| --- | --- |
| 0:00-0:20 | The person and their problem (a real quote if you have one) |
| 0:20-0:30 | "\<Product\> lets \<person\> \<do thing\> in \<time\>." |
| 0:30-2:00 | **Live demo** of the core flow; working by 1:30 |
| 2:00-2:30 | How it works in at most 4 bullets; name the sponsor tech |
| 2:30-3:00 | Impact, what's next, a memorable last line |

**Judge Q&A:** "different from X?" → name the user X doesn't serve and show the feature. "Biggest risk?" → a real one plus your mitigation. Never: "nothing like this exists".

## Before you present (10 minutes)

- [ ] Live URL works on phone data, logged out and logged in
- [ ] API keys valid, quota left, fallback data ready
- [ ] Backup video open in a tab; notifications off; tabs pre-loaded
- [ ] Wired or hotspot connection; screen sharing tested on the real platform
- [ ] Roles: presenter, demo driver, Q&A

**When stuck:** [`03-backend/docs/troubleshooting.md`](03-backend/docs/troubleshooting.md) · **Before the demo:** [`03-backend/docs/demo-day-testing.md`](03-backend/docs/demo-day-testing.md) · **Pitch:** [`pitch-and-demo.md`](01-hackathon-playbook/docs/pitch-and-demo.md)
