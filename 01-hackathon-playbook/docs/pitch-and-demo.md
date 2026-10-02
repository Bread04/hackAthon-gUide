# 🎤 Pitch Resources

<!-- markdownlint-disable MD013 -->

> Structures, timings, video specs and rehearsal drills. Verified 2026-09-21.

---

## The 3-minute pitch skeleton

| Segment | Time | Content | Evidence |
| --- | --- | --- | --- |
| 🪝 Hook + problem | 0:00–0:20 | Make the judge *feel* the frustration. A real user quote works well | [JetBrains](https://blog.jetbrains.com/ai/2026/06/how-to-win-a-hackathon-notes-from-the-judging-table/), [ETHGlobal](https://ethglobal.com/events/tokyo2026/info/details) (≤20 s intro) |
| 💡 One-sentence solution | 0:20–0:30 | `"<Product> lets <person> <do thing> in <time>."` | [Devpost](https://info.devpost.com/blog/6-tips-for-making-a-hackathon-demo-video) |
| 🖥️ Live demo | 0:30–2:00 | The success-signal journey. **Working by 0:90** | JetBrains |
| ⚙️ How it works | 2:00–2:30 | Architecture in ≤4 bullets; name the sponsor tech | ETHGlobal (≤4 bullets/slide), [Devpost criteria](https://info.devpost.com/blog/understanding-hackathon-submission-and-judging-criteria) |
| 🚀 Impact + close | 2:30–3:00 | Who uses it, what's next, a memorable last line | Devpost criteria (impact) |

**Variants:** MLH science fair is ~3 min including questions, so cut the demo to 60 s. ETHGlobal finals are 4 min + 3 min Q&A, so add depth to "how it works".

---

## Slide deck (5–7 slides max)

1. **Title:** name, one-line pitch, team
2. **Problem:** one image and one quote
3. **Demo:** switch to the live app, or embed the video as a fallback
4. **How it works:** a diagram and ≤4 bullets
5. **Sponsor tech:** logos plus exactly what each API does in your product
6. **Impact / next:** who, how many, what you'd build next
7. *(optional)* **Team**

Keep one idea per slide with large visuals and no walls of text.

---

## Demo video specification

| Spec | Value |
| --- | --- |
| Length | MLH digital ≤ 2 min · Devpost usually < 3 min · ETHGlobal 2–4 min |
| Resolution | ≥ 720p (ETHGlobal) |
| Intro | ≤ 20 s; **say the hackathon name** at the start (MLH) |
| Narration | **Your own voice.** No TTS or AI voiceover (ETHGlobal); write your own script (Devpost) |
| Footage | Screen recording (OBS, etc.); no phone recording; don't speed up; **cut out waiting** |
| Upload | YouTube (unlisted or public), marked "Not for Kids"; copy the link before the upload finishes |
| Timing | Start ≥ 2–3 h before the deadline |

Sources: [ETHGlobal](https://ethglobal.com/events/tokyo2026/info/details) · [MLH rules](https://github.com/MLH/mlh-policies/blob/main/standard-hackathon-rules.md) · [Devpost](https://info.devpost.com/blog/6-tips-for-making-a-hackathon-demo-video)

### Recording workflow

1. Reset the demo state (fresh seed data, logged-in demo user).
2. Record the screen in one take per segment; retakes are cheap.
3. Record narration separately if that's easier, then line it up while editing.
4. Cut out every loading spinner and pause.
5. Add a title card (event name) and an end card (repo and demo URL).
6. Export at 1080p, upload, paste the link into the submission, and let it finish processing.

### Bonus: a 1-command launch video with `/brag`

[**latent-spaces/brag**](https://github.com/latent-spaces/brag) 🆓 (MIT, 6.9k ⭐, last push 2026-09-21) is a Claude Code skill that turns your project into a **short launch video with music, motion and share copy**. Run it inside your repo, and you get a `brag-output/` folder with the plan, a composition brief, share copy and `brag.mp4`. The rendering is done by [Hyperframes](https://hyperframes.heygen.com/).

```bash
# Claude Code
/plugin marketplace add latent-spaces/brag
/plugin install brag@brag
# any other agent (Cursor, Codex, Copilot, Gemini CLI, opencode…)
npx skills add https://github.com/latent-spaces/brag --skill brag
```

Then ask your agent `let's /brag`, or steer the tone: `/brag --tone "fake Series A launch from 2016"`.

**Needs:** Node.js 22+, **FFmpeg** on your `PATH`, and the Hyperframes CLI (`npx hyperframes doctor` checks it). **Install and test it the week before**, not at 3 a.m. On Windows, clone with `git clone -c core.symlinks=true`, or copy `skills/brag/` into `~/.claude/skills/` by hand.

> ⚠️ **It's a hype video, not your demo video.** Use it for the end card, a Devpost gallery clip, or your LinkedIn/X post after the event ([`after-the-event.md`](after-the-event.md)). The **judged** demo video still needs a real screen recording of your working app, in the format your event specifies (table above). Leave voiceover **off** (the default): `--voice` adds AI narration, which ETHGlobal bans. Mention it in your AI-use disclosure.

---

## Rehearsal drills

| Drill | How |
| --- | --- |
| **Timed run ×3** | Use a stopwatch. If you're over, cut the `[CUT IF LONG]` lines |
| **Cold judge** | Someone who hasn't seen the project watches once and explains it back. If they can't, clarify the pitch |
| **Q&A drill** | `PROMPTS.md` → P2; memorise 15 s answers to the top 5 questions |
| **Failure drill** | Practise switching to the backup video mid-demo in under 5 seconds |

---

## Story templates (fill in the brackets)

Use one of these for the hook and close; keep every number real.

- **Struggle to solution:** "Meet [person], a [role] who [specific daily struggle]. Every [period] they spend [time] on [task], but [frustration]. So we built [project]: it [one sentence]. When we tested it with [N] people here, [specific result]. Next we [step]."
- **Before / after:** "Until now, [user] had two options: [bad option A] or [bad option B]. With [project] they can [benefit] in [time comparison]."
- **What if:** "[Real fact or short story]. What if [the change your project makes]? [Project] does it by [how]. The result: [specific metric]."
- **Story spine (for the problem):** "Once upon a time [person and world]. Every day [status quo]. Until one day [trigger]. Because of that [consequence]. Until finally [your project]."

Name a specific person, not "users"; one concrete detail beats a big claim.

## Booth vs stage

| | Booth (walk-up judging) | Stage (timed pitch) |
| --- | --- | --- |
| Length | 3-5 min conversation | Fixed slot, strict timer |
| Plan | Open with the hook, then **let the judge steer**: tech judge → architecture; business judge → users and impact | Follow the skeleton above to the second |
| Props | Demo running on loop, QR code or card with the URL, a printed diagram | Slides + live demo + backup video |
| Main risk | Rambling or a judge walking away mid-setup | Running over time |

## Judge questions: answer patterns

| Question | Pattern that works |
| --- | --- |
| "How is this different from X?" | Acknowledge the similarity → name the specific user or case X doesn't serve → show the feature that proves it → evidence (who you tested with) |
| "How would you scale it?" | Name the one bottleneck and the design choice that handles it; don't promise millions of users |
| "What's the business model?" | One plausible model and who pays; "we'd validate it by…" is fine |
| "What's the biggest risk?" | A real risk plus your mitigation; "none" is a red flag |
| "What would you do with more time?" | Two or three priorities and why, not a feature wish-list |
| "How did you split the work?" | Each person's part in one sentence each |
| "What did you learn?" | One honest surprise and what you'd change |

Rehearse these with [`PROMPTS.md`](../PROMPTS.md) → P2 (or `DT14` for datathons).

## Behaviours judges mark down

- Running over time, or skipping the demo to talk about it
- Excuses ("with more time…") instead of owning a focused scope
- Arguing with feedback; say "Good point — here's how we'd handle it"
- Overpromising traction you don't have
- Not knowing your own numbers or architecture
- Criticising other teams

If the demo breaks: stay calm, switch to the backup video within seconds ([rehearsal drills](#rehearsal-drills)), and explain what it shows.

---

## What kills pitches (named judges)

- **Ambiguity:** judges can't tell what the product does
- **A backend-heavy project with no UI**
- **A barely changed template**
- **A rehashed or overly simple idea**
- **Over-indexing on one criterion**
- **Big teams with unequal contribution**

Source: [Devpost: hackathon judging tips](https://info.devpost.com/blog/hackathon-judging-tips)

---

## Further reading

- [JetBrains: How to win a hackathon, notes from the judging table (2026-06)](https://blog.jetbrains.com/ai/2026/06/how-to-win-a-hackathon-notes-from-the-judging-table/)
- [Devpost: Understanding submission & judging criteria](https://info.devpost.com/blog/understanding-hackathon-submission-and-judging-criteria)
- [Devpost: 6 tips for making a hackathon demo video](https://info.devpost.com/blog/6-tips-for-making-a-hackathon-demo-video)
- [MLH Organizer Guide: Judging plan](https://guide.mlh.com/general-information/judging-and-submissions/judging-plan)

---

_Adapted in part from [Hackathon Starter Pack](https://github.com/udaysharmadev/Hackathon-Starter-Pack-Complete-Guide-Roadmap) by Uday Sharma (MIT licence; see [`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md)). Practitioner advice, not research._
