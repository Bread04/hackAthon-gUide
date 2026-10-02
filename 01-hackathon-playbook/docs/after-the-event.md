# 🏁 After the Event: A Step-by-Step Manual

<!-- markdownlint-disable MD013 -->

> **In plain English:** the hackathon is over, win or lose. These steps protect your accounts, keep your project worth showing, and turn the weekend into something that helps your next event and your CV.
>
> **When:** Step 1 the same day. Everything else within a week, while you still remember.

---

## Step 1 · Lock down your keys (same day, 15 minutes)

Keys get pasted into chats, screenshots and AI prompts during a hackathon. Treat them all as leaked.

- [ ] **Rotate every API key** you used (model API, Supabase service key, Clerk secret, any sponsor keys): make a new one in each dashboard, then delete the old one
- [ ] Update the new values in Vercel / Railway → **Redeploy** (if you're keeping the demo online)
- [ ] Delete keys and tokens you gave to MCP servers and tools you won't use again
- [ ] **Set spending limits** or delete paid resources (GPU apps, paid databases, trial credits running out)
- [ ] Remove teammates' access to accounts that were shared "just for the weekend"

Why: [`../../03-backend/docs/security.md`](../../03-backend/docs/security.md) and [`../../05-tools-and-mcp/docs/mcp-security.md`](../../05-tools-and-mcp/docs/mcp-security.md) both say to rotate everything after the event.

## Step 2 · Keep the demo alive (30 minutes)

A dead link on your CV is worse than no link.

- [ ] **Supabase free projects pause after 7 days idle.** Either accept that (and say "restore on request" in the README) or visit the app occasionally
- [ ] Free hosts that sleep take about a minute to wake. Note that in the README
- [ ] Make sure the **video link** works. It never breaks, so it's your permanent demo

## Step 3 · Clean up the repo (1 hour)

- [ ] Finish the README using [`../templates/project-readme.md`](../templates/project-readme.md): live link, video, screenshot, "what's real and what's mocked", team credits
- [ ] Check no secrets are in the history. If one is, you already rotated it in Step 1
- [ ] Add a `LICENSE` if the event or your team wants the code open source (MIT is the simplest)
- [ ] **Pin the repo** on your GitHub profile (profile → Customize your pins)
- [ ] If your team used a personal account, **transfer the repo to a GitHub organisation** or make sure everyone is credited in the README

## Step 4 · Tell people about it (1 hour)

- [ ] Update the **Devpost** page with anything you rushed (screenshots, a clearer description)
- [ ] Post on **LinkedIn** or X. Pick one shape: **story** (the user and the moment), **technical** (one hard problem and how you solved it), **reflection** (what you learned), or **achievement** (the result, with credit to the team). Include the problem, one GIF, what you learned, the prize if you won, and tag your teammates and the sponsors whose tech you used. Want a slick launch clip? Run [`/brag`](https://github.com/latent-spaces/brag) in your repo ([`pitch-and-demo.md`](pitch-and-demo.md) → Bonus)
- [ ] **Follow up with sponsors and judges** you talked to, within 48 hours: *"Thanks for judging at <event>. We built <project> with your <API>. Here's the demo: <link>."* Sponsor engineers are often recruiters
- [ ] Add it to your CV with the formula **"Built [project] using [stack] to [solve problem for whom], resulting in [measurable result]"**, plus your own part ("I built the auth and real-time pipeline"). Examples by type:
  - *AI feature:* "Built a lecture-notes summariser with Python and an LLM API for 3 study groups; cut review time from 2 h to 30 min in our test; won Best AI Hack (of 60 teams)."
  - *Hardware:* "Built an ESP32 air-quality monitor with an MQTT dashboard; ran on battery for the whole 24 h demo; won the hardware prize."
  - *Data:* "Built a churn model and what-if dashboard (LightGBM, Streamlit) with leak-free grouped CV; PR-AUC 0.48 vs 0.11 baseline."
  Use only numbers you measured.

> **Capture during the event** so you can write this later: project name, one-line problem, full stack, your specific role, live URL, 5+ screenshots or GIFs (one every few hours shows progress), teammates and their roles, any measured number, and what went wrong and how you adapted.

## Step 5 · Harvest the lessons (1 hour, within 48 hours)

Follow [`../battle-plan.md`](../battle-plan.md) → **"After the event: harvest the lessons"**. In short:

1. Write down the prompts that worked first time, and the ones that took 5 tries.
2. Save the setup you debugged at 3 a.m. (auth, deploy, database) somewhere reusable.
3. Note every mistake your AI assistant kept making, as pitfalls for your next project's `CLAUDE.md` ([`agent-context.md`](agent-context.md)).
4. Advanced: run `/bmad-retrospective` ([`../../07-bmad-workflow/`](../../07-bmad-workflow/README.md)).

## Step 6 · Decide the project's future

| If… | Then |
| --- | --- |
| Real people asked to use it | Keep going. Rebuild the mocked parts properly (see "Production-grade reference" in [`../../03-backend/docs/hackathon-patterns.md`](../../03-backend/docs/hackathon-patterns.md)) |
| It was fun but that's it | Archive it with a good README. It's done its job |
| A sponsor showed interest | Reply fast, and ask what they'd need to see next |

---

## Quick checklist

- [ ] Keys rotated · spending limits set
- [ ] Demo link and video work
- [ ] README finished · repo pinned
- [ ] Posted · sponsors thanked
- [ ] Lessons written down

---

_Adapted in part from [Hackathon Starter Pack](https://github.com/udaysharmadev/Hackathon-Starter-Pack-Complete-Guide-Roadmap) by Uday Sharma (MIT licence; see [`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md)). Practitioner advice, not research._
