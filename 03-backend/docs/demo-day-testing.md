# 🧪 Demo-Day Testing: Test the 20% Judges Will See

<!-- markdownlint-disable MD013 -->

> You will not write a full test suite in 24 hours. Test what would embarrass you on stage, triage the rest, and make every external API fail gracefully. When something is already broken, use [`troubleshooting.md`](troubleshooting.md); for the final run-through, use [`mvp-and-demo-checklist.md`](../../01-hackathon-playbook/templates/mvp-and-demo-checklist.md).

## What to test, what to skip

| Always test | Usually skip |
| --- | --- |
| The exact demo path, end to end, several times | Edge cases nobody will hit in the demo |
| Sign-up, log-in, log-out, staying logged in | Browsers beyond Chrome plus one other |
| Save, refresh, data still there | Performance tuning unless it is visibly slow |
| What the user sees when an API fails or input is invalid | Internationalisation, exhaustive logging |
| The deployed URL on a different network and a phone | Pixel-perfect layouts on every screen size |

Rule of thumb: **if a bug appearing during the demo would embarrass you, test it.** Keep basic accessibility checks ([section 3](../../01-hackathon-playbook/docs/rules-hardware-a11y-remote.md#3-accessibility-privacy-and-responsible-ai-cheap-evidence-covers-most-of-what-judges-score)); they take minutes and judges notice.

## Bug triage

| Priority | Examples | Action |
| --- | --- | --- |
| **P0** | Crash, data loss, security hole, login broken | Fix now |
| **P1** | Core feature broken, wrong data shown, broken navigation | Fix before the freeze |
| **P2** | Minor UI glitches, slow pages, rare edge cases | Only if time allows |
| **P3** | Cosmetic issues, features not in the demo | Ignore or note in the README |

Fix or work around? If judges will see it and the fix takes minutes, fix it. If it takes hours and you can avoid that path in the demo, work around it (seeded data, disabled button, a mock) and say so honestly if asked. After the feature freeze, every fix risks a new bug; ship known minor issues.

## The seven ways an external API kills a demo

| Failure | Prevention |
| --- | --- |
| **Silent timeout:** the call hangs and the app looks broken | Set a timeout (around 8 s), show a loading state immediately, keep cached results ready |
| **Rate limit (429)** after several rehearsals | Know the quota; budget calls for every rehearsal plus the demo; cache responses |
| **Missing env var** in production | Check every `process.env` / `import.meta.env` value is set on the host; fail fast at startup if one is missing |
| **CORS error** in the browser | Test from the production URL early; call third-party APIs through your own backend route |
| **Exposed key** visible in the browser's network tab | Never ship keys to the front end; proxy through a server function |
| **Unexpected response shape** renders `undefined` | Log one real response, then map it to the shape your UI expects |
| **Free tier runs out** mid-event | Check quota resets (daily vs monthly) before the event; have a fallback provider or saved responses |

## API checklist (2 hours before presenting)

- [ ] Every call works against the real endpoint from the deployed app
- [ ] Rehearsals plus the demo fit inside the rate limits
- [ ] Every call has a loading state and a friendly error state
- [ ] Fallback data is ready if a provider is slow or down
- [ ] The demo path responds in a few seconds
- [ ] Tested on a network other than the one you built on

## A tiny automated check (optional)

One script that hits your deployed health endpoint and the demo path catches the "worked an hour ago" breakage:

```bash
curl -fsS https://your-app.example.com/api/health && echo "health OK"
```

Run it after every deploy and right before you present.

---

_Adapted in part from [Hackathon Starter Pack](https://github.com/udaysharmadev/Hackathon-Starter-Pack-Complete-Guide-Roadmap) by Uday Sharma (MIT licence; see [`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md)). Practitioner advice, not research._
