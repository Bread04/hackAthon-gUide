# MVP & Demo-Readiness Checklist

<!-- markdownlint-disable MD013 -->

> Tick these in order. Anything unticked at the feature freeze (about 3 hours before the deadline) is cut, not rushed. Items are our suggestions, drawn from the rest of this guide.

## 1. MVP scope (hour 0-4)

- [ ] One user, one problem, one core flow of 4 steps or fewer
- [ ] Must-have / nice-to-have / cut lists written down
- [ ] The "AHA" moment is defined and demoable in 30 seconds

## 2. Deployed early (by the end of hour 1-2)

- [ ] "Hello world" is live on a public URL
- [ ] Environment variables are in the host's settings, not in the repo
- [ ] A teammate other than the author opened the live URL on a different network

## 3. Feature freeze

- [ ] No new features; bug fixes only
- [ ] Seed data loaded, so the demo never starts from an empty screen
- [ ] External API keys tested within the last 24 hours; responses cached or mocked as fallback
- [ ] Mobile / small-screen check, keyboard navigation and contrast spot-check (see [`rules-hardware-a11y-remote.md`](../docs/rules-hardware-a11y-remote.md))

## 4. Submission package

- [ ] README follows [`project-readme.md`](project-readme.md): what, why, how to run, tech, AI-use disclosure
- [ ] Repo access works for judges (public, or shared as the rules require)
- [ ] Demo video recorded (check the event's length cap) and audio verified
- [ ] Rules re-read: AI disclosure, pre-existing work, licences

## 5. Demo day

- [ ] Backup: recorded video, screenshots and a saved-data mode
- [ ] Wired connection or hotspot; tested screen sharing on the real platform
- [ ] Two full rehearsals with a timer; roles assigned (presenter, driver, Q&A)
- [ ] Tabs pre-opened and logged in; notifications off
