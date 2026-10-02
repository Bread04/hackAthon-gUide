# Virtual/remote hackathon logistics, team formation, and live-demo reliability

Method note: WebSearch worked. WebFetch was egress-blocked for info.devpost.com, help.devpost.com, medium.com, blog.jetbrains.com, guide.mlh.com and blog.colosseum.com. Only github.com/opportunity-hack was fetched in full. Everything else is SNIPPET-ONLY (search-result summaries, not page reads). Search summaries are AI-paraphrased, so exact numbers should be re-checked on the source page.

## How do virtual/hybrid hackathons work in 2025-2026 (Devpost requirements, Discord/Slack, async judging)?

### Takeaway
Submission requirements are set per hackathon on its Rules page. The common pattern is a demo video capped at about 3 minutes, a code repo with README spin-up instructions or a hosted demo URL, and sometimes a diagram and write-up. Recorded videos let judges review asynchronously.

### Cited Findings
- SNIPPET-ONLY: Demo video limits vary by event: about 4 min (All Things Agentic), about 5 min (ClimateHacks), max 3 min (OpenAI Build Week), under 3 min (PayPal AI Hackathon), 1-3 min (PyTorch 2021). Always read the specific Rules page. — [Devpost listings found via search: All Things Agentic](https://allthingsagentichackathon.devpost.com/), [OpenAI Build Week](https://openai.devpost.com/updates/45282-openai-build-week-submissions-are-open-plugin-launch), [PayPal AI](https://paypalaihackathon.devpost.com/), [ClimateHacks](https://climatehacks.devpost.com/), [PyTorch 2021 rules](https://pytorch2021.devpost.com/rules)
- SNIPPET-ONLY: Repo requirement is a URL to a public or private repo (GitHub/GitLab/Bitbucket) with spin-up instructions in the README. If the repo is private, share access with testing@devpost.com and the organizers. Judges must be able to run a working build, via either setup instructions or a hosted demo URL. — [Perplexity Hackathon rules (Devpost)](https://perplexityhackathon.devpost.com/rules) and other Devpost rules pages in the same search; the exact per-page wording was not confirmed
- SNIPPET-ONLY: Typical submissions are a demo video, repo, architecture diagram and short write-up. — [Devpost template submission](https://devpost.com/software/example-template-submission)
- Opportunity Hack's Spring 2026 retrospective cited two reasons for requiring recorded video demos for Fall 2026: live demos "eat too much time" and "exclude remote judges". The issue says "Recorded videos let judges review async and at scale". It pairs this with replacing DevPost. — [opportunity-hack/ops issue #8](https://github.com/opportunity-hack/ops/issues/8) (fetched in full)
- SNIPPET-ONLY: Devpost's judging dashboard lists the criteria and shows judge progress. Judges get an automatic email with access, and can stop and resume. Judges often watch the video first for context, then test the submission. — [Devpost online judging help](https://help.devpost.com/hc/en-us/articles/360021798712-Online-judging-How-judging-works-for-judges) (not fetched), [Devpost judging video](https://www.youtube.com/watch?v=kpV6-T0KB40)
- SNIPPET-ONLY: Many hackathons run Discord with channels such as #team-formation. — [ServiceNow CreatorCon teammate post](https://www.servicenow.com/community/creatorcon-blogs/find-your-creatorcon-hackathon-teammates/ba-p/2910396)
- SNIPPET-ONLY: MLH publishes a "Judging Plan" page in its organizer guide, under Judging and Submissions. Its contents were not read. — [MLH Judging Plan](https://guide.mlh.com/general-information/judging-and-submissions/judging-plan)

### Inferences
- Because judging is asynchronous, the video and the README or hosted link are effectively the whole submission. The live demo is optional or a secondary check.
- A private repo should be shared with the named reviewer accounts before the deadline.

### Gaps
- Exact Devpost wording on "testing instructions" (login credentials, test accounts) was not confirmed, because help.devpost.com was blocked.
- I found no confirmed 2025-2026 data on Slack mentor setups, or on MLH hybrid-format specifics.

## Evidence-backed practices for finding teammates, roles, and remote communication

### Takeaway
Advice here is practitioner opinion, not research. It converges on three points: start early in the hackathon's Discord, cover distinct skill roles, and keep the team to 3-5 people.

### Cited Findings
- SNIPPET-ONLY: Discord team-formation channels are popular. Searching early beats being assigned last-minute. Teammates can also come from your existing network, social posts, or former teammates. — [ServiceNow CreatorCon](https://www.servicenow.com/community/creatorcon-blogs/find-your-creatorcon-hackathon-teammates/ba-p/2910396), [TechTogether Medium](https://medium.com/techtogether/5-ways-to-conquer-your-fears-of-finding-a-team-at-a-hackathon-5e14525ceda3)
- SNIPPET-ONLY: Diversify roles: front-end, full-stack, project/product management, design. Ideal team size is 3-5 (fewer than 3 risks burnout, more than 5 adds coordination overhead). — [WowDAO LinkedIn tips](https://www.linkedin.com/pulse/5-tips-form-a-list-team-worldwide-ai-hackathon-wowdaoai)
- SNIPPET-ONLY: Virtual events are text-heavy, so miscommunication is the main risk. Spell out the problem statement. Respect time zones and communication styles. — [Built In virtual hackathon](https://builtin.com/articles/virtual-internal-hackathon), [WowDAO LinkedIn](https://www.linkedin.com/pulse/5-tips-form-a-list-team-worldwide-ai-hackathon-wowdaoai)
- SNIPPET-ONLY: Devpost has an official "Teamwork tips for online hackathons" article. Its content was not read. — [Devpost Help Center](https://help.devpost.com/article/79-teamwork-tips-for-online-hackathons)
- SNIPPET-ONLY: In the Upstate Interactive tips, one person narrates and one drives the demo. — [Upstate Interactive](https://medium.com/upstate-interactive/8-tips-to-a-successful-hackathon-demo-and-presentation-4d1ae83415ad)
- SNIPPET-ONLY: A serial winner's thread on selecting hackathon teammates (22 wins claimed). Self-reported, content unread. — [jia on X](https://x.com/jia_seed/status/1956547397653528808)

### Inferences
- A sensible role split for a 3-5 person remote team is: builder(s) for front end, builder(s) for back end or integration, one demo/video owner who is also the presenter, and one person who owns submission logistics (README, repo access, Devpost form).
- A written one-paragraph problem statement and a time zone table in the team channel are cheap mitigations for the text-only miscommunication risk.

### Gaps
- I found no peer-reviewed or survey evidence on team formation outcomes. All of the above is practitioner advice, and the sources are mostly blog posts or vendor content.
- The Devpost teamwork article and the MLH team material were unreadable (blocked).

## Common demo-day failures and standard mitigations

### Takeaway
Network, API and auth failures are the usual cause. The standard mitigations are a recorded backup, local or cached data, a hosted fallback, and repeated dry runs. These are practitioner consensus. I found no quantitative failure statistics.

### Cited Findings
- SNIPPET-ONLY: "Record a backup video (networks fail)." Test the demo about 10 minutes before presenting ("networks fail, APIs go down, have backups"). Practice 10+ times. — [Upstate Interactive 8 tips](https://medium.com/upstate-interactive/8-tips-to-a-successful-hackathon-demo-and-presentation-4d1ae83415ad), [Szeyu Sim, How I Win Most Hackathons](https://szeyusim.medium.com/how-i-win-most-hackathons-stories-pro-tips-from-a-serial-hacker-1969c6470f92)
- SNIPPET-ONLY: Shared venue Wi-Fi gets unreliable when a room is on social media. Use a wired connection if possible. Organizers should keep 3-5 personal hotspots as contingency. If unsure about connectivity, run locally or use a screencast or GIFs. — [AngelHack organizer guide](https://angelhack.com/blog/how-to-organize-a-hackathon/), [Slice WiFi hackathon post (vendor)](https://www.slicewifi.com/blog/hackathons), [NYC hackathon logistics](https://www.nyc.gov/html/hackathon/html/logistics.html)
- SNIPPET-ONLY: Teams often discover rate limits, auth issues or broken endpoints during the presentation. Test API keys 24-48 hours before. Cache responses, pre-fetch sample data as fallback, and space requests (the cited figure was 100-200 ms). Prefer simple API-key auth over complex OAuth. — [APILayer blog](https://blog.apilayer.com/10-apis-that-will-make-your-hackathon-project-stand-out/) (the snippet may be a paraphrase of this post, so confirm before quoting)
- SNIPPET-ONLY: Demo checklists on GitHub issues from real hackathon repos list "backup demo video", device check, QR code, and scripted backup as tasks. These show the practice, not evidence of its effect. — [chiang-mai-hackathon #13](https://github.com/Schummers/chiang-mai-hackathon/issues/13), [ellipsis #75](https://github.com/Alexlgvcode/ellipsis/issues/75), [3d-pcb-puzzle #35](https://github.com/havido/3d-pcb-puzzle/issues/35)
- Live demos "eat too much time" and exclude remote judges, which is why one organizer moved to recorded video. — [opportunity-hack/ops #8](https://github.com/opportunity-hack/ops/issues/8)

### Inferences
- Failure modes in the question not directly sourced (OAuth redirect URIs pointing at localhost, hosting cold starts, screen-share glitches, deploy breakage) are standard engineering reasoning but UNVERIFIED as hackathon-specific evidence. My search returned only generic OAuth localhost-callback examples and nothing on cold starts.
- Mitigation checklist inferred from the sources and engineering practice:
  - Register the production redirect URI before demo day.
  - Pre-warm the server or ping it right before the demo.
  - Seed the data.
  - Keep a feature-flagged offline or mock mode.
  - Freeze deploys before the deadline.
  - Keep a status or health check page.
  - Rehearse screen sharing on the actual platform.
  - Keep the backup video open in a tab.

### Gaps
- No sourced statistics on failure frequency, and no page found specifically about cold starts or screen-share problems at hackathons.
- Several of the most relevant pages (Devpost, JetBrains, Colosseum) could not be read.

## What makes a good 2-3 minute demo video

### Takeaway
Organizers and judges say to keep the video within the cap (usually 3 min), open with a quick overview, show the working product fast, and make sure the audio and screen are clear. Judges watch many submissions back to back.

### Cited Findings
- SNIPPET-ONLY: Colosseum workshop recap: presentation video should be 2-3 min; product demo videos no more than 3 min. Start with a quick overview because judges review multiple projects back to back. The technical demo should be direct, cover core features and the tech stack, and explain prioritization. Judges must be able to clearly see, hear and access the video. Mistakes: exceeding 3 min, flashy visuals with little substance, buzzwords, omitting team background, and not explaining the core idea and impact. — [Colosseum blog](https://blog.colosseum.com/perfecting-your-hackathon-submission/)
- SNIPPET-ONLY: Show something working within about 90 seconds. Put the judge in the user's shoes and walk through what happens, even if the product is not polished. Pause for pacing. Narrate what is on screen. — [Devpost: 6 tips for a winning demo video](https://info.devpost.com/blog/6-tips-for-making-a-hackathon-demo-video), [Devpost: advice from 5 seasoned judges](https://info.devpost.com/blog/hackathon-judging-tips). Which statements come from which page was not confirmed.
- SNIPPET-ONLY: Judges spend 3-5 minutes per project, and a polished UI with no user path is a liability. This came from the search summary of Colosseum-related content. The source for the 3-5 minutes figure is unclear (possibly the Colosseum recap or a dev.to post), so treat it as UNVERIFIED. — [Colosseum guide](https://blog.colosseum.com/how-to-win-a-colosseum-hackathon/)
- SNIPPET-ONLY: JetBrains judging-table notes (June 2026) exist, contents unread. — [JetBrains blog](https://blog.jetbrains.com/ai/2026/06/how-to-win-a-hackathon-notes-from-the-judging-table/)

### Inferences
- A workable structure for 2:30: 0:00-0:20 problem and one-line solution; 0:20-1:50 live walkthrough of the core user path; 1:50-2:20 how it works (stack, key technical choice); 2:20-2:30 impact and what is next. This is derived from the cited advice, not an organizer-published template.
- Audio: use an external mic or headset, record in a quiet room, and listen to the export before uploading. Judges said they need to hear it clearly (Colosseum), but no sourced specifics on microphones or loudness.

### Gaps
- No organizer-sourced numbers on audio quality or captions, and I could not read the Devpost 6-tips page in full.

## Named sources (URLs)
1. Devpost judging tips, 5 judges: https://info.devpost.com/blog/hackathon-judging-tips (snippet-only)
2. Devpost, 6 tips for a demo video: https://info.devpost.com/blog/6-tips-for-making-a-hackathon-demo-video (snippet-only)
3. Devpost Help, teamwork tips for online hackathons: https://help.devpost.com/article/79-teamwork-tips-for-online-hackathons (snippet-only)
4. Colosseum, perfecting your submission: https://blog.colosseum.com/perfecting-your-hackathon-submission/ (snippet-only)
5. JetBrains, notes from the judging table (June 2026): https://blog.jetbrains.com/ai/2026/06/how-to-win-a-hackathon-notes-from-the-judging-table/ (snippet-only)
6. MLH Hackathon Organizer Guide, Judging Plan: https://guide.mlh.com/general-information/judging-and-submissions/judging-plan (snippet-only)
7. Opportunity Hack ops issue #8, recorded demos: https://github.com/opportunity-hack/ops/issues/8 (fetched)
8. AngelHack, how to organize a hackathon: https://angelhack.com/blog/how-to-organize-a-hackathon/ (snippet-only)
9. Upstate Interactive, 8 tips for a hackathon demo: https://medium.com/upstate-interactive/8-tips-to-a-successful-hackathon-demo-and-presentation-4d1ae83415ad (snippet-only)
10. Devpost rules pages (Perplexity, OpenAI Build Week, PayPal AI): https://perplexityhackathon.devpost.com/rules (snippet-only)
