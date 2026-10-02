# Hardware tracks, solo participation, and university hackathon rules

Method note: the sandbox blocked WebFetch/curl to devpost.com, guide.mlh.com, stanforddaily.com, hackathon sites and personal blogs (EGRESS_BLOCKED / proxy 403). raw.githubusercontent.com worked, so every item marked FULL-TEXT below was read in full from GitHub. Everything else comes from search-engine summaries and is marked SNIPPET-ONLY. Where a snippet could not be checked against a primary page, it is also marked UNVERIFIED.

## 1. Hardware/IoT tracks: published judging criteria and winners

### Takeaway
No major event I could read publishes a separate hardware rubric. Hardware prizes are usually judged on the same general criteria, sometimes with a hardware gloss: MLH's Design criterion explicitly means "human-computer interaction" for hardware. Hardware-first events such as StarkHacks weight functionality, technical complexity and impact. In winner READMEs, the hardware-prize projects share three traits: (a) a physical device with a clear sensing-to-actuation loop, (b) software or AI layered on cheap sensors or microcontrollers, and (c) an accessibility, safety or health use case, or an "invisible thing made visible" demo.

### Cited Findings
**Published criteria**
- MLH standard rules (also used for MLH categories at in-person events, which includes MLH "Best Hardware Hack"): four criteria, weighted equally: Technology ("Did the technology involved make you go 'Wow'?"), Design ("For a hardware project, it might be more about how good the human-computer interaction is (e.g. is it easy to use or does it use a cool interface?)"), Completion ("Does the hack work?"), and Learning. The criteria explicitly do NOT include code quality, pitch quality, idea novelty or "how well the project solves a problem". "Pitches and presentations are discouraged... you'll only hurt yourself by not showing a demo." FULL-TEXT — [MLH standard-hackathon-rules.md](https://github.com/MLH/mlh-policies/blob/main/standard-hackathon-rules.md)
- The MLH rules also say criteria "guide judges but ultimately judges are free to make decisions based on their gut feeling." FULL-TEXT — [MLH standard-hackathon-rules.md](https://github.com/MLH/mlh-policies/blob/main/standard-hackathon-rules.md)
- MLH Hardware Lab: MLH brings microcontrollers, components and smart devices (e.g. Raspberry Pi, Arduino) to a limited number of North American member events. Hosts must provide a volunteer and a dedicated hardware table. SNIPPET-ONLY — [MLH Hardware Lab Contents](https://guide.mlh.io/organizer-resources/hardware-lab-contents); MLH also has a hardware-hackathon organiser guide with a prizes page: [guide.mlh.com/hardware-hackathon-guide/prizes](https://guide.mlh.com/hardware-hackathon-guide/prizes) (not readable here).
- TreeHacks 2025 "Best Hardware Hack": honours "excellence in integrating hardware components... from IoT devices to robotics... projects that leverage physical computing elements to create innovative solutions and interactive experiences." SNIPPET-ONLY — [TreeHacks 2025 Devpost](https://treehacks-2025.devpost.com/). Overall TreeHacks judging is on "creativity, technological complexity, and social impact." SNIPPET-ONLY, UNVERIFIED — same source.
- StarkHacks (Purdue; bills itself as "the world's largest hardware hackathon"; 36 hours; $100,000 in prizes): judged on Functionality and Execution, Technical Complexity, and Problem Relevance and Impact. SNIPPET-ONLY — [StarkHacks Devpost](https://starkhacks.devpost.com/), [starkhacks.com](https://starkhacks.com/); Purdue ECE news about a Guinness World Record attempt (Jan 2026): [Purdue ECE](https://engineering.purdue.edu/ECE/News/2026/purdue-students-aim-to-set-guinness-world-record-with-worlds-largest-hardware-hackathon) (SNIPPET-ONLY). The 2026 event dates appear as 4/17–4/19 in a [UW ECE advising post](https://advisingblog.ece.uw.edu/2026/01/08/purdue-university-hackathon-invite-4-17-to-4-19) (SNIPPET-ONLY).
- Hackaday Prize (2023 official rules; the latest found, since no 2025 rules surfaced): five categories: Concept (creativity, originality, functionality, fit to challenge), Design (CAD, planning, user-friendliness), Production (reproducibility, manufacturing detail, scalability), Benchmark (impact, viability, realistic cost, competitor identification), and Communication (requirements met, documentation quality, design openness). SNIPPET-ONLY — [Hackaday Prize 2023 Official Rules PDF](https://cdn.hackaday.io/files/1901528135463168/Hackaday%20Prize%202023%20Official%20Rules.pdf). Contrast: unlike a weekend hackathon, the Hackaday Prize rewards documentation and manufacturability.

**Hardware winners (from the winners' own GitHub READMEs, FULL-TEXT)**
- **WiGhost — Best Hardware Hack, Hack&Roll 2026 (NUS Hackers).** It uses multiple ESP32 nodes (at least 2 secondary nodes plus an ESP32-S3 primary/server) to make Wi-Fi interference, contention and adaptation "visible, physical, and interactive". Why it worked: cheap off-the-shelf microcontrollers plus an invisible phenomenon made tangible and educational. — [soham131345/wighostviz](https://github.com/soham131345/wighostviz)
- **InkSight — 1st, Best Hardware Hack, GenAI Genesis (Canada; year not stated in README, UNVERIFIED likely 2024/2025).** "RAG meets robotics to read and chat with your notebook". A 4-person team with a demo video and Devpost link. Why it worked: it pairs the current AI trend (RAG) with a physical robot. — [AryanK1511/InkSight](https://github.com/AryanK1511/InkSight)
- **OPTimism — Best Hardware Hack, Hack the 6ix (36h).** A vision-care platform. Its hardware is a gyroscope for posture warnings and an ultrasonic sensor for screen-distance tracking, plus an AI chatbot, gamified credits and donations via Circle (a sponsor API). Why it worked: a health problem with a strong statistic (WHO: 2.2B people with vision impairment), simple sensors, and a sponsor-API tie-in. — [ricsign/OPTimism](https://github.com/ricsign/OPTimism)
- **SleepStop / CarSafety — 3rd Overall + Best Hardware Hack, Hack the Valley V.** A webcam plus OpenCV detects closed eyes and plays aircraft-style alarms. It also has crash detection that contacts authorities with the location. The README opens with drowsy-driving statistics (AAA: 328,000 crashes/yr). Why it worked: a life-safety framing and a simple, demoable sensor loop. — [EmreCenk/car-safety](https://github.com/EmreCenk/car-safety)
- **BrainViz — MLH Best Hardware Hack, CodeJam.** An Emotiv EPOC+ EEG headset streams over LSL to Python and then to Unity, so blinks, head movement and emotions control a game. Pitched for accessibility (users with limited movement). — [zephirl/BrainViz](https://github.com/zephirl/BrainViz)
- **Fifth Sense / BrailleWare — PennApps XII (2015) Grand Prize + Best Hardware Hack.** Six buttons with vibration motors give braille input and output, connected by Bluetooth to an Android assistant, plus a distance sensor that replaces a cane. An older example, but it shows that a hardware prize can stack with the grand prize. — [edhyah/BrailleWare](https://github.com/edhyah/BrailleWare)
- Other READMEs claiming "Best Hardware Hack" (mostly older, pre-2024): Octocatch (Hack the Burgh VII, 2021, a physical cabinet build) [deepsea-dev/octocatch](https://github.com/deepsea-dev/octocatch); Axolotl (McHacks 9, Digi-Key-sponsored hardware prize, notably an 8-bit computer *simulation*, i.e. software only) [Eerohne/Axolotl](https://github.com/Eerohne/Axolotl); AirDJ (StuyHacks, Myo armbands) [Averylamp/AirDJ](https://github.com/Averylamp/AirDJ); Electric Vision (Hawkathon) [imdanielsp/electric-vision](https://github.com/imdanielsp/electric-vision).
- MIT Reality Hack 2024 "Best of Hardware Hack": Block Party, a platform letting people in VR headsets interact with people not in headsets. SNIPPET-ONLY — [Reality Hack 2024 tracks & prizes](https://www.realityhackatmit.com/2024-tracks-prizes)
- Hack the North 2024 hardware-flavoured winners named in a sponsor blog: SecureStep, Rex, LooGuessr. SNIPPET-ONLY, UNVERIFIED which prize each won — [Viam blog](https://www.viam.com/post/from-concept-to-creation-highlights-and-builds-from-hack-the-north)

### Inferences
- Under MLH-style rubrics, hardware teams get extra "Technology/wow" and "Design (HCI)" credit for a working physical interaction. Completion carries equal weight, though, so one reliable sensing-to-feedback loop beats an ambitious rig that fails at demo.
- The winners' pattern: cheap sensors or microcontrollers (ESP32, ultrasonic, gyroscope, webcam), one clear human use case (accessibility, safety, health, education), and a layer of AI or a sponsor API. Several also won overall or placement prizes, so hardware projects stack prizes.
- "Hardware hack" prizes are sometimes loosely applied (Axolotl was a CPU simulator). Read the prize text.
- Hackaday-style long-form contests reward documentation and reproducibility, unlike weekend hackathons where pitch and documentation are explicitly de-weighted (MLH).

### Gaps
- The official 2024–2026 hardware-prize winners for TreeHacks, HackMIT and Hack the North could not be confirmed, because Devpost galleries and Stanford Daily were blocked. The Stanford Daily articles ([2025](https://stanforddaily.com/2025/02/18/treehacks-awards-200000-in-prizes-to-students-from-around-the-world/), [2026: "12th annual TreeHacks awards $500,000 in prizes"](https://stanforddaily.com/2026/02/15/12th-annual-treehacks/)) likely name winners but could not be read.
- I found no published hardware-specific rubric for HackMIT or Hack the North.
- The 2025/2026 Hackaday Prize rules were not found; the 2023 rules are the most recent seen.

## 2. Solo participation: which hackathons allow it, pros/cons, strategies

### Takeaway
Most major university hackathons set team size as "up to 4", and the lower bound of 1 appears explicitly at HackMIT (1–4). Cal Hacks says you can come "with a team or solo". Snippets conflict on PennApps (a "2 to 4" minimum appeared). Solo hackers report winning by cutting scope hard, reusing familiar stacks, targeting sponsor/API prize tracks, and leaning on AI coding tools. MLH rules permit AI tools if they are disclosed.

### Cited Findings
- HackMIT 2023 rules: "Teams can be made up of anywhere from 1 to 4 hackers." SNIPPET-ONLY — [HackMIT 2023 Devpost rules](https://hack-mit-2023.devpost.com/rules)
- Cal Hacks: "Up to 4 people can be part of a single team" and "You can come with a team or solo." SNIPPET-ONLY — [calhacks.io](https://www.calhacks.io/)
- PennApps: caps teams at four; a snippet also says "Team required: 2 to 4 members". SNIPPET-ONLY, UNVERIFIED and possibly conflicting — [PennApps XXIV Devpost](https://pennapps-xxiv.devpost.com/). Treat as "check the current year's rules."
- HackHarvard 2025 judging prompt includes "if this is a solo project, did you explore and learn something new?", which implies solo entries are allowed. SNIPPET-ONLY — [HackHarvard 2025 Devpost](https://hackharvard-2025.devpost.com/)
- Hack the North 2025: teams of up to 4, all present in person. Each person may be on only one team, and each team may submit only one project. The rules do not state a minimum. SNIPPET-ONLY — [Hack the North 2025 rules](https://hackthenorth2025.devpost.com/rules)
- TreeHacks: "teams consist of up to four students". SNIPPET-ONLY, UNVERIFIED — [TreeHacks 2025 Devpost](https://treehacks-2025.devpost.com/)
- MLH standard rules set no team-size minimum, and the text refers to "teams" throughout. They require that "All team members should actively participate." FULL-TEXT — [MLH standard rules](https://github.com/MLH/mlh-policies/blob/main/standard-hackathon-rules.md)
- Solo-only events exist on Devpost, e.g. CodeCipher Solo Hackathon (high-schoolers only, virtual) and Solo Hacks 1.0. SNIPPET-ONLY — [CodeCipher](https://codecipher-solo-hackathon.devpost.com/), [Solo Hacks 1.0](https://solo-hacks-1-0.devpost.com/)
- Solo-winner strategies (blog, SNIPPET-ONLY): "Leverage knowledge and code you are already familiar with, as there is much less time"; go for "low hanging fruit" that wows judges; "If there are API sponsors at the event, utilize their software!" — [3 tips on winning a hackathon as a solo contestant](https://jcacperalta.github.io/blog/winning-a-hackathon/). A first-solo-hackathon write-up reports a 3rd-place finish — [Medium, Alena Nikulina](https://medium.com/@alenanikulina0/my-first-solo-hackathon-experience-3rd-place-8b56bfc7c493) (SNIPPET-ONLY; contents not read).
- A cited pro of going solo: "You make all of the decisions. Getting people to agree is hard, especially if you are following a tight timeline." SNIPPET-ONLY — search summary of solo-tips sources above (attribution between the two blogs unclear, UNVERIFIED).
- AI as a solo force-multiplier: a solo developer reportedly won an Anthropic hackathon in 8 hours with a Claude Code toolkit ($15k prize). SNIPPET-ONLY, UNVERIFIED (secondary Medium post) — [Medium/CodeCoup](https://medium.com/@CodeCoup/this-solo-developer-won-the-anthropic-hackathon-in-8-hours-then-open-sourced-the-entire-ai-coding-85c555ccf6ac)
- MLH explicitly allows AI code generation but requires disclosure: "Teams should be honest and transparent about the AI code tools they used. This includes listing them in their project submissions." FULL-TEXT — [MLH standard rules](https://github.com/MLH/mlh-policies/blob/main/standard-hackathon-rules.md)

### Inferences
- Pros of going solo: speed of decisions, total ownership, and fit with AI-assisted building. Cons: no parallel workstreams (hardware plus software plus pitch is hard alone), one person demoing and judging-wandering, and fatigue. Sponsor/API tracks are the most realistic prize targets for solo hackers, since judging there centres on meaningful use of the sponsor tech rather than breadth.
- Under MLH's "Learning" criterion and HackHarvard's solo prompt, a solo hacker can earn credit for stretching into new tech.
- Solo plus hardware is the hardest combination. If you go solo, choose a single microcontroller or sensor loop, or a software track.

### Gaps
- I found no aggregated data on how often solo entries win at major events.
- Exact 2025/2026 minimum team sizes for TreeHacks, PennApps and Hack the North are not confirmed from primary pages.

## 3. University hackathon rules (team size, pre-existing code, AI, eligibility, IP)

### Takeaway
Three rule sets were read in full from GitHub: MLH's standard rules (the template most university MLH-member events fork), TreeHacks' Terms and Conditions (Stanford), and UH CodeRED Astra 2025 (University of Houston, an MLH fork). The common core: no code before hacking starts (ideas are OK), open-source libraries are OK if not pre-built for the event, AI is allowed with disclosure, code must be public, and students are defined broadly. Age limits and IP terms vary by event, and I found no IP clause in any of the full texts read.

### Cited Findings
**MLH standard hackathon rules** (FULL-TEXT — [GitHub](https://github.com/MLH/mlh-policies/blob/main/standard-hackathon-rules.md)):
- Eligibility: "primarily for students, but may also include professionals... Anyone who attends a traditional school, college, or university, and those in bootcamps... Those who graduated within the last 12 months are also considered students." Organisers, volunteers, judges and sponsors "should not participate as a hacker."
- Pre-existing work: "All work on a project should be done during the period of the hackathon." Teams "can use an idea they had before the event" and "can work on an idea that they have worked on before (as long as they do not re-use code or other project materials)." Libraries and open source are allowed, but "Working on a project before the event and open-sourcing it for the sole purpose of using the code during the event... is not allowed."
- After time: small bug fixes during demo are OK ("only a few lines of code"), new features are not.
- Code "must be available publicly (ideally in a git repository)" and "must remain public post event to be eligible for prizes."
- AI: "Teams may use AI to assist them while coding... code completion, code generation, image generation"; tools must be disclosed in submissions and to judges.
- Digital events: demo video of 2 minutes or less, made that weekend, stating the hackathon name; no submitting to other hackathons; no prior work.

**TreeHacks Terms and Conditions** (Stanford; "Last updated on August 27, 2024") (FULL-TEXT — [TreeHacks/policies terms-and-conditions.md](https://github.com/TreeHacks/policies/blob/master/terms-and-conditions.md)):
- "Zero-tolerance policy towards project cheating, which includes... initiating a project prior to the official start of hacking, submitting a project previously created by a participant, or copying another participant's project."
- "All projects must be entirely original, created by the hacking team during the event. No code may be written prior to the official start of hacking."
- Enforcement: an "internal system" with "various checks to detect potential cheating"; manual review of all winning projects; a "holding period" between the winner announcement and prize distribution. Penalties include disqualification, loss of travel reimbursement, a permanent ban, and "notification to our sponsors and company recruiters." A confidential Cheating Report Form is provided.
- Eligibility misrepresentation (e.g. "status as a student") forfeits participation and prizes.
- Data sharing: organisers may share registration details, LinkedIn/GitHub profiles and project details with sponsors, and participation consents to sponsor communications.
- No alcohol, drugs or weapons; zero tolerance for occupying unapproved venues.
- (Oddity: the release section references "the laws of the Province of Ontario", apparently copied from a Canadian template.)
- No AI clause, IP clause or team-size clause appears in the T&C; the [Code of Conduct](https://github.com/TreeHacks/policies/blob/master/code-of-conduct.md) is a harassment policy only.

**UH CodeRED Astra 2025** (University of Houston, Oct 18–19, 2025) (FULL-TEXT — [UHCodeRED/rules](https://github.com/UHCodeRED/rules)):
- An MLH fork that adds an age rule: "All CodeRED Astra Attendees are required to be atleast 18 years of age."
- Makes the hacking window explicit: "starting 12:00 PM CST on Saturday 10/18/2025 and ending 12:00 PM CST on Sunday 10/19/2025".
- Code must be public "Ideally on Github, and publicly accessible without any restrictions to the judges."
- Same AI-with-disclosure rule ("listing them in their project submissions/repositories") and the same MLH four-criterion judging.

**Others (SNIPPET-ONLY)**
- HackMIT 2023: teams of 1–4; "You may plan your project in advance with your team, but may not write any code prior to the event"; open-source code and libraries allowed "but you must cite these clearly in your submission and demo." — [HackMIT 2023 rules](https://hack-mit-2023.devpost.com/rules)
- Hack the North 2025: up to 4 members, all in person; all must be accepted participants; one team per person, one project per team; the HTN Code of Conduct and the MLH Code of Conduct apply. Older HTN text: "only current students in University or High School who have been accepted" may submit (UNVERIFIED for 2025). — [HTN 2025 rules](https://hackthenorth2025.devpost.com/rules), [HTN 2023 Code of Conduct](https://2023.hackthenorth.com/code-of-conduct)

### Inferences
- Guide-ready summary: (1) teams of up to 4, with solo usually allowed (check the event); (2) ideas and planning before the event are fine, but code before the event is not (TreeHacks actively audits); (3) open-source libraries are fine if cited, but a repo pre-built "for the event" is not; (4) AI tools are generally allowed but must be disclosed (MLH and forks); (5) code must stay public to keep a prize; (6) eligibility is "students" broadly, including recent grads within 12 months under MLH, and some events require 18+ (CodeRED) while others admit high-schoolers (HTN, UNVERIFIED for 2025); (7) expect your profile and project to be shared with sponsors.
- IP: none of the full-text rule sets read assign IP to organisers. Absent a clause, the team keeps ownership, but public-repo requirements mean the code is visible. Check sponsor-prize terms separately. (Inference, not a cited rule.)

### Gaps
- No IP clause was found in the full-text university rule sets. HackMIT, PennApps, CalHacks and HackHarvard rule pages (Devpost/official sites) were blocked, so their IP, AI and age terms are unconfirmed.
- No university-specific AI policy beyond the MLH-template wording was found. TreeHacks' T&C is silent on AI.
- Hack the North's 2025 high-school eligibility and minimum age are unconfirmed.
