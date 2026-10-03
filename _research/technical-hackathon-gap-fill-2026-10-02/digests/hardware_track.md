# Succeeding at hardware / IoT / embedded hackathon tracks

NOTE ON SOURCING: WebFetch was blocked by the egress proxy for every domain tried (hackthenorth.medium.com, guide.mlh.io, wokwi.com, edgeimpulse.com, solaride.ee, zbotic.in, purdue.edu, makerbot.com, emqx.com). All findings below come from WebSearch result snippets only, so every claim is SNIPPET-ONLY unless stated. Pricing/limits especially must be re-checked on the vendor pages before the guide states them as fact. Several tip sources are low-authority blogs (zbotic, hack4purpose, angelhack); treat as folk wisdom.

## 1. Common platforms and what hackathons provide

### Takeaway
MLH-style hardware labs lend Raspberry Pi and Arduino plus Grove sensors, servos, webcams and misc parts; ESP32 was not named in the MLH snippet but is the usual bring-your-own Wi-Fi board. Check each event's hardware list in advance, as stock is limited and checked out.

### Cited Findings
- MLH Member Events may receive a hardware lab to rent components to hackers over the weekend: assortments of microcontrollers, components, smart devices and seasonal items such as Raspberry Pis and Arduinos. SNIPPET-ONLY — [MLH Hardware (organizer guide)](https://guide.mlh.io/general-information/event-logistics/hardware)
- Typical lab contents: Raspberry Pi 4B kits, Arduinos and base shields, Logitech webcams, Grove sensors/components, micro servos, motors, resistors, capacitors, cables. SNIPPET-ONLY — [MLH Hardware Lab Contents](https://guide.mlh.io/organizer-resources/hardware-lab-contents)
- MLH has a limited number of labs, sent to US-based events only; organizers must ask their Hackathon Community Manager. SNIPPET-ONLY — [MLH Hardware Lab Contents](https://guide.mlh.io/organizer-resources/hardware-lab-contents)
- Hackers picking up Arduino hardware should install the Arduino IDE to program from their laptop. SNIPPET-ONLY — [MLH Hardware](https://guide.mlh.io/general-information/event-logistics/hardware)
- MLH publishes a Hardware Hackathon Guide with a prizes section: [Prizes](https://guide.mlh.com/hardware-hackathon-guide/prizes), [Why Hardware Hackathons](https://guide.mlh.com/hardware-hackathon-guide/why-hardware-hackathons) (titles only seen; content not read).
- Dedicated hardware events exist at scale: StarkHacks (Purdue) billed as the world's largest hardware hackathon, 750+ hackers from 80 universities and 10 countries, sponsors incl. Analog Devices, Qualcomm, Microsoft, Tesla, Anthropic, >$100k prizes; tracks spanned robotics, AI, accessibility, wearables, space. SNIPPET-ONLY — [Purdue Engineering news](https://engineering.purdue.edu/Engr/AboutUs/News/Spotlights/2026/2026-0519-purdue-starkhacks-hardware-hackathon); [StarkHacks Devpost](https://starkhacks.devpost.com/)
- StarkHacks coverage notes 3D printing enabled rapid prototyping. SNIPPET-ONLY — [MakerBot story](https://www.makerbot.com/stories/starkhacks-purdue-3d-printing-hackathon/)
- Platform comparison: ESP32-S3 suited to TinyML with typical model sizes 100KB-500KB. SNIPPET-ONLY from a vendor blog — [JLCPCB ESP32 vs Raspberry Pi](https://jlcpcb.com/blog/esp32-vs-raspberry-pi)
- Prep advice: if using a Raspberry Pi, have the OS installed and tested ahead of time; download drivers/libraries before the event. SNIPPET-ONLY — [Zbotic](https://zbotic.in/how-to-win-a-hackathon-electronics-build-tips-for-students/) and [Solaride](https://solaride.ee/blog/prepare-yourselves-how-to-survive-a-hardware-hackathon-aka-buildathon) (snippet attributed to search summary, not clearly to one page)

### Inferences
- Beginner default: ESP32 (Wi-Fi/BLE, cheap, Arduino-IDE/PlatformIO/MicroPython) for sensing+cloud; Raspberry Pi for camera/ML/local broker; Arduino when the lab only has Uno-class boards. Pico/Jetson were not substantiated by sources found.
- Verify with organizers which parts are lent and whether a deposit/checkout applies.

### Gaps
- No source found ranking platform popularity (ESP32 vs Arduino vs Pi vs Pico vs Jetson) at hackathons; Jetson/Pico loaner availability unverified (UNVERIFIED).
- MLH guide pages not readable in full (blocked), so lending rules (deposits, return) are missing.

## 2. Winning project patterns and failure modes

### Takeaway
Reproducible patterns: sensor -> MQTT/cloud -> live dashboard; TinyML/edge classification on ESP32/Pico/XIAO; wearables/gesture-controlled robotics. Failures are dominated by loose wires, power, and venue Wi-Fi.

### Cited Findings
- "90% of the time" sudden hardware failure is a loose wire; hot-glue wires to breadboard; avoid soldering during the event, pre-solder headers. SNIPPET-ONLY, low-authority — [Zbotic / search summary](https://zbotic.in/how-to-win-a-hackathon-electronics-build-tips-for-students/)
- Power: bring power banks (e.g. 2x 20,000mAh) as outlets are limited. SNIPPET-ONLY — [Zbotic](https://zbotic.in/how-to-win-a-hackathon-electronics-build-tips-for-students/)
- Wi-Fi: have fallback — pre-downloaded libraries, local MQTT broker on a Raspberry Pi, cellular backup; don't rely on venue Wi-Fi for big downloads. SNIPPET-ONLY — [Zbotic](https://zbotic.in/how-to-win-a-hackathon-electronics-build-tips-for-students/), [Hack the North hardware advice](https://hackthenorth.medium.com/advice-for-hardware-hackers-at-hackathons-5736821e12fe) (title seen; content not read)
- Pattern examples (all Hackster.io, none verified as hackathon winners; use as reference builds):
  1. Gas detection + safety monitoring, ESP32 + MQ2 + MQTT + 3-tier LED/buzzer alerts (Sept 2025) — [IoT Projects Part 9](https://www.hackster.io/theembeddedthings/iot-projects-part-9-gas-detection-and-safety-monitoring-8de553)
  2. Motion tracking, ESP32 + MPU6050 IMU, JSON over MQTT, live plots — [IoT Projects Part 8](https://www.hackster.io/theembeddedthings/iot-projects-part-8-motion-tracking-with-mpu6050-b87fb2)
  3. ESP32 4G LTE GPS tracker with MQTT web dashboard (cellular = no venue Wi-Fi dependency) — [techgyanset](https://www.hackster.io/techgyanset/esp32-4g-lte-gps-tracker-with-mqtt-dashboard-19e738)
  4. Temperature/humidity monitor with ESP32 + Home Assistant — [sarful](https://www.hackster.io/sarful/temperature-humidity-monitor-with-esp32-and-home-assistant-4d94ab)
  5. Series overview/architecture for ESP32+MQTT IoT — [Part 1](https://www.hackster.io/theembeddedthings/iot-projects-part-1-system-overview-architecture-e78836)
  6. Air-quality monitor with ESP32 — [dudelabweb](https://www.hackster.io/dudelabweb/air-quality-monitoring-made-easy-an-iot-project-with-esp32-ee2777)
- Edge AI examples: gesture recognition on Raspberry Pi Pico ("up-down", "left-right", "circle") — [MJRoBot](https://www.hackster.io/mjrobot/tinyml-motion-recognition-using-raspberry-pi-pico-6b6071); image classification on XIAO ESP32S3 Sense — [MJRoBot](https://www.hackster.io/mjrobot/tinyml-made-easy-image-classification-w-xiao-esp32s3-sense-cb42ae); object detection on same board — [MJRoBot](https://www.hackster.io/mjrobot/tinyml-made-easy-object-detection-with-xiao-esp32s3-sense-6be28d)
- "Take It to the Edge with Edge Impulse" Hackster contest; snippet says TinyML-on-Pico winners were Christopher Mendez Martinez (1st), Wen-Liang Lin (2nd), Xiao Shi Zi Yi (3rd) and mentions an electronic-nose (ESP32 + MEMS gas sensors) project distinguishing juice from alcoholic drinks. SNIPPET-ONLY, attribution of the e-nose to the contest unclear — [Hackster event page](https://events.hackster.io/take-it-to-the-edge)
- StarkHacks highlighted project: robotic hand with 3D-printed parts mirroring a sensor-equipped glove in real time (wearable + robotics). SNIPPET-ONLY — [Purdue news](https://engineering.purdue.edu/Engr/AboutUs/News/Spotlights/2026/2026-0519-purdue-starkhacks-hardware-hackathon)

### Inferences
- Judges likely reward a visible physical interaction plus a live data view; a sensor->dashboard demo is the lowest-risk story. Hands-on wearable/robotic builds appear to get press attention (one example only).
- Cellular or Pi-local broker removes the biggest single external dependency (venue Wi-Fi).

### Gaps
- No Devpost winner write-up with post-mortem found (search returned only event pages). No Devpost hardware-winner URL verified. Judging criteria for hardware tracks not found.

## 3. Time plan, fallback, demo protocol

### Takeaway
Experienced-hacker advice (low-authority, snippet-only) converges on: prep software/OS beforehand, avoid soldering, rehearse demo many times, always have a video/simulated-data fallback.

### Cited Findings
- Run the demo ~20 times, fix every failure mode, keep a video recording as emergency backup. SNIPPET-ONLY — [Zbotic](https://zbotic.in/how-to-win-a-hackathon-electronics-build-tips-for-students/)
- Another source: 5-7 dry runs including offline/recorded demos in case of connectivity issues. SNIPPET-ONLY — [Hack4Purpose](https://hack4purpose.in/hackathon-2025/)
- Prepare a software-only fallback (video walkthrough or simulated data) so the pitch lands if hardware fails; screen recordings or click-through mockups as backup. SNIPPET-ONLY — [AngelHack best practices](https://angelhack.com/blog/hackathon-best-practices/)
- Simulators for pre-event/fallback: Wokwi (Arduino, Pi Pico, ESP32 in browser; Espressif docs list it as supported) — [Espressif Arduino-ESP32 docs](https://docs.espressif.com/projects/arduino-esp32/en/latest/third_party/wokwi.html); [CNX Software](https://www.cnx-software.com/2023/04/10/wokwi-arduino-raspberry-pi-pico-esp32-board-simulator/) (2023). Tinkercad vs Wokwi vs Proteus comparison — [Zbotic](https://zbotic.in/best-arduino-simulator-tools-tinkercad-vs-wokwi-vs-proteus/)

### Inferences
- Suggested split for a 24-36h event (my synthesis, NOT sourced): first 20% scope + parts check + hello-world on each board; 50% build one end-to-end thin slice before polish; last 20% freeze features, record backup video, rehearse; keep a spare board flashed identically.
- Record the video as soon as the demo first works end-to-end, not at the end.

### Gaps
- No time-plan percentages from a primary hardware-hackathon source found. Hack the North article (likely the best source) unreadable.

## 4. Free tools and current free-tier limits (verify before publishing)

### Takeaway
All of Wokwi, Edge Impulse, HiveMQ Cloud and EMQX Serverless have usable free tiers, but figures are snippet-only and change often.

### Cited Findings
- Wokwi Community (free): unlimited simulations and unlimited public projects; free CI simulation limited to 50 min/month; Hobby plan EUR5.6/mo (100 fast-build minutes, unlisted projects, custom libraries, private IoT gateway); Hobby+ EUR8.1/mo. SNIPPET-ONLY, search summary — [Wokwi pricing](https://wokwi.com/pricing); CI docs [Wokwi CI](https://docs.wokwi.com/wokwi-ci/getting-started)
- Edge Impulse Developer Plan (free): GPU access, 60-minute training jobs (60 min compute per job); Enterprise unlimited. SNIPPET-ONLY, and Edge Impulse has revised free plans (see blog "We're Updating Our Free Community Plan") so numbers may be dated — [Edge Impulse pricing](https://www.edgeimpulse.com/pricing), [update blog](https://www.edgeimpulse.com/blog/were-updating-our-free-community-plan-here-are-the-details/), [Developer plan intro](https://www.edgeimpulse.com/blog/introducing-the-developer-plan/)
- HiveMQ Cloud Serverless free: 100 device connections, 10 GB traffic/month, 3-day data retention, 5 MB message size, no SLA. SNIPPET-ONLY — [HiveMQ pricing](https://www.hivemq.com/pricing/), [HiveMQ blog](https://www.hivemq.com/blog/hivemq-cloud-serverless-vs-starter-mqtt-iot-or-iiot/)
- EMQX Serverless free: 1M session minutes/month and 1 GB traffic, up to 1000 connections (1M min ~ 23 devices online all month). SNIPPET-ONLY — [EMQX free MQTT broker blog](https://www.emqx.com/en/blog/free-mqtt-broker), [EMQX pricing](https://www.emqx.com/en/pricing)
- Public free test brokers exist and were benchmarked (2022; not for sensitive data) — [goughlui benchmark](https://goughlui.com/2022/10/23/project-benchmarking-public-free-mqtt-brokers/)
- Arduino IDE is the MLH-recommended programming tool for lent Arduino hardware — [MLH Hardware](https://guide.mlh.io/general-information/event-logistics/hardware)

### Inferences
- A local Mosquitto/Pi broker or hosted free broker plus Home Assistant or a simple web dashboard covers the sensor->cloud->dashboard pattern at $0.

### Gaps
- PlatformIO, Home Assistant, Tinkercad Circuits limits: no searches done / no sources (UNVERIFIED). Wokwi per-month figures for 2026 not confirmed on primary page. Edge Impulse current (2026) free limits unconfirmed.
