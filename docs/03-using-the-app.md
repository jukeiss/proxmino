# 3. Using the app

Open http://127.0.0.1:3000 after `./run.sh`. Type in the box at the bottom. The pills above it change how
Proxmino answers.

## Auto mode (the default)

Proxmino picks an approach for each message and shows why as the first step:

| You ask | What it does |
|---|---|
| A maths or logic problem | Works it out step by step and checks the result with a small Python program |
| For code | Writes it, writes tests for it, runs them, and fixes the code once if a test fails |
| For a web page or UI | Writes one complete HTML file (with phone layout and dark mode) you can preview and download |
| About something recent | Searches the web and cites sources as `[1]`, `[2]` |
| Anything else | Answers straight away |

## Pills and commands

| Pill | Command | What it does |
|---|---|---|
| Think | `/think` | Shows its reasoning as a numbered list before the answer |
| Search | `/search` | Searches the web first and cites sources |
| DeepSearch | `/deep` | Plans several searches, fills gaps, then writes a report |
| Heavy | `/heavy` | Three agents draft answers; a fourth merges the best parts |
| | `/check <claim>` | Fact-checks a claim, or the last answer when used alone |

## Attach files

Click **+** and choose a file: PDF, Word (.docx), Excel (.xlsx), PowerPoint (.pptx), or any text or code
file. Then ask about it, for example "What was the total revenue for the North region?". The file is read on
your computer. For a small table, totals and averages are worked out with a program, so the sums are exact.

Pictures and old .doc/.xls/.ppt files can't be read yet; save them as PDF or the newer format first.

## Agents

The crew (Atlas, Probe, Edge, Nexus, Warden and others) are personalities with their own instructions,
defined in `crew/*.yaml`. Mention one with `@name` or pick one from the list. More ready-made agents are in
`market/*.yaml`; copy one into `crew/` to use it.

## Tips

- A model on your own computer writes about 10 to 30 words a second, so long answers take a while.
  Proxmino Lite is less than half the size of v5, so it answers faster.
- Every answer can be regenerated, edited or branched from the message's hover menu.
