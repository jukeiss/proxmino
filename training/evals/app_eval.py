"""36 questions with known answers, asked through the running Proxmino app (local model only).
Usage: python app_eval.py <label>   -> results/<label>.json and results/<label>/*.md
Every expected answer was worked out by a program beforehand. Maths is graded on the final lines only."""
import json, os, random, re, sys
from common import ask, code_of, final_lines, run_py

HERE = os.path.dirname(os.path.abspath(__file__))

CODING = [
    ("brackets", "Write a Python function is_balanced(s) that returns True if every (, [ and { in s is closed in the right order, ignoring other characters.",
     "assert is_balanced('a(b[c]{d}e)f')\nassert not is_balanced('([)]')\nassert not is_balanced('((')\nassert is_balanced('')\nassert not is_balanced('}')"),
    ("intervals", "Write a Python function merge(intervals) that merges overlapping [start, end] intervals (touching ones like [1,2],[2,3] count as overlapping) and returns them sorted.",
     "assert merge([[1,3],[2,6],[8,10],[15,18]])==[[1,6],[8,10],[15,18]]\nassert merge([[1,4],[4,5]])==[[1,5]]\nassert merge([])==[]\nassert merge([[5,7],[1,2]])==[[1,2],[5,7]]"),
    ("roman", "Write a Python function roman_to_int(s) that converts a Roman numeral like 'MCMXCIV' to an integer.",
     "assert roman_to_int('III')==3\nassert roman_to_int('LVIII')==58\nassert roman_to_int('MCMXCIV')==1994\nassert roman_to_int('IV')==4"),
    ("substring", "Write a Python function longest_unique(s) that returns the length of the longest substring of s with no repeated characters. It must run in O(n).",
     "assert longest_unique('abcabcbb')==3\nassert longest_unique('bbbbb')==1\nassert longest_unique('pwwkew')==3\nassert longest_unique('')==0\nassert longest_unique('abba')==2\nimport time;t=time.time();longest_unique('ab'*200000);assert time.time()-t<2"),
    ("topk", "Write a Python function top_k(words, k) that returns the k most frequent words, ordered by frequency from high to low, and alphabetically when frequencies tie.",
     "assert top_k(['i','love','code','i','love','coding'],2)==['i','love']\nassert top_k(['the','day','is','sunny','the','the','the','sunny','is','is'],4)==['the','is','sunny','day']\nassert top_k(['b','a'],2)==['a','b']"),
    ("lru", "Write a Python class LRUCache(capacity) with get(key) (returns -1 if missing) and put(key, value), both O(1), evicting the least recently used key when full.",
     "c=LRUCache(2);c.put(1,1);c.put(2,2);assert c.get(1)==1;c.put(3,3);assert c.get(2)==-1;c.put(4,4);assert c.get(1)==-1;assert c.get(3)==3;assert c.get(4)==4"),
    ("expr", "Write a Python function evaluate(s) that evaluates an integer arithmetic expression string with +, -, *, /, parentheses, spaces and unary minus (like '-(2+3)*4' or '2*-3'). Division truncates toward zero, like C. Do not use eval or exec.",
     "assert evaluate('1 + 2 * 3')==7\nassert evaluate('(1+2)*3')==9\nassert evaluate('-(2+3)*4')==-20\nassert evaluate('2*-3')==-6\nassert evaluate('7/-2')==-3\nassert evaluate('10 - 4 - 3')==3\nassert evaluate('2*(3+(4-1))/4')==3\nassert evaluate('--5')==5"),
    ("lis", "Write a Python function lis_length(nums) that returns the length of the longest strictly increasing subsequence. It must handle 200,000 numbers in under a second (O(n log n)).",
     "assert lis_length([10,9,2,5,3,7,101,18])==4\nassert lis_length([0,1,0,3,2,3])==4\nassert lis_length([7,7,7])==1\nassert lis_length([])==0\nimport random,time;random.seed(3);a=[random.randint(0,10**9) for _ in range(200000)]\nt=time.time();r=lis_length(a);assert time.time()-t<2\nassert lis_length(list(range(200000)))==200000"),
    ("queens", "Write a Python function count_queens(n) that returns the number of ways to place n queens on an n by n board so that no two attack each other. count_queens(10) must finish in a few seconds.",
     "assert count_queens(1)==1\nassert count_queens(2)==0\nassert count_queens(4)==2\nassert count_queens(8)==92\nimport time;t=time.time();assert count_queens(10)==724;assert time.time()-t<8"),
    ("editdist", "Write a Python function edit_distance(a, b) that returns the minimum number of single-character inserts, deletes and substitutions to turn string a into string b.",
     "assert edit_distance('kitten','sitting')==3\nassert edit_distance('','abc')==3\nassert edit_distance('intention','execution')==5\nassert edit_distance('same','same')==0\nassert edit_distance('ab','ba')==2"),
    ("windowmax", "Write a Python function window_max(nums, k) that returns the maximum of every contiguous window of size k, in O(n) total time.",
     "assert window_max([1,3,-1,-3,5,3,6,7],3)==[3,3,5,5,6,7]\nassert window_max([4],1)==[4]\nassert window_max([9,8,7,6],2)==[9,8,7]\nimport time;a=list(range(300000,0,-1));t=time.time();r=window_max(a,1000);assert time.time()-t<2 and r[0]==300000 and len(r)==299001"),
    ("dijkstra", "Write a Python function shortest(n, edges, src) where edges is a list of (u, v, w) directed edges with non-negative weights on nodes 0..n-1. Return a list of shortest distances from src, with -1 for unreachable nodes.",
     "assert shortest(4,[(0,1,4),(0,2,1),(2,1,2),(1,3,1)],0)==[0,3,1,4]\nassert shortest(3,[(0,1,5)],0)==[0,5,-1]\nassert shortest(2,[],1)==[-1,0]\nassert shortest(3,[(0,1,0),(1,2,0)],0)==[0,0,0]"),
]

MATHS = [
    ("arith", "What is 17 * 23 + 144 / 12?", ["403"]),
    ("discount", "A jacket costs $40. It is 25% off, then 8% sales tax is added to the discounted price. What do I pay?", ["32.4"]),
    ("sum35", "What is the sum of all whole numbers from 1 to 200 that are divisible by 3 or by 5?", ["9368"]),
    ("train", "A train goes 60 km/h for 2.5 hours, then 80 km/h for 1 hour 15 minutes. How far does it go in total?", ["250"]),
    ("dice", "Two fair six-sided dice are rolled. What is the probability that they sum to 8?", ["5/36", "0.1389", "13.89"]),
    ("linear", "Solve 3x + 7 = 2x - 5.", ["-12"]),
    ("ages", "Maria is 3 times as old as her son. In 12 years she will be twice as old as him. How old is Maria now?", ["36"]),
    ("percent", "A town's population grew from 12,500 to 14,000. What was the percentage increase?", ["12%", "12 %", "12 percent"]),
    ("digitsum", "How many integers from 1 to 10,000 inclusive have a digit sum of exactly 20?", ["633"]),
    ("powmod", "What is the remainder when 3^2024 is divided by 1000?", ["481"]),
    ("tiling", "In how many ways can a 3 by 10 rectangle be tiled completely with 1 by 2 dominoes?", ["571"]),
    ("zeros", "What is the smallest positive integer n such that n! ends in at least 100 zeros?", ["405"]),
    ("sixes", "You roll a fair six-sided die repeatedly until you get two sixes in a row. What is the expected number of rolls?", ["42"]),
    ("frog", "A frog starts on pad 1 of a row of pads numbered 1, 2, 3, ... Each jump moves it forward 1 pad or 2 pads, each with probability 1/2. What is the probability that it lands on pad 10 at some point? Give an exact fraction.", ["341/512"]),
    ("triples", "How many triples of positive integers (a, b, c) with a < b < c satisfy a + b + c = 30?", ["61"]),
    ("lastdigits", "What are the last two nonzero digits of 50! (50 factorial)?", ["12"]),
    ("clock", "Between 3:00 and 4:00, at how many minutes past 3 are the hour and minute hands of a clock exactly opposite each other? Give an exact answer.", ["49 1/11", "540/11", "49.09"]),
    ("liars", "A, B and C each either always tell the truth or always lie. A says: \"B is a liar.\" B says: \"A and C are the same type.\" How many of the three are truth-tellers? (The answer is the same in every consistent case.)", ["1", "one"]),
    ("incircle", "A right triangle has legs 20 and 21. What is the radius of its inscribed circle?", ["6"]),
    ("fullhouse", "How many 5-card hands from a standard 52-card deck are a full house?", ["3744"]),
]

CSV = "region,month,units,price\nNorth,Jan,120,9.5\nSouth,Jan,80,9.5\nNorth,Feb,150,9.0\nSouth,Feb,95,9.0\nEast,Feb,60,10.0\nNorth,Mar,130,9.0\n"
FILES = [
    ("csv", f'<file name="sales.csv">\n{CSV}</file>\n\nUsing the attached file: what was the total revenue (units times price) for the North region?', ["3660"]),
    ("csv2", f'<file name="sales.csv">\n{CSV}</file>\n\nWhich month in the attached file sold the most units in total, and how many?', ["Feb", "305"]),
]

UI = ("Build a single-file HTML page: a pricing section with three plans (Free, Pro, Team), the Pro plan highlighted as most "
      "popular, a monthly/yearly toggle that changes the prices, responsive down to phone width, and a dark mode that follows "
      "the system setting. Put all CSS and JS inline.")


def has(text, want):
    return any(re.search(r"(?<![\d/.])" + re.escape(w) + r"(?![\d/])", text, re.I) for w in want)


def main(label):
    out = os.path.join(HERE, "results", label)
    os.makedirs(out, exist_ok=True)
    res = []

    def rec(kind, name, ok, secs, route, note, text):
        res.append({"kind": kind, "name": name, "ok": bool(ok), "secs": secs, "route": route, "note": note[:200]})
        open(os.path.join(out, f"{kind}_{name}.md"), "w").write(text)
        print(f"{kind:7} {name:10} {'PASS' if ok else 'FAIL'} {secs:7}s route={route} {note[:110]}", flush=True)

    for name, q, tests in CODING:
        text, model, route, secs = ask(q)
        src = code_of(text).split("if __name__")[0]
        ok, o = run_py(src + "\n\n" + tests + "\nprint('ok')") if src else (False, "no code block")
        rec("coding", name, ok, secs, route, (o.strip().splitlines() or [""])[-1], text)
    for name, q, want in MATHS:
        text, model, route, secs = ask(q)
        rec("maths", name, has(final_lines(text), want), secs, route, f"want {want} | {final_lines(text, 1)[:80]}", text)
    for name, q, want in FILES:
        text, model, route, secs = ask(q)
        flat = text.replace(",", "").replace("$", "")
        rec("files", name, all(w in flat for w in want), secs, route, f"want {want}", text)
    text, model, route, secs = ask(UI)
    html = code_of(text, "html")
    checks = {"html": "<html" in html.lower(), "3 plans": all(p in html for p in ("Free", "Pro", "Team")),
              "media query": "@media" in html, "dark mode": "prefers-color-scheme" in html,
              "toggle js": "<script" in html.lower() and ("yearly" in html.lower() or "annual" in html.lower()),
              "viewport": "viewport" in html}
    open(os.path.join(out, "pricing.html"), "w").write(html)
    rec("ui", "pricing", all(checks.values()), secs, route, str([k for k, v in checks.items() if not v]), text)
    print("model:", model)
    total = sum(r["ok"] for r in res)
    print("TOTAL", total, "of", len(res))
    json.dump({"label": label, "model": model, "total": total, "of": len(res), "results": res},
              open(os.path.join(HERE, "results", f"{label}.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "run")
