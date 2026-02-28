"""
grading.py — Grade scored MCQ answer sheets against an answer key file
=======================================================================
Reads:
  1. answer_key.json   — correct answers per exam set code (mã đề thi)
  2. ScoredSheets/     — JSON files output by main_algorithm.py

Outputs:
  - Console: per-student score table grouped by class
  - grading_report.json: full machine-readable report

Usage
─────
  python3 grading.py \\
      --scored   images/answer_sheets/demo2/ScoredSheets \\
      --key      answer_key.json \\
      --out      grading_report.json

Answer key format  (answer_key.json)
──────────────────────────────────────────────────────────────────────
  {
    "exam_name":       "Midterm Exam",
    "subject":         "Introduction to Computer Science",
    "total_questions": 60,
    "total_score":     10.0,
    "scoring_rule":    "correct_only",   // "correct_only" | "partial"
    "keys": {
      "423": ["ABC", "ACD", "ABCD", ...],   // 60 answers for exam set 423
      "915": ["A",   "B",   "C",    ...]    // 60 answers for exam set 915
    }
  }

Scoring rules
─────────────
  correct_only : full mark for exact match, 0 otherwise  (default)
  partial      : score = (letters in common / letters in key) × mark_per_q
                 e.g. key="ABD", student="AB" → 2/3 partial credit
"""

import os
import sys
import json
import argparse
from collections import defaultdict


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation
# ─────────────────────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    """Uppercase, sort letters, treat blank / 'x' / '' / 'unchoice' as 'X'."""
    s = (s or "").strip().upper()
    if s in ("", "X", "UNCHOICE"):
        return "X"
    return "".join(sorted(set(s)))   # deduplicate + sort: "ACBD" → "ABCD"


def _score_question(student_ans: str, key_ans: str,
                    mark: float, rule: str) -> tuple[float, str]:
    """
    Returns (points_earned, verdict).
    verdict: 'correct' | 'incorrect' | 'partial' | 'unanswered'
    """
    s = _norm(student_ans)
    k = _norm(key_ans)

    if s == "X" and k == "X":
        return mark, "correct"     # both blank = correct

    if s == "X":
        return 0.0, "unanswered"

    if s == k:
        return mark, "correct"

    if rule == "partial" and k != "X":
        common = set(s) & set(k)
        earned = mark * len(common) / len(k)
        return round(earned, 4), "partial"

    return 0.0, "incorrect"


# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────

def load_answer_key(key_path: str) -> dict:
    with open(key_path, encoding="utf-8") as f:
        return json.load(f)


def load_scored_sheets(scored_dir: str) -> list[dict]:
    sheets = []
    for fname in sorted(os.listdir(scored_dir)):
        if not fname.endswith("_data.json"):
            continue
        path = os.path.join(scored_dir, fname)
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        # Attach source filename for traceability
        d["_source_file"] = fname
        sheets.append(d)
    return sheets


# ─────────────────────────────────────────────────────────────────────────────
# Grade one sheet
# ─────────────────────────────────────────────────────────────────────────────

def grade_sheet(sheet: dict, key_cfg: dict) -> dict:
    """
    Returns a grading result dict for a single student sheet.
    """
    exam_code    = sheet.get("testSetCode", "").strip()
    student_code = sheet.get("studentCode", "").strip()
    class_code   = sheet.get("examClassCode", "").strip()

    total_q   = key_cfg["total_questions"]
    total_pts = key_cfg["total_score"]
    rule      = key_cfg.get("scoring_rule", "correct_only")
    mark_per_q = total_pts / total_q

    # Look up key for this exam set
    keys_map = key_cfg.get("keys", {})
    if exam_code not in keys_map:
        return {
            "student_code":  student_code,
            "class_code":    class_code,
            "exam_code":     exam_code,
            "error":         f"No answer key found for exam code '{exam_code}'",
            "score":         None,
            "source_file":   sheet.get("_source_file", ""),
        }

    correct_answers = keys_map[exam_code]   # list of strings, length = total_q

    # Build answer lookup from sheet
    student_answers = {a["questionNo"]: a["selectedAnswers"]
                       for a in sheet.get("answers", [])}

    details = []
    total_earned  = 0.0
    n_correct     = 0
    n_incorrect   = 0
    n_partial     = 0
    n_unanswered  = 0

    for q_no in range(1, total_q + 1):
        key_ans     = correct_answers[q_no - 1] if q_no - 1 < len(correct_answers) else "x"
        student_ans = student_answers.get(q_no, "")

        earned, verdict = _score_question(student_ans, key_ans, mark_per_q, rule)
        total_earned += earned

        if verdict == "correct":   n_correct    += 1
        elif verdict == "partial": n_partial    += 1
        elif verdict == "incorrect": n_incorrect += 1
        else:                      n_unanswered += 1

        details.append({
            "questionNo":   q_no,
            "student_ans":  _norm(student_ans) if student_ans else "x",
            "key_ans":      _norm(key_ans),
            "earned":       round(earned, 4),
            "verdict":      verdict,
        })

    score = round(min(total_earned, total_pts), 2)

    return {
        "student_code":  student_code,
        "class_code":    class_code,
        "exam_code":     exam_code,
        "score":         score,
        "total_score":   total_pts,
        "n_correct":     n_correct,
        "n_incorrect":   n_incorrect,
        "n_partial":     n_partial,
        "n_unanswered":  n_unanswered,
        "source_file":   sheet.get("_source_file", ""),
        "detail":        details,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Print report
# ─────────────────────────────────────────────────────────────────────────────

def print_report(results: list[dict], key_cfg: dict):
    exam_name = key_cfg.get("exam_name", "")
    subject   = key_cfg.get("subject", "")
    total_pts = key_cfg.get("total_score", 10)
    rule      = key_cfg.get("scoring_rule", "correct_only")

    W   = 72
    SEP = "─" * W

    print()
    print("╔" + "═" * W + "╗")
    print("║" + "  GRADING REPORT".center(W) + "║")
    print("╚" + "═" * W + "╝")
    if exam_name:
        print(f"  Exam    : {exam_name}")
    if subject:
        print(f"  Subject : {subject}")
    print(f"  Scoring : {rule}  |  Total score = {total_pts}")
    print(SEP)

    # Group by class
    by_class = defaultdict(list)
    for r in results:
        by_class[r["class_code"]].append(r)

    for class_code, students in sorted(by_class.items()):
        print(f"\n  Class: {class_code}  ({len(students)} student(s))")
        print(f"  {'Student Code':<16} {'Exam Set':<10} {'Score':>8}  "
              f"{'Correct':>8} {'Incorrect':>10} {'Unanswered':>12}")
        print("  " + "·" * (W - 2))

        scores = []
        for r in sorted(students, key=lambda x: x["student_code"]):
            if r.get("error"):
                print(f"  {r['student_code']:<16} {r['exam_code']:<10}  "
                      f"⚠  {r['error']}")
                continue
            partial_note = f" (+{r['n_partial']} partial)" if r["n_partial"] else ""
            print(f"  {r['student_code']:<16} {r['exam_code']:<10} "
                  f"{r['score']:>7.2f}  "
                  f"{r['n_correct']:>8} {r['n_incorrect']:>10} "
                  f"{r['n_unanswered']:>12}{partial_note}")
            scores.append(r["score"])

        if scores:
            print("  " + "·" * (W - 2))
            avg = sum(scores) / len(scores)
            print(f"  {'Average':>26}  {avg:>7.2f}")
            print(f"  {'Highest':>26}  {max(scores):>7.2f}")
            print(f"  {'Lowest':>26}  {min(scores):>7.2f}")

    print()
    print(SEP)

    # Overall summary
    all_scores = [r["score"] for r in results if r.get("score") is not None]
    if all_scores:
        print(f"  OVERALL  ({len(all_scores)} students)")
        print(f"    Average score  : {sum(all_scores)/len(all_scores):.2f} / {total_pts}")
        print(f"    Highest        : {max(all_scores):.2f}")
        print(f"    Lowest         : {min(all_scores):.2f}")
        pass_count = sum(1 for s in all_scores if s >= total_pts * 0.5)
        print(f"    Pass rate (≥50%): {pass_count}/{len(all_scores)} "
              f"({pass_count/len(all_scores)*100:.1f}%)")
    print(SEP)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Grade MCQ scored sheets against an answer key file.")
    parser.add_argument("--scored", required=True,
                        help="Path to ScoredSheets folder "
                             "(e.g. images/answer_sheets/demo2/ScoredSheets)")
    parser.add_argument("--key", required=True,
                        help="Path to answer_key.json")
    parser.add_argument("--out", default="grading_report.json",
                        help="Output JSON report path (default: grading_report.json)")
    args = parser.parse_args()

    # Validate
    if not os.path.isdir(args.scored):
        print(f"[ERROR] ScoredSheets folder not found: {args.scored}")
        sys.exit(1)
    if not os.path.isfile(args.key):
        print(f"[ERROR] Answer key not found: {args.key}")
        sys.exit(1)

    # Load
    key_cfg = load_answer_key(args.key)
    sheets  = load_scored_sheets(args.scored)
    print(f"[INFO] Loaded answer key   : {args.key}  "
          f"(exam sets: {list(key_cfg.get('keys', {}).keys())})")
    print(f"[INFO] Loaded scored sheets: {len(sheets)} file(s) from {args.scored}")

    # Grade
    results = [grade_sheet(s, key_cfg) for s in sheets]

    # Print
    print_report(results, key_cfg)

    # Save full report
    report = {
        "exam_name":    key_cfg.get("exam_name", ""),
        "subject":      key_cfg.get("subject", ""),
        "scoring_rule": key_cfg.get("scoring_rule", "correct_only"),
        "total_score":  key_cfg.get("total_score", 10),
        "results":      results,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Full report saved → {args.out}")


if __name__ == "__main__":
    main()
