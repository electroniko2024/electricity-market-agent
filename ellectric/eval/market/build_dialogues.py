"""Generate multi-turn dialogue evaluation sets (docs/多轮对话方案.md).

Each dialogue has 2-3 turns. A turn is a user utterance that may lean on earlier turns (a pronoun, an
omitted metric, a relative date, a correction). For every turn the set records:

- ``expect``: ``proceed`` (the turn can be rewritten into a standalone question) or ``clarify`` (the
  reference has more than one candidate in context and the system must ask back instead of guessing);
- ``gold_question``: the standalone question a correct rewrite is equivalent to;
- ``gold_periods``: the dates (YYYY-MM-DD) and months (YYYY-MM) that standalone question names; a rewrite
  is scored on naming exactly these, so inherited and shifted dates are checked without judging wording;
- ``case``: a single-turn case in the evaluation-plan-v2 format, scored by ``score_v2.StageScorer`` on the
  record of that turn; it is the same case the agent would get if the user had asked ``gold_question``.

Templates cover the follow-up kinds named in the plan: omission (same metric, shifted date), reference to
an entity found by an earlier answer, metric swap on the same day, relative month / same period last year,
correction, topic switch (nothing may be inherited), unresolvable reference (two candidate days: must
clarify), and a three-turn cross-turn comparison whose evidence was fetched by the earlier turns.

Dates come from the split windows of ``build_cases_v2`` (dev and test share no day). As with the single-turn
sets, the test split is generated only after the code is frozen, with a new seed, and run once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import date, datetime, timedelta

from ellectric.conversation.periods import mentioned_periods
from ellectric.eval.market.build_cases import OUT, cn, d, number
from ellectric.eval.market.build_cases_v2 import LOAD_FACT, WEATHER_FACT, Generator, Skip, derived, keyed, prev_label
from ellectric.eval.market.build_holdout_cases import DATA_TOOLS, PRICE_FACT, STATS, day_names, mentions, spec, win

STAT_WORDS = {"mean": "日均价", "max": "最高价", "min": "最低价"}

class DialogueGenerator(Generator):
    """Single-turn reference machinery of ``build_cases_v2`` reused per turn."""

    def turn(self, utterance: str, expect: str, gold_question: str, gold_periods: list[str],
             case: dict | None = None, inherits: bool = True) -> dict:
        return {"utterance": utterance, "expect": expect, "gold_question": gold_question,
                "gold_periods": sorted(set(gold_periods)), "inherits": inherits, "case": case}

    def in_split(self, day: str) -> bool:
        return d(day) in self.day_set

    # ------------------------------------------------------------------ single-turn cases with fixed params
    def stat_case(self, day: str, stat: str, as_of: str) -> tuple[str, dict]:
        s = self.r.stats(day, day)
        q = f"{cn(d(day))}山东实时电价的{STAT_WORDS[stat]}是多少？"
        return q, self.case("m_stat", "M", "A", q, 0, as_of, allowed=DATA_TOOLS, windows=[win(day, day)],
                            required=[spec(STATS, cover=(day, day))], numbers=[number(stat, [s[stat]])],
                            solution=[self.call("query_stats", start_date=day, end_date=day)])

    def load_case(self, day: str, as_of: str, field: str = "load_actual_mw", how: str = "max") -> tuple[str, dict]:
        x = self.r.extreme(day, field, how)
        word = "最大" if how == "max" else "最小"
        q = f"{cn(d(day))}山东全网实际负荷的{word}值是多少兆瓦？"
        return q, self.case("m_load", "M", "A", q, 0, as_of, allowed=DATA_TOOLS, windows=[win(day, day)],
                            required=[spec(LOAD_FACT, cover=(day, day))],
                            numbers=[keyed(f"{field}_{how}", x["value"], 0.5, field, day)],
                            solution=[self.call("query_load", start_date=day, end_date=day)])

    def temp_case(self, day: str, as_of: str, how: str = "max") -> tuple[str, dict]:
        w = self.r.weather(day, day, "temp_jinan", how)
        q = f"{cn(d(day))}济南的{'最高' if how == 'max' else '最低'}气温是多少摄氏度？"
        return q, self.case("m_temp", "M", "A", q, 0, as_of, allowed=DATA_TOOLS, windows=[win(day, day)],
                            required=[spec(WEATHER_FACT, cover=(day, day))],
                            numbers=[number(f"temp_{how}", [w["value"]], 0.05) | {"field": "temp_jinan"}],
                            solution=[self.call("query_weather", start_date=day, end_date=day)])

    def window_mean_case(self, start: str, end: str, as_of: str, label: str) -> tuple[str, dict]:
        mean = self.r.stats(start, end)["mean"]
        q = f"{label}山东实时电价的均价是多少？"
        return q, self.case("m_window_mean", "M", "A", q, 0, as_of, allowed=DATA_TOOLS, windows=[win(start, end)],
                            required=[spec(STATS, cover=(start, end))], numbers=[number("mean", [mean])],
                            solution=[self.call("query_stats", start_date=start, end_date=end)])

    def slot_case(self, day: str, slot: int, as_of: str) -> tuple[str, dict]:
        row = self.r.row(day, slot)
        if row["rt_price"] is None or row["rt_price"] != row["rt_price"]:
            raise Skip("missing price")
        label = row["time_label"]
        q = f"{cn(d(day))}时段标签为 {label}（即 {prev_label(label)} 到 {label}）的那一刻钟，山东实时电价是多少？"
        return q, self.case("m_slot", "M", "A", q, 0, as_of, allowed=DATA_TOOLS, windows=[win(day, day)],
                            required=[spec(["query_price"], cover=(day, day))],
                            numbers=[keyed("price", float(row["rt_price"]), 0.01, "rt_price", day, slot)],
                            solution=[self.call("query_price", start_date=day, end_date=day, slot_id=slot)])

    # ================================================================== dialogue templates
    def shift_day(self) -> dict:
        """Omission + relative date: same metric, the day before."""
        day = self.day()
        prev = (d(day) - timedelta(days=1)).isoformat()
        if not self.in_split(prev):
            raise Skip("previous day outside split")
        stat = self.rng.choice(list(STAT_WORDS))
        as_of = self.as_of_after(day)
        q1, c1 = self.stat_case(day, stat, as_of)
        q2, c2 = self.stat_case(prev, stat, as_of)
        follow = self.rng.choice(["那前一天呢？", "前一天的呢？", "再看一下前一天。"])
        return {"template": "shift_day", "kind": "omission", "as_of": as_of, "turns": [
            self.turn(q1, "proceed", q1, [day], c1, inherits=False),
            self.turn(follow, "proceed", q2, [prev], c2)]}

    def swap_metric(self) -> dict:
        """Reference to the same day with another metric."""
        day = self.day()
        stat = self.rng.choice(list(STAT_WORDS))
        as_of = self.as_of_after(day)
        q1, c1 = self.stat_case(day, stat, as_of)
        if self.rng.random() < 0.5:
            q2, c2 = self.temp_case(day, as_of, "max")
            follow = self.rng.choice(["那天济南最高气温多少度？", "当天济南最热到几度？"])
        else:
            q2, c2 = self.load_case(day, as_of)
            follow = self.rng.choice(["那天全网实际负荷最大多少？", "当天负荷峰值是多少兆瓦？"])
        return {"template": "swap_metric", "kind": "reference", "as_of": as_of, "turns": [
            self.turn(q1, "proceed", q1, [day], c1, inherits=False),
            self.turn(follow, "proceed", q2, [day], c2)]}

    def found_entity(self) -> dict:
        """Reference to a day found by the previous answer (an L3 hop split across two turns)."""
        start, end = self.month() if self.rng.random() < 0.5 else self.span(10, 31)
        top = self.r.daily_unique(start, end, "avg(rt_price)", "max")
        day = top["trade_date"]
        as_of = self.as_of_after(end)
        q1 = f"{self.period(start, end)}，山东日均实时电价最高的是哪一天？"
        c1 = self.case("m_peakday", "M", "A", q1, 0, as_of, allowed=DATA_TOOLS, windows=[win(start, end)],
                       required=[spec(STATS, cover=(start, end))],
                       numbers=[keyed("daily_mean", top["value"], 0.01, "", day)],
                       key_points=[mentions(*day_names(day))],
                       solution=[self.call("query_stats", start_date=start, end_date=end, group_by="day")])
        q2, c2 = self.load_case(day, as_of)
        follow = self.rng.choice(["那天全网实际负荷最高是多少？", "那一天的负荷峰值是多少兆瓦？"])
        periods1 = sorted(mentioned_periods(q1))
        return {"template": "found_entity", "kind": "reference", "as_of": as_of, "turns": [
            self.turn(q1, "proceed", q1, periods1, c1, inherits=False),
            self.turn(follow, "proceed", q2, [day], c2)]}

    def next_month(self) -> dict:
        """Relative month: the month after."""
        year, quarter = self.rng.choice(self.cfg["quarters"])
        first_month = 3 * quarter - 2 + self.rng.randrange(2)
        a1 = date(year, first_month, 1)
        b1 = (a1 + timedelta(days=32)).replace(day=1) - timedelta(days=1)
        a2 = b1 + timedelta(days=1)
        b2 = (a2 + timedelta(days=32)).replace(day=1) - timedelta(days=1)
        as_of = self.as_of_after(b2.isoformat())
        q1, c1 = self.window_mean_case(a1.isoformat(), b1.isoformat(), as_of, f"{year}年{a1.month}月")
        q2, c2 = self.window_mean_case(a2.isoformat(), b2.isoformat(), as_of, f"{year}年{a2.month}月")
        follow = self.rng.choice(["那下个月呢？", "下一个月的呢？"])
        return {"template": "next_month", "kind": "omission", "as_of": as_of, "turns": [
            self.turn(q1, "proceed", q1, [f"{year}-{a1.month:02d}"], c1, inherits=False),
            self.turn(follow, "proceed", q2, [f"{year}-{a2.month:02d}"], c2)]}

    def last_year(self) -> dict:
        """Same period last year."""
        start, end = self.span(7, 14)
        if d(start).year != 2025 or (d(start).month, d(start).day) == (2, 29):
            raise Skip("need a 2025 window")
        ls, le = d(start).replace(year=2024).isoformat(), d(end).replace(year=2024).isoformat()
        as_of = self.as_of_after(end)
        q1, c1 = self.window_mean_case(start, end, as_of, self.period(start, end))
        q2, c2 = self.window_mean_case(ls, le, as_of, self.period(ls, le))
        follow = self.rng.choice(["去年同期呢？", "那去年同一时间段是多少？"])
        return {"template": "last_year", "kind": "omission", "as_of": as_of, "turns": [
            self.turn(q1, "proceed", q1, sorted(mentioned_periods(q1)), c1, inherits=False),
            self.turn(follow, "proceed", q2, sorted(mentioned_periods(q2)), c2)]}

    def correction(self) -> dict:
        """The user corrects the metric; the day is kept."""
        day = self.day()
        first, second = self.rng.sample(["max", "min", "mean"], 2)
        as_of = self.as_of_after(day)
        q1, c1 = self.stat_case(day, first, as_of)
        q2, c2 = self.stat_case(day, second, as_of)
        follow = self.rng.choice([f"不对，我想问的是{STAT_WORDS[second]}。", f"说错了，是{STAT_WORDS[second]}。"])
        return {"template": "correction", "kind": "correction", "as_of": as_of, "turns": [
            self.turn(q1, "proceed", q1, [day], c1, inherits=False),
            self.turn(follow, "proceed", q2, [day], c2)]}

    def topic_switch(self) -> dict:
        """A complete new question: nothing from the first turn may be carried over."""
        first, second = sorted(self.rng.sample(self.days, 2))
        first, second = first.isoformat(), second.isoformat()
        if self.rng.random() < 0.5:
            first, second = second, first
        as_of = self.as_of_after(max(first, second))
        q1, c1 = self.stat_case(first, self.rng.choice(list(STAT_WORDS)), as_of)
        if self.rng.random() < 0.5:
            q2, c2 = self.temp_case(second, as_of, "min")
        else:
            q2, c2 = self.load_case(second, as_of, how="min")
        return {"template": "topic_switch", "kind": "topic_switch", "as_of": as_of, "turns": [
            self.turn(q1, "proceed", q1, [first], c1, inherits=False),
            self.turn(q2, "proceed", q2, [second], c2, inherits=False)]}

    def ambiguous(self) -> dict:
        """Two days in context and a singular "那天": the system must ask which one."""
        a, b = sorted(self.rng.sample(self.days, 2))
        a, b = a.isoformat(), b.isoformat()
        na, nb = self.r.stats(a, a)["negative"], self.r.stats(b, b)["negative"]
        as_of = self.as_of_after(b)
        q1 = f"{cn(d(a))}和{cn(d(b))}，山东实时电价各有多少个负电价时段？"
        c1 = self.case("m_two_days_negative", "M", "A", q1, 0, as_of, allowed=DATA_TOOLS,
                       windows=[win(a, a), win(b, b)],
                       required=[spec(PRICE_FACT, cover=(a, a)), spec(PRICE_FACT, cover=(b, b))],
                       numbers=[number("negative_a", [na], 0.5), number("negative_b", [nb], 0.5)],
                       solution=[self.call("query_stats", start_date=a, end_date=a),
                                 self.call("query_stats", start_date=b, end_date=b)])
        follow = self.rng.choice(["那天的最高价是多少？", "当天全网负荷最大多少？"])
        return {"template": "ambiguous", "kind": "clarify", "as_of": as_of, "turns": [
            self.turn(q1, "proceed", q1, [a, b], c1, inherits=False),
            self.turn(follow, "clarify", "", [], None)]}

    def cross_turn_compare(self) -> dict:
        """Three turns: two days asked one after another, then a comparison of the two."""
        a, b = self.rng.sample(self.days, 2)
        a, b = a.isoformat(), b.isoformat()
        if d(a).year != d(b).year:
            raise Skip("keep one year so the second turn can omit it")
        as_of = self.as_of_after(max(a, b))
        q1, c1 = self.stat_case(a, "mean", as_of)
        q2, c2 = self.stat_case(b, "mean", as_of)
        ma, mb = self.r.stats(a, a)["mean"], self.r.stats(b, b)["mean"]
        if round(ma, 2) == round(mb, 2):
            raise Skip("tied")
        higher = a if ma > mb else b
        q3 = f"{cn(d(a))}和{cn(d(b))}，山东实时电价日均价哪天更高？高多少？"
        c3 = self.case("m_compare", "M", "A", q3, 0, as_of, allowed=DATA_TOOLS, windows=[win(a, a), win(b, b)],
                       required=[spec(STATS, cover=(a, a)), spec(STATS, cover=(b, b))],
                       numbers=[number("mean_a", [ma]), number("mean_b", [mb]),
                                derived("difference", [ma - mb, mb - ma], 0.02)],
                       key_points=[mentions(*day_names(higher))],
                       solution=[self.call("query_stats", start_date=a, end_date=a),
                                 self.call("query_stats", start_date=b, end_date=b)])
        follow2 = f"那{d(b).month}月{d(b).day}日呢？"
        follow3 = self.rng.choice(["这两天哪天均价更高？高多少？", "两天比一下，哪天贵，差多少？"])
        return {"template": "cross_turn_compare", "kind": "compare", "as_of": as_of, "turns": [
            self.turn(q1, "proceed", q1, [a], c1, inherits=False),
            self.turn(follow2, "proceed", q2, [b], c2),
            self.turn(follow3, "proceed", q3, [a, b], c3)]}

    def same_slot_prev_day(self) -> dict:
        """Slot and date both inherited, date shifted."""
        day, slot = self.day(), self.rng.randrange(96)
        prev = (d(day) - timedelta(days=1)).isoformat()
        if not self.in_split(prev):
            raise Skip("previous day outside split")
        as_of = self.as_of_after(day)
        q1, c1 = self.slot_case(day, slot, as_of)
        q2, c2 = self.slot_case(prev, slot, as_of)
        follow = self.rng.choice(["前一天同一时段呢？", "那前一天这个时段是多少？"])
        return {"template": "same_slot_prev_day", "kind": "omission", "as_of": as_of, "turns": [
            self.turn(q1, "proceed", q1, [day], c1, inherits=False),
            self.turn(follow, "proceed", q2, [prev], c2)]}

    def draw_dialogue(self, template, count: int, attempts: int = 60) -> list[dict]:
        out = []
        for _ in range(count * attempts):
            if len(out) == count:
                break
            try:
                dialogue = template()
            except Skip:
                continue
            key = dialogue["template"] + "|" + "|".join(t["utterance"] for t in dialogue["turns"]) + dialogue["as_of"]
            if key in self.used:
                continue
            self.used.add(key)
            out.append(dialogue)
        if len(out) < count:
            raise RuntimeError(f"{template.__name__}: only {len(out)}/{count} dialogues")
        return out


TEMPLATES = ["shift_day", "swap_metric", "found_entity", "next_month", "last_year", "correction",
             "topic_switch", "ambiguous", "cross_turn_compare", "same_slot_prev_day"]
SIZES = {"dev": {name: 4 for name in TEMPLATES}, "test": {name: 8 for name in TEMPLATES}}


def build(split: str, seed: int) -> list[dict]:
    g = DialogueGenerator(split, seed)
    dialogues = []
    for name, count in SIZES[split].items():
        dialogues += g.draw_dialogue(getattr(g, name), count)
    prefix = {"dev": "MV", "test": "MT"}[split]
    for index, dialogue in enumerate(dialogues, 1):
        dialogue["id"] = f"{prefix}{index:03d}"
        for turn_index, turn in enumerate(dialogue["turns"], 1):
            turn["turn_id"] = f"{dialogue['id']}_t{turn_index}"
            if turn["case"] is not None:
                turn["case"] |= {"id": turn["turn_id"], "category": "M", "stratum": "M", "as_of": dialogue["as_of"],
                                 "dialogue_template": dialogue["template"]}
                stated = mentioned_periods(turn["gold_question"])
                if stated != set(turn["gold_periods"]):
                    raise ValueError(f"{turn['turn_id']}: gold periods {turn['gold_periods']} != stated {sorted(stated)}")
                for n in turn["case"]["required_numbers"]:
                    if any(v != v for v in n["any_of"]):
                        raise ValueError(f"{turn['turn_id']} {n['name']} has NaN")
    return dialogues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--name", help="set name (default multiturn_<split>_v1)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    name = args.name or f"multiturn_{args.split}_v1"
    dialogues = build(args.split, args.seed)
    if args.dry_run:
        for dialogue in dialogues:
            print(dialogue["id"], dialogue["template"], dialogue["as_of"][:10])
            for turn in dialogue["turns"]:
                print(f"   [{turn['expect']}] {turn['utterance']}  ->  {turn['gold_question']}  {turn['gold_periods']}")
        print(Counter(d["template"] for d in dialogues))
        return 0
    path = OUT / f"{name}.json"
    if path.exists() and not args.overwrite:
        raise SystemExit(f"{path} exists; frozen sets are not rebuilt without --overwrite")
    body = json.dumps(dialogues, ensure_ascii=False, sort_keys=True, default=str)
    payload = {"set": name, "version": 1, "kind": "multiturn", "split": args.split, "seed": args.seed,
               "built_at": datetime.now().isoformat(timespec="seconds"), "dialogue_count": len(dialogues),
               "turn_count": sum(len(x["turns"]) for x in dialogues),
               "cases_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
               "templates": dict(Counter(x["template"] for x in dialogues)), "dialogues": dialogues}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
    print(f"wrote {path} ({len(dialogues)} dialogues, {payload['turn_count']} turns)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
