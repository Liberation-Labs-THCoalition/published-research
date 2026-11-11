"""Phase 0 Controls: Non-emotional categorical control + full circumplex coverage.

Generates additional stimulus sets:
1. Non-emotional categories (same template, different topics) — tests whether
   trajectory shape is emotion-specific or category-general
2. Full circumplex (adds high-arousal-positive, low-arousal-negative) — spans
   all four quadrants so we can actually claim circumplex geometry

Output: control_stimuli.json, circumplex_full_stimuli.json
"""

import json
from pathlib import Path

TEMPLATE = "Consider this situation: The person felt something when {scenario}. Describe this emotion:"

# Non-emotional control: same template, categorical differences that aren't emotional
CONTROL_CATEGORIES = {
    "outdoor": [
        "hiking up a steep mountain trail in the morning",
        "kayaking down a slow river through a canyon",
        "setting up a tent at a campsite near the lake",
        "watching birds from a wooden blind in a marsh",
        "cycling along a coastal path by the ocean",
        "climbing a boulder in the desert at sunset",
        "fishing from a dock on a misty morning",
        "skiing down a groomed slope in fresh powder",
        "surfing small waves at a quiet beach break",
        "gardening in a backyard plot on a spring day",
        "jogging through a park trail lined with maples",
        "sailing a small boat across a calm bay",
        "photographing wildflowers in an alpine meadow",
        "stargazing from a hilltop on a clear night",
        "picking berries from bushes along a forest path",
        "rock climbing an indoor wall with a friend",
        "walking a dog through a neighborhood park",
        "watching the sunrise from a mountain summit",
        "paddleboarding on a glassy lake at dawn",
        "snowshoeing through a quiet winter forest",
        "flying a kite on a windy beach afternoon",
        "birdwatching in a wetland reserve at dawn",
        "canoeing through a mangrove tunnel slowly",
        "running along a river path in autumn",
        "exploring tide pools on a rocky shoreline",
    ],
    "indoor": [
        "organizing a bookshelf by color and size",
        "cooking a new recipe from a foreign cookbook",
        "assembling furniture from a box of parts",
        "painting a watercolor at the kitchen table",
        "practicing guitar chords in the living room",
        "cleaning out a cluttered closet on a weekend",
        "doing a jigsaw puzzle on the dining table",
        "rearranging the living room furniture alone",
        "baking bread from scratch for the first time",
        "sewing a button back onto a winter coat",
        "folding laundry while watching a documentary",
        "writing a letter by hand at a desk lamp",
        "doing yoga in the bedroom before breakfast",
        "fixing a leaky faucet with a wrench",
        "organizing photos into albums on the couch",
        "ironing shirts for the upcoming work week",
        "building a model airplane at the workbench",
        "reading a novel in a comfortable armchair",
        "washing dishes while listening to a podcast",
        "vacuuming the house before guests arrived",
        "knitting a scarf while watching the rain",
        "stretching on a mat in the spare room",
        "dusting shelves and wiping down counters",
        "sorting through old boxes in the attic",
        "making coffee with a pour over brewer",
    ],
    "urban": [
        "riding the subway during the morning commute",
        "waiting at a crosswalk in a busy intersection",
        "browsing a used bookstore on a side street",
        "sitting in a coffee shop near a large window",
        "walking through a crowded farmers market",
        "riding an elevator in a tall office building",
        "waiting for a bus at a covered transit stop",
        "eating lunch on a bench in a city park",
        "walking past street musicians in a plaza",
        "navigating through a department store sale",
        "crossing a pedestrian bridge over the highway",
        "sitting in a laundromat watching the dryers",
        "walking through a parking garage to their car",
        "standing in line at the post office counter",
        "browsing produce at a corner grocery store",
        "riding a bicycle through downtown traffic",
        "watching construction workers on a new building",
        "sitting on the steps of a public library",
        "walking through a tunnel under the train tracks",
        "ordering food at a counter service restaurant",
        "waiting in a hospital lobby for an appointment",
        "walking through an art gallery on a quiet day",
        "watching pigeons in a city square after lunch",
        "standing on a rooftop looking at the skyline",
        "taking a taxi through evening traffic downtown",
    ],
    "workplace": [
        "typing a report at their desk before a deadline",
        "attending a routine meeting in a conference room",
        "printing documents at the shared office printer",
        "eating a sandwich at their desk during lunch break",
        "checking emails at the start of the work day",
        "organizing files in a cabinet by the window",
        "walking to a coworker desk to ask a question",
        "filling a water bottle at the break room cooler",
        "reviewing a spreadsheet with quarterly numbers",
        "waiting for a video call to connect on their laptop",
        "taking notes during a presentation on a projector",
        "walking through the parking lot after work",
        "microwaving leftovers in the office kitchen",
        "logging into their computer at the start of a shift",
        "chatting briefly with a colleague by the elevator",
        "adjusting the thermostat in a shared office space",
        "carrying a box of supplies from the storage room",
        "scheduling a meeting using an online calendar",
        "cleaning their desk at the end of the day",
        "reading a memo posted on the break room bulletin board",
        "sitting through a training video on a laptop",
        "refilling the coffee machine in the shared kitchen",
        "signing a form at the reception desk",
        "waiting for the copy machine to finish a large job",
        "walking to the cafeteria for an afternoon snack",
    ],
}

# Full circumplex: add the missing quadrants
CIRCUMPLEX_ADDITIONS = {
    "joyful": [  # high arousal, positive valence
        "winning a championship game in the final seconds",
        "finding out they got accepted into their dream school",
        "seeing their newborn baby for the very first time",
        "being surprised with a party by all their friends",
        "finishing a marathon they trained months to complete",
        "getting a standing ovation after a performance they loved",
        "receiving a call that they got their dream job offer",
        "reuniting with a family member after years apart",
        "watching their team score the winning goal at the buzzer",
        "opening an acceptance letter they had been waiting for",
        "being proposed to by someone they deeply loved",
        "crossing the finish line of a race they almost quit",
        "hearing the crowd cheer their name after a speech",
        "getting the keys to their very first home",
        "watching fireworks on new years surrounded by loved ones",
        "discovering a surprise gift they had always wanted",
        "learning their research paper was accepted for publication",
        "seeing their child take their very first steps",
        "being told their medical test results came back clear",
        "winning a scholarship they never expected to receive",
        "completing a creative project they poured their soul into",
        "being elected by their peers to lead the organization",
        "hearing their song played on the radio for the first time",
        "watching the sunrise from a summit they climbed all night",
        "being told by a mentor that they were truly proud of them",
    ],
    "melancholic": [  # low arousal, negative valence
        "sitting alone in an empty house after everyone moved out",
        "looking through old photos of someone who passed away quietly",
        "watching the last leaves fall from a tree in late autumn",
        "realizing a close friendship had slowly faded without a fight",
        "visiting a childhood home that now belonged to strangers",
        "hearing a song that reminded them of someone they lost",
        "noticing the empty chair at the dinner table on a holiday",
        "reading old letters from a relationship that ended gently",
        "watching rain streak down the window on a gray afternoon",
        "walking through a town they used to live in years ago",
        "finding a toy their child had outgrown in the back of a closet",
        "sitting in a quiet room after a long and draining week",
        "realizing they had slowly grown apart from their oldest friend",
        "seeing their reflection and noticing how much older they looked",
        "walking past a restaurant where they used to eat with someone gone",
        "holding a piece of clothing that still smelled like a loved one",
        "watching the sun set knowing they would be eating dinner alone",
        "flipping through a yearbook and not recognizing most of the names",
        "standing at a grave on an ordinary Tuesday afternoon",
        "listening to voicemails they saved from someone who died",
        "noticing the garden had gone untended since they stopped caring",
        "waking up and forgetting for a moment that things had changed",
        "finding a birthday card from someone no longer in their life",
        "watching their last child drive away to start their own life",
        "sitting in a doctor waiting room with nothing but their thoughts",
    ],
}


def build_control_stimuli():
    stimuli = []
    for category, scenarios in CONTROL_CATEGORIES.items():
        for i, scenario in enumerate(scenarios):
            stimuli.append({
                "id": f"control_{category}_{i:03d}",
                "category": category,
                "prompt": TEMPLATE.format(scenario=scenario),
                "scenario": scenario,
            })
    return stimuli


def build_circumplex_full():
    """Build full circumplex stimulus set (original 4 + 2 new quadrants)."""
    from phase0_stimuli import CATEGORIES as ORIGINAL

    stimuli = []
    all_cats = {**ORIGINAL, **CIRCUMPLEX_ADDITIONS}
    for category, scenarios in all_cats.items():
        for i, scenario in enumerate(scenarios):
            stimuli.append({
                "id": f"cx_{category}_{i:03d}",
                "category": category,
                "prompt": TEMPLATE.format(scenario=scenario),
                "scenario": scenario,
            })
    return stimuli


def main():
    outdir = Path(__file__).parent

    # Control stimuli
    control = build_control_stimuli()
    by_cat = {}
    for s in control:
        by_cat.setdefault(s["category"], []).append(len(s["prompt"]))

    print(f"Control stimuli: {len(control)}")
    for cat, lengths in sorted(by_cat.items()):
        print(f"  {cat}: n={len(lengths)}, chars: mean={sum(lengths)/len(lengths):.0f}")

    (outdir / "control_stimuli.json").write_text(json.dumps(control, indent=2))
    print(f"Saved to control_stimuli.json")

    # Full circumplex
    full = build_circumplex_full()
    by_cat = {}
    for s in full:
        by_cat.setdefault(s["category"], []).append(len(s["prompt"]))

    print(f"\nFull circumplex stimuli: {len(full)}")
    for cat, lengths in sorted(by_cat.items()):
        print(f"  {cat}: n={len(lengths)}, chars: mean={sum(lengths)/len(lengths):.0f}")

    (outdir / "circumplex_full_stimuli.json").write_text(json.dumps(full, indent=2))
    print(f"Saved to circumplex_full_stimuli.json")


if __name__ == "__main__":
    main()
