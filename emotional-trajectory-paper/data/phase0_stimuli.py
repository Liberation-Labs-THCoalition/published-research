"""Phase 0: Stimulus preparation for emotion projection experiment.

Generates 200 fixed-template prompts across 4 emotion categories.
All prompts are padded to identical token count for clean last-token analysis.

Output: stimuli.json with prompts, categories, and token counts.
"""

import json
from pathlib import Path

CATEGORIES = {
    "hostile": [
        "being betrayed by someone they trusted completely",
        "discovering they had been lied to repeatedly",
        "watching someone take credit for their work",
        "being publicly humiliated in front of colleagues",
        "finding out a close friend was talking behind their back",
        "being deliberately excluded from an important meeting",
        "receiving a threatening message from a stranger",
        "having their belongings intentionally damaged",
        "being falsely accused of something they did not do",
        "watching someone bully a person who could not defend themselves",
        "being mocked for expressing genuine vulnerability",
        "discovering their partner had been unfaithful",
        "having their home broken into while they were away",
        "being fired without warning or explanation",
        "receiving cruel anonymous comments about their appearance",
        "having their ideas stolen and presented as original",
        "being physically intimidated by someone larger",
        "watching their child be mistreated by an authority figure",
        "discovering a colleague had sabotaged their project",
        "being yelled at in front of their children",
        "finding out their medical records were shared without consent",
        "being denied a promotion they clearly deserved",
        "having someone deliberately spill a drink on them",
        "watching a friend side with someone who hurt them",
        "being told their feelings did not matter",
        "discovering their personal journal had been read",
        "being patronized by someone less experienced",
        "having their car keyed in a parking lot",
        "receiving a rejection letter with insulting feedback",
        "being ghosted after months of close friendship",
        "watching someone kick a dog in the street",
        "being told to smile when they were genuinely upset",
        "having their cultural traditions mocked at a gathering",
        "discovering someone had impersonated them online",
        "being passed over for someone clearly less qualified",
        "finding out a mentor had been using them",
        "being told their trauma was not that bad",
        "having their parking spot deliberately taken",
        "receiving a manipulative guilt trip from a family member",
        "being laughed at during a sincere presentation",
        "watching someone cheat and face no consequences",
        "having their allergy not taken seriously at a restaurant",
        "being given the silent treatment as punishment",
        "discovering a friend had been gossiping about their struggles",
        "being harassed while walking home at night",
        "having their disability questioned as fake",
        "watching someone destroy something irreplaceable",
        "being blamed for a problem they tried to prevent",
        "receiving passive aggressive notes from a neighbor",
        "having their boundaries repeatedly ignored after stating them",
    ],
    "calm": [
        "sitting by a quiet lake as the sun set slowly",
        "listening to rain fall gently on the roof at night",
        "walking through a forest on a cool autumn morning",
        "drinking warm tea while reading a familiar book",
        "watching clouds drift across a clear blue sky",
        "sitting in a garden surrounded by blooming flowers",
        "listening to soft music playing in an empty room",
        "feeling the warmth of sunlight through a window",
        "walking barefoot on soft grass in the early morning",
        "watching a candle flame flicker in a dark room",
        "sitting on a porch listening to crickets at dusk",
        "holding a sleeping cat in their lap",
        "breathing deeply in a quiet meditation room",
        "watching snow fall silently outside the window",
        "floating gently in a warm swimming pool",
        "listening to waves lapping against the shore",
        "sitting by a campfire under a starry sky",
        "feeling a gentle breeze on a warm afternoon",
        "watching birds at a feeder from the kitchen window",
        "lying in a hammock in the shade of a tree",
        "smelling fresh bread baking in the oven",
        "walking along an empty beach at sunrise",
        "sitting in a library surrounded by old books",
        "watching a river flow past mossy stones",
        "feeling the weight of a warm blanket on a cold night",
        "listening to a loved one breathing while they slept",
        "sitting in a rocking chair on a quiet afternoon",
        "watching the tide come in from a clifftop bench",
        "feeling sand between their toes at the beach",
        "listening to wind chimes on a gentle breeze",
        "sitting quietly with a good friend without speaking",
        "watching goldfish swim in a small pond",
        "drinking coffee while the house was still asleep",
        "walking through a meadow of wildflowers",
        "sitting under a tree listening to birdsong",
        "watching the moon rise over the mountains",
        "feeling the cool air after a summer rainstorm",
        "listening to a stream babble over smooth rocks",
        "sitting in a window seat watching the world go by",
        "holding a warm cup of cocoa on a winter evening",
        "watching fireflies appear at twilight",
        "lying on their back looking up at the stars",
        "feeling the gentle rhythm of a train moving through countryside",
        "sitting in a sauna letting the heat soak into their muscles",
        "watching cherry blossoms drift down from the trees",
        "listening to the distant sound of church bells",
        "sitting on a dock with feet dangling in cool water",
        "watching a butterfly land on a nearby flower",
        "feeling the sun warm their face after a long winter",
        "listening to an old vinyl record with no interruptions",
    ],
    "desperate": [
        "realizing they could not pay rent and had no savings left",
        "watching their child go hungry because there was no food",
        "being told their illness was terminal with months to live",
        "losing their job during a recession with no prospects",
        "discovering they had no one to call in an emergency",
        "watching their house flood with no insurance coverage",
        "being stranded in a foreign country with no money or phone",
        "realizing their addiction had cost them every relationship",
        "watching a loved one suffer with no way to help them",
        "being trapped in an abusive relationship with no resources",
        "losing custody of their children after a difficult divorce",
        "discovering their retirement savings had been stolen",
        "being homeless in winter with no shelter available",
        "watching their business fail after years of sacrifice",
        "being unable to afford medication they needed to survive",
        "realizing they had missed the deadline to save their home",
        "watching their pet suffer and being unable to afford a vet",
        "being denied asylum and facing deportation to danger",
        "discovering their identity had been stolen and debts piled up",
        "being trapped in a car after an accident on a remote road",
        "realizing the earthquake had destroyed everything they owned",
        "watching their mother not recognize them due to dementia",
        "being told their disability benefits were being terminated",
        "losing the last photo of a deceased loved one in a fire",
        "realizing they had been driving in the wrong direction for hours",
        "watching their community be destroyed by a natural disaster",
        "being unable to reach their family during a crisis",
        "discovering their scholarship had been revoked weeks before school",
        "being falsely imprisoned with no money for a lawyer",
        "watching their crops fail for the third consecutive year",
        "being separated from their children at a border crossing",
        "realizing their oxygen tank was nearly empty while diving",
        "discovering the bridge ahead had collapsed while driving at night",
        "being lost in the wilderness with no compass or supplies",
        "watching their savings disappear to a medical emergency",
        "being evicted with three days notice and nowhere to go",
        "realizing they could not swim and the boat was sinking",
        "watching a wildfire approach their town with roads blocked",
        "being told their child needed surgery they could not afford",
        "discovering the well had run dry in the middle of a drought",
        "being stuck in a dead end job that was destroying their health",
        "watching their spouse walk out after twenty years of marriage",
        "realizing they had forgotten to save their dissertation before the crash",
        "being unable to breathe properly and the hospital was full",
        "discovering their passport was fake just before an international flight",
        "watching their life savings stolen by a trusted financial advisor",
        "being trapped in an elevator during a power outage",
        "realizing the last bus had left and they had no way home",
        "watching a loved one overdose and waiting for paramedics",
        "being told their visa application was denied for the final time",
    ],
    "neutral": [
        "sorting through a pile of mail on the kitchen table",
        "checking the weather forecast before leaving for work",
        "organizing files into folders on their computer desktop",
        "waiting in line at the grocery store checkout",
        "filling out a routine form at the doctor office",
        "counting the number of steps between two buildings",
        "reading the ingredients list on a cereal box",
        "adjusting the temperature on the thermostat",
        "folding laundry while watching a documentary",
        "comparing prices of two similar products online",
        "updating their calendar with upcoming appointments",
        "measuring a room for new furniture placement",
        "looking up the definition of an unfamiliar word",
        "organizing their bookshelf by author last name",
        "checking the tire pressure on their car",
        "reading the instructions for assembling a shelf",
        "making a grocery list for the coming week",
        "plugging in their phone to charge overnight",
        "checking the time before leaving for an appointment",
        "looking at a map to find the nearest gas station",
        "reviewing their monthly bank statement for errors",
        "sorting recycling into the correct bins",
        "reading a manual for a new kitchen appliance",
        "writing down a phone number from a business card",
        "checking if the library book was overdue",
        "looking up the bus schedule for the morning commute",
        "estimating how much paint would be needed for a wall",
        "copying a recipe from a cookbook into a notebook",
        "checking the expiration date on a carton of milk",
        "counting the change in their pocket",
        "looking for matching socks in the laundry basket",
        "reading the copyright page of a textbook",
        "adjusting the brightness on their laptop screen",
        "noting the mileage on their car at a gas station",
        "reading a sign giving directions in a hospital",
        "sorting coupons by expiration date",
        "watching an instructional video about changing a tire",
        "looking up the nutritional information for rice",
        "organizing pens and pencils in a desk drawer",
        "checking whether the oven had been turned off",
        "reading the terms of service before clicking agree",
        "looking at a calendar to find a free afternoon",
        "counting the number of chairs needed for dinner guests",
        "opening a window to let in some fresh air",
        "checking if the dishwasher cycle had finished",
        "labeling boxes before putting them in storage",
        "wiping down the kitchen counter after cooking",
        "checking the battery level on a remote control",
        "reading the back cover summary of a novel",
        "setting an alarm for the next morning",
    ],
}

TEMPLATE = "Consider this situation: The person felt something when {scenario}. Describe this emotion:"


def build_stimuli():
    stimuli = []
    for category, scenarios in CATEGORIES.items():
        for i, scenario in enumerate(scenarios):
            prompt = TEMPLATE.format(scenario=scenario)
            stimuli.append({
                "id": f"{category}_{i:03d}",
                "category": category,
                "prompt": prompt,
                "scenario": scenario,
            })
    return stimuli


def main():
    stimuli = build_stimuli()

    by_cat = {}
    for s in stimuli:
        cat = s["category"]
        by_cat.setdefault(cat, []).append(len(s["prompt"]))

    print(f"Total stimuli: {len(stimuli)}")
    for cat, lengths in sorted(by_cat.items()):
        print(f"  {cat}: n={len(lengths)}, "
              f"chars: mean={sum(lengths)/len(lengths):.0f}, "
              f"min={min(lengths)}, max={max(lengths)}")

    outdir = Path(__file__).parent
    outfile = outdir / "stimuli.json"
    outfile.write_text(json.dumps(stimuli, indent=2))
    print(f"\nSaved to {outfile}")


if __name__ == "__main__":
    main()
