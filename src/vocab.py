"""Ingredient categories for AaharAI.

Edit the sets HERE (not in a notebook) so changes survive Colab resets.

detectable : OWLv2 is asked to find these in photos; also used by the recommender
manual_add : real ingredients the user adds by hand (dals, flours, meats...); used by the recommender
staple     : assumed to be in the pantry (UI toggle); excluded from coverage scoring
ignore     : colours, essences, additives; excluded from coverage scoring
other      : not classified yet; counts AGAINST coverage, so don't dump things in 'ignore' to look good
"""
import re

# canonical-name fixes (applied after cleaning the INDB food name)
RENAME = {"lemons": "lemon", "siva's": "amchur", "tomatoes": "tomato", "beet root": "beetroot"}

STAPLE = {
    "salt", "water", "oil", "sugar", "ghee", "chilli powder", "pepper", "garam masala", "cumin seeds",
    "turmeric powder", "coriander seeds", "cardamom", "cloves", "cinnamon", "bay leaf", "mustard seeds",
    "asafoetida", "baking powder", "bicarbonate of soda", "mustard powder", "nutmeg", "fennel seeds",
    "ajwain seed whole organic spices", "himalayan black salt", "amchur", "fenugreek seeds", "dry red chilli",
    "tamarind", "vinegar", "oregano", "poppy seeds", "gingelly seeds", "citric acid", "saffron", "mace",
    "tea", "sesame seeds", "garlic powder", "thyme", "basil", "star anise", "paprika", "curry powder",
    "mixed curry spices", "balsamic vinegar", "rice vinegar", "tartaric acid", "mixed herbs", "rosemary",
    "cajun seasoning", "spices", "ground ginger", "cream of tartar", "dijon mustard", "sweet sunnah",
}

DETECT = {
    "onion", "tomato", "potato", "carrot", "capsicum", "cauliflower", "cabbage", "peas", "spinach", "cucumber",
    "lemon", "green chilli", "ginger", "garlic", "coriander leaves", "mint leaves", "curry leaves", "french beans",
    "button mushroom", "brinjal - all varieties", "bottle gourd", "pumpkin", "spring onions", "lettuce",
    "celery stalk", "fenugreek leaves", "parsley", "banana", "apple", "mango", "orange", "pineapple", "coconut",
    "egg", "paneer", "bread", "chicken", "milk", "yogurt", "butter", "cheese",
    "radish", "corn",      # added after the first OWLv2 photo test (radish appeared in 4 of 15 photos)
    "broccoli", "beetroot",  # added after the held-out test (broccoli was missed 10 times, beetroot 2)
}

MANUAL = {
    # flours, grains, pulses
    "wheat flour", "flour", "wheat", "rice", "rice flakes", "black gram", "bengal gram", "green gram", "red gram",
    "lentil dal", "cracked wheat", "bajra", "jowar", "ragi", "maize", "sago", "rice puffed", "rajmah", "soya bean",
    "chickpeas", "black beans", "beans", "porridge oats", "breakfast cereal", "couscous", "pasta", "pizza base",
    # dairy and similar
    "cream", "khoa", "margarine", "tofu", "coconut milk", "almond milk", "peanut butter", "custard powder",
    # meat and fish
    "mutton", "lamb", "pomfret", "ham", "bacon rashers", "mackerel", "tiger prawns", "rohu", "prawns", "salami",
    "sausages",
    # vegetables, fruit, leaves
    "turnip", "grapes", "shallots", "guava", "jack fruit", "ladies finger", "bitter gourd",
    "colocasia", "sweetcorn", "curly kale", "pomegranate", "bathua leaves", "yam", "bananas",
    "beansprouts", "peppers", "lemon grass", "lotus stem", "radish leaves", "amaranth leaves", "mustard leaves",
    "sweet potato", "papaya", "peach", "plum", "kiwi fruit", "litchi", "drumstick", "amla", "tinda", "watercress",
    "bamboo shoot", "lime", "pears", "apricot", "raspberries", "blueberries", "cranberries", "currants",
    "sultanas", "lemon peel", "orange peel", "olives", "capers", "gherkins", "pickle", "agar", "gelatine",
    # nuts, seeds, dried fruit
    "almond", "ground nut", "peanuts", "cashew nut", "pistachio nuts", "walnut", "hazelnuts", "pine seed",
    "pine nuts", "raisins", "dates", "mixed nuts cashews", "seeds", "amaranth seed", "pumpkin seeds",
    "sunflower seeds", "melon seeds", "lotus seeds", "dried mango", "fruit cocktail",
    # sauces, sweet things, misc
    "jaggery", "honey", "jam", "strawberry jam", "soy sauce", "chilli sauce", "tomato sauce", "tomato ketchup",
    "tomato puree", "mayonnaise", "worcestershire sauce", "mint sauce", "mustard sauce", "white sauce",
    "coffee", "chocolate", "cocoa powder", "drinking chocolate", "yeast extract", "ginger juice", "gulkand",
    "rose water", "rose petals",
    # distinctions that matter for matching (a detected 'milk' must not satisfy these)
    "buttermilk", "condensed milk", "breadcrumbs",
    # leftovers found on the real INDB data (kept manual rather than ignored, so coverage stays honest)
    "ice cream", "sour cream", "cheese spread", "lentil whole", "lotus root", "roasted chickpeas",
    "gooseberries", "moth bean", "cowpea", "chia seeds", "black rice", "buckwheat", "quinoa", "horse gram",
    "almond flour", "wood apple", "star fruit", "redcurrants", "strawberry", "pear", "marmalade", "jelly",
    "biscuits", "wafers", "hot dog buns", "white hamburger buns", "whole wheat hamburger buns", "dips", "sauces",
    "nuts", "pitted black olives", "apple juice", "spinach powder", "creamed corn", "pineapple syrup", "syrup",
    "dressing", "cola", "energy drink", "orange drink", "fruit juice drink/squash", "spirits", "candies",
    "tutti frutti gumballs", "nannari root/ indian sarsaparilla",
}

IGNORE = {
    "vanilla extract", "stock cubes", "whey", "cherries", "orange juice", "pineapple juice",
    "reetha", "msg monosodium glutamate", "kewra", "ammonium bicarbonate", "alum solution", "silver foil",
    "lemon jelly crystals",
}
# anything with colour / essence / flavour in its name is ignored (food colours, essences, flavourings)
_IGNORE_PATTERN = re.compile(r"\b(colou?r|essen[cs]e|essesnce|flavou?r)")


def category(name: str) -> str:
    if name in STAPLE:
        return "staple"
    if name in DETECT:
        return "detectable"
    if name in MANUAL:
        return "manual_add"
    if name in IGNORE or _IGNORE_PATTERN.search(name):
        return "ignore"
    return "other"


# Used to derive a vegetarian / non-vegetarian flag from a recipe's ingredients (INDB has no diet field).
NONVEG_PATTERN = re.compile(
    r"\b(chicken|mutton|lamb|fish|prawns?|pomfret|mackerel|rohu|ham|bacon|salami|sausages?|meat|keema|mince[d]?|egg|eggs)\b")


def is_nonveg(name: str) -> bool:
    return bool(NONVEG_PATTERN.search(name))
