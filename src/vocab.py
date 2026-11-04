"""Ingredient categories for AaharAI.

Edit the sets HERE (not in a notebook) so changes survive Colab resets.

detectable : OWLv2 is asked to find these in photos; also used by the recommender
manual_add : real ingredients the user adds by hand (dals, flours, meats...); used by the recommender
staple     : assumed to be in the pantry (UI toggle); excluded from coverage scoring
ignore     : colours, essences, additives; excluded from coverage scoring
other      : not classified yet; counts AGAINST coverage, so don't dump things in 'ignore' to look good

Priority when a name is in several sets: staple > detectable > manual_add > ignore.
"""
import re

# canonical-name fixes (applied after cleaning the INDB food name)
RENAME = {
    "lemons": "lemon",
    "siva's": "amchur",
    "white channa": "chickpeas",
    "tomatoes": "tomato",
    "beet root": "beetroot",
    "pear": "pears",
}

STAPLE = {
    "ajwain seed whole organic spices", "amchur", "asafoetida", "baking powder", "balsamic vinegar",
    "basil", "bay leaf", "bicarbonate of soda", "cajun seasoning", "cardamom", "chilli powder",
    "cinnamon", "citric acid", "cloves", "coriander seeds", "cumin seeds", "curry powder",
    "dry red chilli", "fennel seeds", "fenugreek seeds", "garam masala", "garlic powder", "ghee",
    "gingelly seeds", "himalayan black salt", "mace", "mixed curry spices", "mixed herbs",
    "mustard powder", "mustard seeds", "nutmeg", "oil", "oregano", "paprika", "pepper", "poppy seeds",
    "rice vinegar", "rosemary", "saffron", "salt", "sesame seeds", "spices", "star anise", "sugar",
    "tamarind", "tartaric acid", "tea", "thyme", "turmeric powder", "vinegar", "water",
}

DETECT = {
    # base names (as in INDB food_name)
    "apple", "banana", "bottle gourd", "bread", "brinjal - all varieties", "butter", "button mushroom",
    "cabbage", "capsicum", "carrot", "cauliflower", "celery stalk", "cheese", "chicken", "coconut",
    "coriander leaves", "cucumber", "curry leaves", "egg", "fenugreek leaves", "french beans",
    "garlic", "ginger", "green chilli", "lemon", "lettuce", "mango", "milk", "mint leaves", "onion",
    "orange", "paneer", "parsley", "peas", "pineapple", "potato", "pumpkin", "spinach",
    "spring onions", "tomato", "yogurt",
    # names produced by indb.add_canonical() from the original ingredient text
    "cheddar cheese", "fresh coconut", "fresh ginger", "processed cheese", "white bread", "whole egg",
    "whole lemon", "whole milk",
}

MANUAL = {
    # --- from the starter bundle ---
    "agar", "almond", "almond milk", "amaranth leaves", "amaranth seed", "amla", "apricot",
    "bacon rashers", "bajra", "bamboo shoot", "bananas", "bathua leaves", "beans", "beansprouts",
    "beetroot", "bengal gram", "bitter gourd", "black beans", "black gram", "blueberries",
    "breakfast cereal", "broccoli", "capers", "cashew nut", "chickpeas", "chilli sauce", "chocolate",
    "cocoa powder", "coconut milk", "coffee", "colocasia", "corn", "couscous", "cracked wheat",
    "cranberries", "cream", "curly kale", "currants", "custard powder", "dates", "dried mango",
    "drinking chocolate", "drumstick", "flour", "fruit cocktail", "gherkins", "ginger juice", "grapes",
    "green gram", "ground nut", "guava", "gulkand", "ham", "hazelnuts", "honey", "jack fruit",
    "jaggery", "jam", "jowar", "khoa", "kiwi fruit", "ladies finger", "lamb", "lemon grass",
    "lemon peel", "lentil dal", "lime", "litchi", "lotus seeds", "lotus stem", "mackerel", "maize",
    "margarine", "mayonnaise", "melon seeds", "mint sauce", "mixed nuts cashews", "mustard leaves",
    "mustard sauce", "mutton", "olives", "orange peel", "papaya", "pasta", "peach", "peanut butter",
    "peanuts", "pears", "peppers", "pickle", "pine nuts", "pine seed", "pistachio nuts", "pizza base",
    "plum", "pomegranate", "pomfret", "porridge oats", "prawns", "pumpkin seeds", "radish",
    "radish leaves", "ragi", "raisins", "rajmah", "raspberries", "red gram", "rice", "rice flakes",
    "rice puffed", "rohu", "rose petals", "rose water", "sago", "salami", "sausages", "seeds",
    "shallots", "soy sauce", "soya bean", "strawberry jam", "sultanas", "sunflower seeds",
    "sweet potato", "sweetcorn", "tiger prawns", "tinda", "tofu", "tomato ketchup", "tomato puree",
    "tomato sauce", "turnip", "walnut", "watercress", "wheat", "wheat flour", "white sauce",
    "worcestershire sauce", "yam", "yeast extract",
    # --- from the earlier list / indb.add_canonical() names ---
    "almond flour", "beet root", "black rice", "breadcrumbs", "brown bread", "buckwheat", "buttermilk",
    "chia seeds", "chicken bones", "chickpeas (garbanzo beans", "condensed milk", "cornflakes",
    "cornflour", "cowpea", "dried coconut", "egg white", "egg yolk", "fish fillets", "fresh cream",
    "ginger garlic paste", "gooseberries", "gram flour", "granola", "ground ginger", "horse gram",
    "jackfruit", "lemon juice", "lemon rind", "lentil whole", "lotus root", "moth bean", "oatmeal",
    "quinoa", "redcurrants", "refined flour", "rice flour", "roasted chickpeas", "semolina",
    "star fruit", "strawberry", "vermicelli", "whipping cream", "whole wheat flour", "wood apple",
}

IGNORE = {
    # additives, colours, flavourings, drinks, misc
    "MSG monosodium glutamate", "almond essence", "almonds essence", "alum solution",
    "ammonium bicarbonate", "banana essence", "banana flavour", "candies", "cherries", "cola",
    "colour", "dressing", "energy drink", "essence", "food colour", "fruit essence",
    "fruit juice drink/squash", "gelatine", "green colour", "kewra", "kewra essence", "lemon color",
    "lemon colour", "lemon essence", "lemon jelly crystals", "mango essence",
    "msg monosodium glutamate", "orange and green color", "orange colour", "orange drink",
    "orange essence", "orange essesnce", "orange flavoured jelly crystals", "orange juice",
    "pineapple essence", "pineapple essesnce", "pineapple juice", "red colour", "reetha",
    "silver foil", "spirits", "stock cubes", "strawberry colour", "strawberry essence",
    "strawberry flavour", "strawberry jam", "syrawberry flavour", "syrup", "tomato colour",
    "tutti frutti gumballs", "vanilla extract", "whey", "yellow color", "yellow colour",
}
# anything with colour / essence / flavour in its name is ignored too (catches variants not listed above)
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