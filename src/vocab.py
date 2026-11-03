RENAME = {"lemons": "lemon", "siva's": "amchur"}

STAPLE = {"salt","water","oil","sugar","ghee","chilli powder","pepper","garam masala","cumin seeds",
 "turmeric powder","coriander seeds","cardamom","cloves","cinnamon","bay leaf","mustard seeds","asafoetida",
 "baking powder","bicarbonate of soda","mustard powder","nutmeg","fennel seeds","ajwain seed whole organic spices",
 "himalayan black salt","amchur","fenugreek seeds","dry red chilli","tamarind","vinegar","oregano",
 "poppy seeds","gingelly seeds","citric acid","saffron","tea",
 "sesame seeds","garlic powder","thyme","basil","star anise","paprika","curry powder",
 "mixed curry spices","balsamic vinegar","rice vinegar","tartaric acid","mace"}

DETECT = {"onion","tomato","potato","carrot","capsicum","cauliflower","cabbage","peas","spinach","cucumber",
 "lemon","green chilli","ginger","garlic","coriander leaves","mint leaves","curry leaves","french beans",
 "button mushroom","brinjal - all varieties","bottle gourd","pumpkin","spring onions","lettuce","celery stalk",
 "fenugreek leaves","parsley","banana","apple","mango","orange","pineapple","coconut","egg","paneer","bread",
 "chicken","milk","yogurt","butter","cheese"}

MANUAL = {"wheat flour","flour","wheat","rice","rice flakes","black gram","bengal gram","green gram","red gram",
 "lentil dal","cream","khoa","jaggery","pasta","mutton","lamb","pomfret","beansprouts","almond","ground nut",
 "cashew nut","pistachio nuts","walnut","raisins","dates","mixed nuts cashews","soy sauce","chilli sauce",
 "tomato sauce","tomato ketchup","mayonnaise","honey","jam","coffee","chocolate","cocoa powder","breadcrumbs","brown bread","white bread","cheddar cheese",
 "processed cheese","broccoli", "tofu", "rajmah", "strawberry jam",
 # Vegetables and fruits
 "broccoli", "turnip", "grapes", "radish", "shallots",
 "guava", "jackfruit", "ladies finger", "bitter gourd",
 "colocasia", "corn", "curly kale", "lime", "lotus root",
 "papaya", "peach", "plum", "pomegranate", "sweet potato",
 "sweetcorn", "amla", "amaranth leaves", "drumstick",
 "gooseberries", "kiwi fruit", "litchi", "mustard leaves",
 "tinda", "watercress", "wood apple", "bamboo shoot",
 # Pulses, grains and seeds
 "rajmah", "soya bean", "cracked wheat", "bajra","ragi", "jowar", "maize", "black rice", "quinoa",
 "buckwheat", "cowpea", "horse gram", "moth bean",
 "lentil whole", "roasted chickpeas", "chickpeas",
 "rice puffed", "sago", "lotus seeds", "pumpkin seeds",
 "sesame seeds", "sunflower seeds", "chia seeds",
 "melon seeds", "pine nuts",

 # Other ingredients
 "tofu", "coconut milk", "tomato puree", "garlic powder",
 "custard powder", "curry powder", "mixed curry spices",
 "worcestershire sauce", "mint sauce", "rose water",
 "rose petals", "almond flour", "almond milk",
 "peanut butter", "dried mango", "fruit cocktail",
 "breakfast cereal", "cornflakes", "granola",
 "drinking chocolate", "couscous", "oatmeal",
 "pizza base", "white sauce", "mustard sauce",
 "mackerel", "tiger prawns", "rohu", "prawns",
 "ham", "bacon rashers", "salami", "sausages",
 "mutton", "fish fillets", "chicken bones"
 "chicken bones",
 "dried coconut","fresh coconut","whipping cream",
 "fresh cream",
 "whole egg",
 "egg yolk",
 "egg white",
 "cornflour",
 "gram flour",
 "rice flour",
 "ground ginger",
 "ginger garlic paste",
 "fresh ginger",
 "lemon juice",
 "lemon rind",
 "whole lemon",
 "condensed milk",
 "buttermilk",
 "whole milk",
 "vermicelli",
 "semolina","whole wheat flour",
    "refined flour"
}

IGNORE = {"vanilla extract","kewra essence","red colour", "orange colour", 
          "green colour", "red colour" , "lemon essence"
          "yellow colour", "yellow color", "tomato colour", 
          "lemon color", "lemon colour", "orange and green color", 
          "colour", "food colour", "strawberry colour", "essence", 
          "lemon essence", "orange essence", "banana essence", 
          "mango essence", "pineapple essence", "almond essence", 
          "almonds essence", "strawberry essence", "fruit essence", 
          "orange essesnce", "pineapple essesnce", "strawberry flavour", 
          "banana flavour", "MSG monosodium glutamate", 
          "ammonium bicarbonate", "alum solution", "silver foil", 
          "lemon jelly crystals", "orange flavoured jelly crystals", 
          "strawberry jam""gelatine","stock cubes","whey","cherries",
          "orange juice","pineapple juice"}

def category(name):
    if name in STAPLE: return "staple"
    if name in DETECT: return "detectable"
    if name in MANUAL: return "manual_add"
    if name in IGNORE: return "ignore"
    return "other"