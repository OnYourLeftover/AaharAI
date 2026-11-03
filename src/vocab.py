RENAME = {"lemons": "lemon", "siva's": "amchur"}

STAPLE = {"salt","water","oil","sugar","ghee","chilli powder","pepper","garam masala","cumin seeds",
 "turmeric powder","coriander seeds","cardamom","cloves","cinnamon","bay leaf","mustard seeds","asafoetida",
 "baking powder","bicarbonate of soda","mustard powder","nutmeg","fennel seeds","ajwain seed whole organic spices",
 "himalayan black salt","amchur","fenugreek seeds","dry red chilli","tamarind","vinegar","oregano",
 "poppy seeds","gingelly seeds","citric acid","saffron","mace"}

DETECT = {"onion","tomato","potato","carrot","capsicum","cauliflower","cabbage","peas","spinach","cucumber",
 "lemon","green chilli","ginger","garlic","coriander leaves","mint leaves","curry leaves","french beans",
 "button mushroom","brinjal - all varieties","bottle gourd","pumpkin","spring onions","lettuce","celery stalk",
 "fenugreek leaves","parsley","banana","apple","mango","orange","pineapple","coconut","egg","paneer","bread",
 "chicken","milk","yogurt","butter","cheese"}

MANUAL = {"wheat flour","flour","wheat","rice","rice flakes","black gram","bengal gram","green gram","red gram",
 "lentil dal","cream","khoa","jaggery","pasta","mutton","lamb","pomfret","beansprouts","almond","ground nut",
 "cashew nut","pistachio nuts","walnut","raisins","dates","mixed nuts cashews","soy sauce","chilli sauce",
 "tomato sauce","tomato ketchup","mayonnaise","honey","jam","coffee","chocolate","cocoa powder"}

IGNORE = {"vanilla extract","kewra essence","gelatine","stock cubes","whey","cherries",
          "orange juice","pineapple juice"}

def category(name):
    if name in STAPLE: return "staple"
    if name in DETECT: return "detectable"
    if name in MANUAL: return "manual_add"
    if name in IGNORE: return "ignore"
    return "other"