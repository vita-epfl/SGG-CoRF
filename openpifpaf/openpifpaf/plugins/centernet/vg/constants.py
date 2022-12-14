OBJ_CATEGORIES = ["airplane",
"animal",
"arm",
"bag",
"banana",
"basket",
"beach",
"bear",
"bed",
"bench",
"bike",
"bird",
"board",
"boat",
"book",
"boot",
"bottle",
"bowl",
"box",
"boy",
"branch",
"building",
"bus",
"cabinet",
"cap",
"car",
"cat",
"chair",
"child",
"clock",
"coat",
"counter",
"cow",
"cup",
"curtain",
"desk",
"dog",
"door",
"drawer",
"ear",
"elephant",
"engine",
"eye",
"face",
"fence",
"finger",
"flag",
"flower",
"food",
"fork",
"fruit",
"giraffe",
"girl",
"glass",
"glove",
"guy",
"hair",
"hand",
"handle",
"hat",
"head",
"helmet",
"hill",
"horse",
"house",
"jacket",
"jean",
"kid",
"kite",
"lady",
"lamp",
"laptop",
"leaf",
"leg",
"letter",
"light",
"logo",
"man",
"men",
"motorcycle",
"mountain",
"mouth",
"neck",
"nose",
"number",
"orange",
"pant",
"paper",
"paw",
"people",
"person",
"phone",
"pillow",
"pizza",
"plane",
"plant",
"plate",
"player",
"pole",
"post",
"pot",
"racket",
"railing",
"rock",
"roof",
"room",
"screen",
"seat",
"sheep",
"shelf",
"shirt",
"shoe",
"short",
"sidewalk",
"sign",
"sink",
"skateboard",
"ski",
"skier",
"sneaker",
"snow",
"sock",
"stand",
"street",
"surfboard",
"table",
"tail",
"tie",
"tile",
"tire",
"toilet",
"towel",
"tower",
"track",
"train",
"tree",
"truck",
"trunk",
"umbrella",
"vase",
"vegetable",
"vehicle",
"wave",
"wheel",
"window",
"windshield",
"wing",
"wire",
"woman",
"zebra"]

REL_CATEGORIES = ["above",
"across",
"against",
"along",
"and",
"at",
"attached to",
"behind",
"belonging to",
"between",
"carrying",
"covered in",
"covering",
"eating",
"flying in",
"for",
"from",
"growing on",
"hanging from",
"has",
"holding",
"in",
"in front of",
"laying on",
"looking at",
"lying on",
"made of",
"mounted on",
"near",
"of",
"on",
"on back of",
"over",
"painted on",
"parked on",
"part of",
"playing",
"riding",
"says",
"sitting on",
"standing on",
"to",
"under",
"using",
"walking in",
"walking on",
"watching",
"wearing",
"wears",
"with"]

REL_CATEGORIES_FLIP = {
"above": "above",
"across": "across",
"against": "against",
"along": "along",
"and": "and",
"at": "at",
"attached to": "attached to",
"behind": "behind",
"belonging to": "belonging to",
"between": "between",
"carrying": "carrying",
"covered in": "covered in",
"covering": "covering",
"eating": "eating",
"flying in": "flying in",
"for": "for",
"from": "from",
"growing on": "growing on",
"hanging from": "hanging from",
"has": "has",
"holding": "holding",
"in": "in",
"in front of": "in front of",
"laying on": "laying on",
"looking at": "looking at",
"lying on": "lying on",
"made of": "made of",
"mounted on": "mounted on",
"near": "near",
"of": "of",
"on": "on",
"on back of": "on back of",
"over": "over",
"painted on": "painted on",
"parked on": "parked on",
"part of": "part of",
"playing": "playing",
"riding": "riding",
"says": "says",
"sitting on": "sitting on",
"standing on": "standing on",
"to": "to",
"under": "under",
"using": "using",
"walking in": "walking in",
"walking on": "walking on",
"watching": "watching",
"wearing": "wearing",
"wears": "wears",
"with": "with"
}

BBOX_KEYPOINTS = [
    'top_left',            # 1
    'top_right',        # 2
    'bottom_right',       # 3
    'bottom_left',        # 4
    'center',       # 5
]

BBOX_HFLIP = {
    'top_left': 'top_right',
    'top_right': 'top_left',
    'bottom_right': 'bottom_left',
    'bottom_left': 'bottom_right',
    'center': 'center',
}
