from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline
import io
import base64

app = Flask(__name__)
CORS(app)

# Global variables for models
mobilenet_model = None
gpt2_tokenizer = None
gpt2_model = None
stable_diffusion_pipe = None

def load_models():
    """Load all AI models once at startup"""
    global mobilenet_model, gpt2_tokenizer, gpt2_model, stable_diffusion_pipe
    
    print("Loading MobileNetV2...")
    mobilenet_model = MobileNetV2(weights='imagenet')
    
    print("Loading GPT-2...")
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    gpt2_model = GPT2LMHeadModel.from_pretrained('distilgpt2')
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
    
    print("Loading Stable Diffusion (this may take a while)...")
    try:
        stable_diffusion_pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None  # Disable safety checker for faster generation
        )
        if torch.cuda.is_available():
            stable_diffusion_pipe = stable_diffusion_pipe.to("cuda")
            print("Using GPU for image generation!")
        else:
            print("Using CPU for image generation (this will be slower)")
        
        # Optimize for faster generation
        stable_diffusion_pipe.enable_attention_slicing()
        
    except Exception as e:
        print(f"Warning: Could not load Stable Diffusion: {e}")
        print("Image generation will not be available")
        stable_diffusion_pipe = None
    
    print("Models loaded successfully!")

def preprocess_image(image_data):
    """Preprocess image for MobileNetV2"""
    image_bytes = base64.b64decode(image_data.split(',')[1])
    image = Image.open(io.BytesIO(image_bytes))
    
    img_array = np.array(image)
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    resized_img = cv2.resize(img_array, (224, 224))
    if len(resized_img.shape) == 3:
        resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    
    img_batch = np.expand_dims(resized_img, axis=0)
    preprocessed = preprocess_input(img_batch)
    
    return preprocessed

def classify_material(preprocessed_image):
    """Classify material using MobileNetV2"""
    predictions = mobilenet_model.predict(preprocessed_image, verbose=0)
    decoded_predictions = decode_predictions(predictions, top=5)[0]
    
    material_mapping = {
        'water_bottle': 'plastic bottle', 'pop_bottle': 'plastic bottle',
        'wine_bottle': 'glass', 'beer_bottle': 'glass', 'jar': 'glass',
        'can_opener': 'metal', 'beer_can': 'metal', 'tin_can': 'metal',
        'envelope': 'paper', 'book_jacket': 'paper', 'newspaper': 'paper',
        'carton': 'cardboard', 'cardboard_box': 'cardboard'
    }
    
    for pred_class, confidence in [(pred[1], pred[2]) for pred in decoded_predictions]:
        for keyword, material in material_mapping.items():
            if keyword in pred_class.lower():
                return material, confidence
    
    return 'plastic bottle', decoded_predictions[0][2]

def get_curated_diy_ideas(material_type):
    """
    Return 5 high-quality, practical DIY ideas with detailed information
    These are curated, meaningful ideas that users can actually implement
    """
    diy_database = {
        'plastic bottle': [
            {
                'id': 1,
                'title': 'Self-Watering Planter System',
                'description': 'Create an efficient self-watering planter perfect for herbs and small plants',
                'difficulty': 'Easy',
                'time': '15 minutes',
                'materials': ['Plastic bottle', 'Scissors/knife', 'Cotton string or shoelace', 'Soil', 'Small plant'],
                'steps': [
                    'Cut the plastic bottle in half horizontally, creating top and bottom sections',
                    'Remove the bottle cap and make a small hole in it',
                    'Thread a cotton string or shoelace through the cap hole, leaving 6 inches hanging down',
                    'Place the top half upside down into the bottom half (cap pointing down into the water reservoir)',
                    'Fill the bottom section with water',
                    'Fill the top section (now inverted) with soil, ensuring the string touches the soil',
                    'Plant your herb or small plant in the soil',
                    'The string will wick water from the reservoir to keep soil moist',
                    'Refill water reservoir weekly or when low'
                ],
                'tips': ['Use organic cotton string for better water absorption', 'Best for herbs like basil, mint, or small succulents', 'Keep in a sunny spot for best results']
            },
            {
                'id': 2,
                'title': 'Bird Feeder with Perch',
                'description': 'Attract beautiful birds to your garden with this eco-friendly feeder',
                'difficulty': 'Easy',
                'time': '20 minutes',
                'materials': ['Plastic bottle', 'Two wooden spoons', 'Scissors/knife', 'Birdseed', 'String or wire'],
                'steps': [
                    'Clean the plastic bottle thoroughly and remove the label',
                    'Mark two holes on opposite sides of the bottle, about 3 inches from the bottom',
                    'Cut small circular holes (slightly larger than your wooden spoon handle)',
                    'Insert wooden spoon through both holes - handle in one side, spoon head out the other',
                    'Cut a small seed dispensing hole just above the spoon head',
                    'Repeat steps 2-5 about 3 inches above the first spoon for a second perch',
                    'Fill the bottle with birdseed through the top opening',
                    'Create a hanger by punching two holes in the cap and threading wire through',
                    'Hang from a tree branch or hook in your garden',
                    'Refill seed as needed and clean monthly'
                ],
                'tips': ['Use different seed types to attract various bird species', 'Position near trees for bird safety', 'Clean regularly to prevent mold']
            },
            {
                'id': 3,
                'title': 'Desktop Organizer Tower',
                'description': 'Create a multi-level storage system for pens, scissors, and office supplies',
                'difficulty': 'Medium',
                'time': '30 minutes',
                'materials': ['3-4 plastic bottles', 'Scissors', 'Hot glue gun', 'Decorative paper/paint', 'Cardboard base'],
                'steps': [
                    'Cut bottles at different heights: first at 4 inches, second at 6 inches, third at 3 inches',
                    'Smooth any sharp edges with sandpaper or tape',
                    'Decorate each section with paint, washi tape, or decorative paper',
                    'Cut a cardboard circle (8-10 inches diameter) as the base',
                    'Cover the cardboard base with decorative paper or paint',
                    'Arrange the bottle pieces on the base in a visually pleasing pattern',
                    'Use hot glue to securely attach each bottle section to the base',
                    'Allow glue to dry completely (15-20 minutes)',
                    'Fill with pens, pencils, scissors, rulers, and other supplies',
                    'Optional: add labels to each section'
                ],
                'tips': ['Use bottles of different colors for visual appeal', 'Group similar items in each section', 'Make extra compartments for paper clips and push pins']
            },
            {
                'id': 4,
                'title': 'Hanging Vertical Garden',
                'description': 'Maximize space by creating a vertical garden for herbs or flowers',
                'difficulty': 'Medium',
                'time': '45 minutes',
                'materials': ['5-6 plastic bottles', 'Rope or strong string', 'Scissors/knife', 'Soil', 'Small plants', 'Drill or hot nail'],
                'steps': [
                    'Cut a large rectangular window (3x4 inches) on one side of each bottle',
                    'Make two small drainage holes at the bottom opposite side',
                    'Create four holes for rope: two at top, two at bottom of each bottle',
                    'Thread rope through top holes of the first bottle and tie knots to secure',
                    'Add soil to the bottle through the cut window, filling 3/4 full',
                    'Plant your herb or flower, ensuring roots are covered',
                    'Connect the next bottle by threading rope through its top holes',
                    'Leave 6-8 inches of rope between bottles for spacing',
                    'Repeat until all bottles are connected in a vertical chain',
                    'Hang the top of the rope on a sturdy hook or fence',
                    'Water carefully from top, allowing excess to drain through holes'
                ],
                'tips': ['Start with hardy plants like succulents or herbs', 'Ensure rope is UV-resistant if outdoors', 'Water less frequently than regular pots']
            },
            {
                'id': 5,
                'title': 'Piggy Bank with Coin Counter',
                'description': 'Fun saving tool for kids with a visible coin tracking system',
                'difficulty': 'Easy',
                'time': '25 minutes',
                'materials': ['Clear plastic bottle', 'Scissors/knife', 'Markers or paint', 'Ruler', 'Stickers for decoration'],
                'steps': [
                    'Choose a clear bottle so you can see the coins accumulating',
                    'Cut a coin slot (about 2 inches long, 1/4 inch wide) near the top',
                    'Smooth the edges with tape to prevent cuts',
                    'Use a ruler and permanent marker to create measurement lines up the side',
                    'Mark every inch with a line and number (like a thermometer)',
                    'Decorate the bottle with stickers, paint, or markers',
                    'Create a goal tracker: "Goal: $50" or "Saving for: Bicycle"',
                    'Optional: Cover the bottom half with colored paper so you can\'t see exact amounts',
                    'Start saving! Drop coins through the slot',
                    'Track progress by watching the coin level rise',
                    'When full, cut the bottom to retrieve coins or break the bottle'
                ],
                'tips': ['Set specific savings goals to stay motivated', 'Save only one coin type to make counting easier', 'Create a weekly deposit schedule']
            }
        ],
        'glass': [
            {
                'id': 1,
                'title': 'Elegant Candle Holders',
                'description': 'Transform glass jars into beautiful ambient lighting',
                'difficulty': 'Easy',
                'time': '30 minutes',
                'materials': ['Glass jar/bottle', 'Candles or LED tea lights', 'Decorative stones/sand', 'Ribbon', 'Paint (optional)'],
                'steps': [
                    'Clean the glass jar thoroughly, removing all labels and residue',
                    'Soak in warm soapy water for 10 minutes to remove sticky label glue',
                    'Dry completely with a clean cloth',
                    'Optional: Paint the outside with glass paint in your desired color or pattern',
                    'If painting, let dry for 2-3 hours according to paint instructions',
                    'Fill the bottom with 1-2 inches of decorative stones, pebbles, or sand',
                    'This provides stability and adds visual interest',
                    'Place a candle or LED tea light in the center on top of the stones',
                    'Wrap decorative ribbon around the neck of the jar and tie a bow',
                    'Optional: Add dried flowers, twine, or lace for rustic charm',
                    'Light the candle and enjoy the warm, ambient glow',
                    'For outdoor use, the jar protects the flame from wind'
                ],
                'tips': ['Use multiple jars of different heights for visual impact', 'LED tea lights are safer for children and pets', 'Add essential oils to the sand for aromatherapy']
            },
            {
                'id': 2,
                'title': 'Miniature Terrarium Garden',
                'description': 'Create a low-maintenance indoor garden ecosystem',
                'difficulty': 'Medium',
                'time': '40 minutes',
                'materials': ['Wide-mouth glass jar', 'Small pebbles', 'Activated charcoal', 'Potting soil', 'Small plants (succulents/moss)', 'Decorative items'],
                'steps': [
                    'Choose a clean, clear glass jar with a wide opening for easy access',
                    'Add a 1-inch layer of small pebbles for drainage at the bottom',
                    'Add a thin layer (1/2 inch) of activated charcoal to prevent odors and mold',
                    'Add 2-3 inches of potting soil suitable for your chosen plants',
                    'Use a spoon to create small holes for planting',
                    'Carefully plant 2-3 small succulents or add patches of moss',
                    'Gently pat soil around plant roots to secure them',
                    'Add decorative elements: small rocks, miniature figurines, or driftwood',
                    'Lightly water the soil (less is more for terrariums)',
                    'Place in indirect sunlight - avoid direct hot sun',
                    'For closed terrariums: add the lid and open weekly for air circulation',
                    'Water only when soil feels dry (every 2-3 weeks for succulents)'
                ],
                'tips': ['Choose plants with similar water needs', 'Succulents are easiest for beginners', 'Closed terrariums create their own water cycle']
            },
            {
                'id': 3,
                'title': 'Kitchen Herb Storage Jars',
                'description': 'Organize dried herbs and spices in stylish glass containers',
                'difficulty': 'Easy',
                'time': '20 minutes per jar',
                'materials': ['Glass jars with lids', 'Chalkboard paint/labels', 'Dried herbs', 'Funnel', 'Brush'],
                'steps': [
                    'Sterilize glass jars by boiling in water for 10 minutes',
                    'Let jars dry completely - moisture can spoil herbs',
                    'If using chalkboard paint: apply two coats to the lid or jar surface',
                    'Allow paint to dry between coats (30 minutes each)',
                    'Once dry, write herb names with chalk or chalk marker',
                    'Use a funnel to transfer dried herbs into jars to avoid spills',
                    'Fill jars only 3/4 full to allow for easy access',
                    'Ensure lids are tightly sealed to maintain freshness',
                    'Label each jar clearly with herb name and date',
                    'Store in a cool, dark place away from direct sunlight',
                    'Arrange on a spice rack or in a drawer for easy access'
                ],
                'tips': ['Use smaller jars for expensive spices like saffron', 'Group by cuisine type (Italian, Indian, Mexican)', 'Replace dried herbs every 6-12 months for best flavor']
            },
            {
                'id': 4,
                'title': 'Decorative Vase with Rope Accent',
                'description': 'Create a rustic, textured vase perfect for any room',
                'difficulty': 'Easy',
                'time': '35 minutes',
                'materials': ['Glass bottle', 'Jute rope or twine', 'Hot glue gun', 'Scissors', 'Flowers'],
                'steps': [
                    'Clean and dry the glass bottle completely',
                    'Start at the bottom: apply a line of hot glue around the base',
                    'Quickly press the end of your rope into the glue',
                    'Apply glue in small sections (2-3 inches at a time) to keep it from drying',
                    'Wrap the rope tightly around the bottle, pushing each layer against the previous',
                    'Keep rows straight and tightly packed with no gaps',
                    'Continue wrapping and gluing until you reach the neck of the bottle',
                    'For the neck, you can leave it clear or continue wrapping',
                    'Cut the rope and secure the end with extra glue',
                    'Let dry completely for 15-20 minutes',
                    'Optional: add buttons, lace, or shells with hot glue for decoration',
                    'Fill with water and add your favorite flowers'
                ],
                'tips': ['Use different rope thicknesses for varied texture', 'Combine with painted sections for contrast', 'Perfect for weddings and rustic decor']
            },
            {
                'id': 5,
                'title': 'Bathroom Storage Jars',
                'description': 'Organize cotton balls, Q-tips, and bathroom essentials stylishly',
                'difficulty': 'Easy',
                'time': '15 minutes',
                'materials': ['Various sized glass jars', 'Paint/labels', 'Cotton balls/swabs', 'Ribbon'],
                'steps': [
                    'Collect glass jars of various sizes for different items',
                    'Remove all labels by soaking in hot soapy water',
                    'Scrub away label residue with a sponge or scraper',
                    'Dry jars thoroughly inside and out',
                    'Optional: Paint lids in coordinating colors',
                    'Create labels using printable sticker paper or handwrite tags',
                    'Suggested labels: "Cotton Balls", "Q-Tips", "Bath Salts", "Hair Ties"',
                    'Fill each jar with its designated items',
                    'Arrange jars on bathroom counter or shelf',
                    'Keep frequently used items in the front',
                    'Larger jars can hold loofahs or rolled washcloths',
                    'Add a ribbon around the neck for decorative touch'
                ],
                'tips': ['Use uniform jar sizes for a cohesive look', 'Choose jars with wide mouths for easy refilling', 'Keep lids on to prevent dust accumulation']
            }
        ],
        'metal': [
            {
                'id': 1,
                'title': 'Desk Organizer System',
                'description': 'Create a functional multi-compartment office organizer',
                'difficulty': 'Easy',
                'time': '25 minutes',
                'materials': ['Tin cans (various sizes)', 'Sandpaper', 'Spray paint', 'Strong adhesive/glue', 'Wooden board'],
                'steps': [
                    'Remove labels from all tin cans and clean thoroughly',
                    'Check for sharp edges around the rim - use sandpaper to smooth',
                    'Wash cans with soap and water, then dry completely',
                    'Apply primer if painting (helps paint stick to metal)',
                    'Spray paint cans in your chosen colors - apply 2-3 thin coats',
                    'Allow 30 minutes drying time between coats',
                    'While cans dry, cut or purchase a wooden base (12x8 inches works well)',
                    'Paint or stain the wooden base if desired',
                    'Arrange cans on the base in your preferred layout',
                    'Mark positions with pencil before gluing',
                    'Apply strong adhesive to the bottom rim of each can',
                    'Press firmly onto base and let dry overnight for maximum strength',
                    'Fill with pens, pencils, scissors, rulers, and supplies'
                ],
                'tips': ['Mix can heights for visual interest', 'Use magnetic cans to stick on metal surfaces', 'Add labels with chalkboard paint']
            },
            {
                'id': 2,
                'title': 'Succulent Planter with Drainage',
                'description': 'Perfect mini planters for desk or windowsill gardens',
                'difficulty': 'Easy',
                'time': '20 minutes',
                'materials': ['Small tin cans', 'Hammer and nail', 'Spray paint', 'Small rocks', 'Cactus soil', 'Succulents'],
                'steps': [
                    'Clean the can and remove any labels completely',
                    'Flip can upside down and place on a thick towel or wood block',
                    'Use hammer and nail to punch 3-5 drainage holes in the bottom',
                    'Holes should be evenly spaced for proper water drainage',
                    'Smooth any sharp points around holes with sandpaper',
                    'Paint the outside with rust-resistant spray paint (2-3 coats)',
                    'Let dry completely - 2-3 hours or overnight',
                    'Add a layer of small pebbles (1/2 inch) at the bottom for drainage',
                    'Fill with cactus/succulent soil mix to about 1 inch from top',
                    'Remove succulent from its nursery pot and gently loosen roots',
                    'Plant succulent in the center, adding more soil around it',
                    'Press soil gently to secure plant',
                    'Wait 2-3 days before first watering to let roots settle',
                    'Water sparingly - only when soil is completely dry'
                ],
                'tips': ['Group multiple planters for a mini garden', 'Succulents need bright indirect light', 'Water less in winter months']
            },
            {
                'id': 3,
                'title': 'Musical Wind Chime',
                'description': 'Create soothing sounds with recycled metal pieces',
                'difficulty': 'Medium',
                'time': '45 minutes',
                'materials': ['Multiple metal cans/lids', 'Fishing line or string', 'Wooden dowel/branch', 'Drill', 'Beads', 'Metal washer'],
                'steps': [
                    'Collect 5-7 metal cans or lids of different sizes for varied tones',
                    'Clean all metal pieces thoroughly',
                    'Use a drill to make one hole in the center of each piece',
                    'Hole should be large enough for your string to pass through twice',
                    'Cut the wooden dowel or branch to about 12 inches length',
                    'Mark 5-7 points along the dowel for hanging positions',
                    'Drill small holes at each marked point',
                    'Cut fishing line into varying lengths: 8", 10", 12", 14", 16"',
                    'Thread each piece of fishing line through a hole in the metal',
                    'Tie a knot and add a bead below the metal to prevent slipping',
                    'Thread the top of the line through a hole in the dowel',
                    'Tie securely, ensuring metals hang at different heights',
                    'Attach a larger piece of string to both ends of dowel for hanging',
                    'Hang outside where wind will move the pieces',
                    'Adjust lengths so pieces gently tap each other'
                ],
                'tips': ['Different metals create different tones - experiment!', 'Add beads between pieces for visual appeal', 'Seal metal with clear coat to prevent rust']
            },
            {
                'id': 4,
                'title': 'Kitchen Utensil Holder',
                'description': 'Organize cooking tools in a decorative countertop holder',
                'difficulty': 'Easy',
                'time': '30 minutes',
                'materials': ['Large tin can (coffee can size)', 'Decorative paper/fabric', 'Mod Podge or glue', 'Ribbon', 'Brush', 'Scissors'],
                'steps': [
                    'Wash and thoroughly dry a large tin can',
                    'Remove any labels and adhesive residue',
                    'Measure the height and circumference of the can',
                    'Cut decorative paper or fabric to fit, adding 1 inch extra to circumference',
                    'Apply a thin layer of Mod Podge to the outside of the can',
                    'Carefully wrap the paper/fabric around the can, smoothing out bubbles',
                    'Overlap the edges slightly and trim excess at top and bottom',
                    'Apply another layer of Mod Podge over the paper to seal',
                    'This creates a protective, water-resistant coating',
                    'Let dry for 1-2 hours until completely set',
                    'Glue ribbon around the top and bottom edges for a finished look',
                    'Optional: add a layer of clear acrylic sealer for extra protection',
                    'Fill with wooden spoons, spatulas, whisks, and other utensils'
                ],
                'tips': ['Use waterproof decorative paper for kitchen use', 'Add weight to bottom (sand/rice in a bag) for stability', 'Make multiple cans for different utensil types']
            },
            {
                'id': 5,
                'title': 'Wall-Mounted Pencil Holders',
                'description': 'Space-saving storage perfect for small workspaces',
                'difficulty': 'Medium',
                'time': '35 minutes',
                'materials': ['3-4 small cans', 'Wooden board', 'Metal hose clamps', 'Screws', 'Paint', 'Screwdriver', 'Drill'],
                'steps': [
                    'Clean all cans and remove sharp edges with sandpaper',
                    'Paint cans in coordinating colors and let dry completely',
                    'Cut or purchase a wooden board (18 inches long, 4 inches wide)',
                    'Sand the wood smooth and paint or stain as desired',
                    'Let wood dry for 2-3 hours',
                    'Position cans on the board to determine spacing',
                    'For each can, wrap a metal hose clamp around it loosely',
                    'Mark where the clamp screw will attach to the wood',
                    'Drill pilot holes at marked positions to prevent wood splitting',
                    'Place can with clamp on board and tighten screw into pilot hole',
                    'Ensure can is held firmly but not crushing the metal',
                    'Repeat for all cans, spacing them evenly',
                    'Attach mounting hardware to back of board (keyhole hangers or D-rings)',
                    'Mount on wall above desk or workspace',
                    'Fill with pens, markers, paintbrushes, or craft supplies'
                ],
                'tips': ['Angle cans slightly upward so items don\'t fall out', 'Add small labels below each can', 'Use anchors if mounting on drywall']
            }
        ],
        'paper': [
            {
                'id': 1,
                'title': 'Beautiful Origami Flowers',
                'description': 'Create lasting paper flowers for decoration',
                'difficulty': 'Medium',
                'time': '30 minutes per flower',
                'materials': ['Colorful paper', 'Scissors', 'Wire/sticks', 'Glue', 'Green tape'],
                'steps': [
                    'Cut paper into squares (6x6 inches for medium flowers)',
                    'For each flower, you\'ll need 5-6 squares of the same color',
                    'Fold each square diagonally to create a triangle',
                    'Fold triangle in half again to create a smaller triangle',
                    'Cut a petal shape along the curved edge (rounded, not pointed)',
                    'Unfold to reveal an 8-petaled flower shape',
                    'Cut off one petal, leaving 7 petals',
                    'Overlap the cut edges and glue to create a cone shape',
                    'Repeat this process with all 5-6 paper pieces',
                    'Each piece should have a slightly different cone depth',
                    'Stack the cone-shaped petals from largest to smallest',
                    'Glue them together in the center, offsetting petals',
                    'Poke a wire or stick through the center for a stem',
                    'Wrap stem with green floral tape for a realistic look',
                    'Curl petal edges slightly with scissors or pencil for dimension',
                    'Create several flowers and arrange in a vase'
                ],
                'tips': ['Use patterned scrapbook paper for unique designs', 'Practice with plain paper first', 'Add paper leaves to stems']
            },
            {
                'id': 2,
                'title': 'Seed Starter Pots',
                'description': 'Eco-friendly biodegradable pots for gardening',
                'difficulty': 'Easy',
                'time': '15 minutes',
                'materials': ['Newspaper', 'Glass or can (for mold)', 'Scissors', 'Seeds', 'Soil'],
                'steps': [
                    'Cut newspaper into strips about 12 inches long and 4 inches wide',
                    'Fold strip in half lengthwise for double thickness strength',
                    'Wrap the paper strip around a glass or can, leaving 2 inches overhanging at bottom',
                    'The glass acts as a mold to shape your pot',
                    'Fold the overhanging bottom portion inward, like wrapping a gift',
                    'Overlap the folds to create a solid bottom',
                    'Press folds firmly to crease them well',
                    'Slide the paper pot off the glass mold carefully',
                    'The pot should hold its shape when you release it',
                    'Fill the pot with seed-starting soil mix',
                    'Plant 2-3 seeds per pot at recommended depth',
                    'Water gently - the paper will absorb moisture',
                    'Place pots in a seed tray to contain any water drainage',
                    'Once seedlings outgrow the pot, plant the entire pot in soil',
                    'The newspaper will decompose and roots will grow through it'
                ],
                'tips': ['Use black and white newspaper only (colored ink may contain toxins)', 'Make pots slightly larger than needed for root growth', 'Keep soil moist but not waterlogged']
            },
            {
                'id': 3,
                'title': 'Custom Gift Wrapping Paper',
                'description': 'Transform old paper into beautiful personalized wrapping',
                'difficulty': 'Easy',
                'time': '20 minutes',
                'materials': ['Brown paper/newspaper', 'Stamps/stencils', 'Paint', 'Sponges', 'Ribbon'],
                'steps': [
                    'Flatten and smooth your paper - brown kraft paper or newspaper work well',
                    'Lay paper on a protected work surface (use newspaper underneath)',
                    'Choose your design method: stamps, stencils, or freehand',
                    'For stamps: apply paint evenly to stamp and press firmly onto paper',
                    'For stencils: place stencil on paper and dab paint with sponge',
                    'For freehand: use brushes or fingers to create patterns',
                    'Create repeating patterns by spacing designs evenly',
                    'Popular patterns: dots, stars, flowers, geometric shapes',
                    'Use metallic paints or markers for elegant accents',
                    'Let paint dry completely (30-60 minutes) before handling',
                    'Optional: add personal messages or recipient\'s name',
                    'Once dry, use to wrap gifts as you would regular wrapping paper',
                    'Pair with natural twine, ribbon, or dried flowers',
                    'Add a handmade gift tag to complete the personalized look'
                ],
                'tips': ['Use potato stamps for easy custom shapes', 'White paint on brown paper looks elegant', 'Add glitter while paint is wet for sparkle']
            },
            {
                'id': 4,
                'title': 'Paper Mache Bowl',
                'description': 'Create decorative bowls from recycled paper',
                'difficulty': 'Medium',
                'time': '2 hours + drying time',
                'materials': ['Newspaper strips', 'Flour/water paste', 'Bowl (as mold)', 'Plastic wrap', 'Paint', 'Sealant'],
                'steps': [
                    'Tear newspaper into strips about 1 inch wide and 6 inches long',
                    'Tearing (not cutting) creates better edges for layering',
                    'Make paste by mixing 1 cup flour with 1 cup water until smooth',
                    'For stronger paste, heat mixture while stirring until it thickens',
                    'Cover your mold bowl with plastic wrap to prevent sticking',
                    'Flip bowl upside down - you\'ll apply paper to the outside',
                    'Dip newspaper strips into paste, removing excess by sliding between fingers',
                    'Apply strips to outside of bowl, overlapping edges',
                    'Smooth out wrinkles and air bubbles with your fingers',
                    'Complete one full layer covering the entire bowl',
                    'Let first layer dry for 30 minutes before adding next layer',
                    'Apply 4-5 more layers, drying between each',
                    'Let final layer dry completely overnight (12-24 hours)',
                    'Carefully remove paper bowl from mold and peel off plastic wrap',
                    'Sand edges smooth if needed',
                    'Paint inside and outside with acrylic paint (2-3 coats)',
                    'Once paint dries, apply clear sealant for durability',
                    'Use for holding lightweight items like jewelry or keys'
                ],
                'tips': ['Add more layers for stronger bowls', 'Mix colored paper for interesting patterns', 'Apply metallic paint for elegant finish']
            },
            {
                'id': 5,
                'title': 'Decorative Bookmarks',
                'description': 'Create personalized bookmarks for yourself or as gifts',
                'difficulty': 'Easy',
                'time': '15 minutes',
                'materials': ['Cardstock or thick paper', 'Scissors', 'Markers/colored pencils', 'Ribbon', 'Laminator or clear tape', 'Hole punch'],
                'steps': [
                    'Cut cardstock into bookmark rectangles: 2 inches wide by 6-7 inches tall',
                    'Use a ruler to ensure straight, even edges',
                    'Sketch your design lightly in pencil first',
                    'Design ideas: quotes, patterns, characters, nature scenes',
                    'Color your design with markers, colored pencils, or watercolors',
                    'For watercolors, use thicker paper to prevent warping',
                    'Let any wet media dry completely (20-30 minutes)',
                    'Optional: add decorative elements like stickers or washi tape',
                    'Write inspiring quotes or personal messages',
                    'Laminate the bookmark for durability (or cover with clear tape)',
                    'Trim excess lamination, leaving 1/8 inch border',
                    'Punch a hole in the top center of the bookmark',
                    'Cut a 6-inch piece of ribbon and fold in half',
                    'Thread folded end through hole and pull ribbon ends through loop',
                    'Your bookmark is complete and ready to use!'
                ],
                'tips': ['Make sets as gifts for book lovers', 'Use pressed flowers sealed between layers', 'Create themed sets (seasons, holidays)']
            }
        ],
        'cardboard': [
            {
                'id': 1,
                'title': 'Drawer Organizer System',
                'description': 'Custom-fit organizers for any drawer size',
                'difficulty': 'Easy',
                'time': '30 minutes',
                'materials': ['Cardboard boxes', 'Ruler', 'Scissors/box cutter', 'Decorative paper', 'Glue', 'Tape'],
                'steps': [
                    'Measure your drawer dimensions: length, width, and height',
                    'Plan your compartment layout on paper first',
                    'Decide what you\'ll store: socks, office supplies, jewelry, etc.',
                    'Cut cardboard pieces for dividers based on your drawer height',
                    'Cut strips about 1/2 inch shorter than drawer height',
                    'Create vertical dividers running length-wise and width-wise',
                    'Make slots in dividers where they intersect (like a grid)',
                    'Slots should be half the height of each divider',
                    'Interlock dividers by sliding slots together to form a grid',
                    'Test fit in drawer and adjust sizes as needed',
                    'Remove from drawer to decorate (optional but recommended)',
                    'Cover cardboard with decorative paper, fabric, or contact paper',
                    'Use glue or double-sided tape to secure covering',
                    'Let dry completely if using glue',
                    'Place finished organizer in drawer',
                    'Arrange items in each compartment by category'
                ],
                'tips': ['Label compartments for easy organization', 'Use different colored paper for each section', 'Make compartments slightly larger than items']
            },
            {
                'id': 2,
                'title': 'Cat Playhouse with Scratcher',
                'description': 'Multi-level play structure cats will love',
                'difficulty': 'Hard',
                'time': '90 minutes',
                'materials': ['Large cardboard boxes', 'Box cutter', 'Strong tape', 'Sisal rope', 'Hot glue', 'Markers'],
                'steps': [
                    'Gather 2-3 large sturdy boxes (moving boxes work great)',
                    'Design your playhouse: multiple levels, windows, doors, tunnels',
                    'Draw windows and doors on boxes with marker before cutting',
                    'Windows should be 4-6 inches diameter for cat to see through',
                    'Door should be 8-10 inches diameter for easy entry',
                    'Use box cutter to carefully cut out marked shapes',
                    'Save some cutout pieces to use as ramps or decorations',
                    'Stack boxes to create levels - strongest/largest on bottom',
                    'Cut holes in floor/ceiling of boxes to connect levels',
                    'Secure boxes together with strong packing tape',
                    'Reinforce all joints and seams with extra tape',
                    'Wrap cardboard tubes or box corners with sisal rope',
                    'Apply hot glue to secure rope end, then wrap tightly',
                    'Continue wrapping and gluing every few inches',
                    'Create at least one good scratching post section',
                    'Add carpet remnants inside for comfort (optional)',
                    'Decorate outside with non-toxic paint or markers',
                    'Place catnip toys inside to attract your cat'
                ],
                'tips': ['Ensure all edges are smooth and safe', 'Make entrance holes large enough for your cat', 'Place on washable mat to catch litter']
            },
            {
                'id': 3,
                'title': 'Picture Frames',
                'description': 'Create custom frames for photos or art',
                'difficulty': 'Medium',
                'time': '40 minutes',
                'materials': ['Cardboard', 'Ruler', 'Scissors', 'Decorative paper', 'Glue', 'Clear plastic sheet', 'Photo'],
                'steps': [
                    'Decide on frame size: photo size plus 2-inch border all around',
                    'For 4x6 photo, cut cardboard base to 8x10 inches',
                    'Cut a second piece of cardboard the exact same size',
                    'In the second piece, cut a window opening',
                    'Window should be slightly smaller than your photo (3.5x5.5 for 4x6 photo)',
                    'Use a ruler to measure and mark the window precisely',
                    'Cut window carefully with box cutter for clean edges',
                    'Cut decorative paper slightly larger than cardboard',
                    'Wrap decorative paper around the frame front (piece with window)',
                    'Fold edges to back and glue down smoothly',
                    'Cut away paper from window opening, leaving 1/2 inch',
                    'Fold this excess paper to the back and glue',
                    'Optional: tape clear plastic behind window for protection',
                    'Place photo on the base cardboard piece',
                    'Apply glue around the edges of base (not center where photo sits)',
                    'Press frame front onto base, creating a pocket for photo',
                    'Leave top edge open to slide photo in and out',
                    'Create a stand: cut triangle of cardboard and tape to back'
                ],
                'tips': ['Use corrugated cardboard for extra stiffness', 'Add multiple layers for 3D effect', 'Embellish with buttons, ribbons, or beads']
            },
            {
                'id': 4,
                'title': 'Cable Management Box',
                'description': 'Hide messy cables and power strips stylishly',
                'difficulty': 'Medium',
                'time': '35 minutes',
                'materials': ['Shoebox or cardboard box', 'Box cutter', 'Decorative paper/paint', 'Rubber bands', 'Label maker'],
                'steps': [
                    'Choose a box large enough for your power strip and excess cables',
                    'Shoebox size (14x8x5 inches) works for most setups',
                    'Measure and mark holes on both short ends of box',
                    'Holes should align with where cables need to exit',
                    'Cut circular holes slightly larger than your cable plugs',
                    'Use a box cutter or scissors to cut cleanly',
                    'Sand or tape edges if they\'re rough',
                    'Decorate the box exterior before adding cables',
                    'Paint with acrylic paint or cover with decorative paper',
                    'Let decoration dry completely',
                    'Place power strip inside box',
                    'Plug devices into power strip',
                    'Thread cables through the appropriate holes',
                    'Use rubber bands or velcro ties to bundle excess cable inside',
                    'This prevents tangling and keeps interior organized',
                    'Place lid on box (or leave open for ventilation)',
                    'Optional: cut small ventilation holes in top',
                    'Position box near wall outlet for easy access'
                ],
                'tips': ['Label each cable before hiding for easy identification', 'Ensure adequate ventilation to prevent overheating', 'Use removable lid for easy access']
            },
            {
                'id': 5,
                'title': 'Kids Play Kitchen',
                'description': 'Imaginative play set for toddlers and preschoolers',
                'difficulty': 'Hard',
                'time': '2 hours',
                'materials': ['Large cardboard box', 'Smaller boxes', 'Paper plates', 'Bottle caps', 'Paint', 'Glue', 'Markers'],
                'steps': [
                    'Use a large box (refrigerator or washing machine box ideal)',
                    'Stand box upright and determine front panel',
                    'Draw oven door outline (10x10 inches) in lower section',
                    'Cut three sides of door, leaving top edge as hinge',
                    'Door should open downward like a real oven',
                    'Above oven, mark 4 circles for stovetop burners',
                    'Use a bowl to trace 4-inch diameter circles',
                    'Cut burner holes completely through for 3D effect',
                    'Cut a small box to attach as oven knobs panel',
                    'Glue bottle caps as knobs - these can turn',
                    'Cut another small box for sink area',
                    'Cut a circular hole and insert a small bowl as sink basin',
                    'Paint the entire kitchen in bright, child-friendly colors',
                    'Use blue for oven door window, silver for sink faucet',
                    'Add details with markers: temperature numbers, brand logos',
                    'Paint paper plates black and glue over burner holes',
                    'Glue red/orange paper under burner plates for "flames" effect',
                    'Add shelves inside using cardboard strips for storage',
                    'Include play food items or let children use toys',
                    'Reinforce all joints with extra tape for durability'
                ],
                'tips': ['Involve kids in decorating for more fun', 'Add hooks for hanging play utensils', 'Make matching cardboard food items']
            }
        ]
    }
    
    return diy_database.get(material_type, diy_database['plastic bottle'])

def generate_step_by_step_guide(idea_id, material_type):
    """
    Generate detailed step-by-step instructions for a specific DIY idea
    """
    ideas = get_curated_diy_ideas(material_type)
    
    for idea in ideas:
        if idea['id'] == idea_id:
            return idea
    
    return None

# API Routes
@app.route('/')
def home():
    """Serve a simple test page"""
    return """
    <h1>DIY AI Backend is Running!</h1>
    <p>Backend server is working. Now open your frontend HTML file.</p>
    <p>API endpoints:</p>
    <ul>
        <li>POST /process-image - Process uploaded image</li>
        <li>POST /get-guide - Get detailed guide for specific idea</li>
        <li>GET /health - Health check</li>
    </ul>
    """

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "AI models loaded"})

@app.route('/process-image', methods=['POST'])
def process_image():
    """Main API endpoint to process uploaded images"""
    try:
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({"error": "No image provided"}), 400
        
        print("Preprocessing image...")
        preprocessed_image = preprocess_image(image_data)
        
        print("Classifying material...")
        material_type, confidence = classify_material(preprocessed_image)
        
        print(f"Getting curated DIY ideas for {material_type}...")
        diy_ideas = get_curated_diy_ideas(material_type)
        
        # Return only basic info for the list view
        ideas_summary = []
        for idea in diy_ideas:
            ideas_summary.append({
                'id': idea['id'],
                'title': idea['title'],
                'description': idea['description'],
                'difficulty': idea['difficulty'],
                'time': idea['time']
            })
        
        response = {
            "success": True,
            "material_type": material_type,
            "confidence": float(confidence),
            "diy_ideas": ideas_summary
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/get-guide', methods=['POST'])
def get_guide():
    """Get detailed step-by-step guide for a specific DIY idea"""
    try:
        data = request.json
        idea_id = data.get('idea_id')
        material_type = data.get('material_type')
        
        if not idea_id or not material_type:
            return jsonify({"error": "Missing idea_id or material_type"}), 400
        
        print(f"Fetching guide for idea {idea_id}...")
        guide = generate_step_by_step_guide(idea_id, material_type)
        
        if guide:
            return jsonify({
                "success": True,
                "guide": guide
            })
        else:
            return jsonify({"error": "Guide not found"}), 404
        
    except Exception as e:
        print(f"Error fetching guide: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting DIY AI Backend Server...")
    load_models()
    print("Server ready!")
    app.run(debug=True, host='0.0.0.0', port=5000)