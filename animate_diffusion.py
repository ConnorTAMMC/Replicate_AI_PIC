import replicate
from dotenv import load_dotenv
load_dotenv()

# from IPython.display import Image
#https://civitai.com/models/81458
# Image(url=output[0])
lora_url = "https://replicate.com/zylim0702/sdxl-lora-customize-model."
lora='makima_offset.safetensors'
image = open("iccc.jpg", "rb")
# or...
# image = "https://example.com/mystery.jpg"
#
output = replicate.run(
    # "pagebrain/absolutereality-v1-8-1:1c9d76b62790e891aefc6c015e576a2ba27ddb08d013936a4e6d205210e2e332",
"pagebrain/dreamshaper-v8:6cb38fe374c4fd4d5bb6a18dcdd71b08512f25bbf1753f8db4bb22f1d5fea9be",
#        "pagebrain/epicrealism-v5:222465e57e4d9812207f14133c9499d47d706ecc41a8bf400120285b2f030b42",
       #(photorealistic:1.4)
    input={"prompt":"caption of the white truffle ice-cream cone,the brand name is DALLOYAU suah as the choclate ice cream cone, extremely detailed CG unity 8k wallpaper,masterpiece, (photorealistic:1.4), best quality" ,
           #Specify things to not see in the output. Supported embeddings: realisticvision-negative-embedding, BadDream, EasyNegative, negative_hand-neg, ng_deepnegative_v1_75t, FastNegativeV2, UnrealisticDream
           "negative_prompt":"BadDream, (UnrealisticDream:1.2),text, watermark, low quality, medium quality, blurry, censored, wrinkles, deformed, mutated text, watermark, blurry, censored, wrinkles, deformed, mutated",
           # "image_dimensions":"512x512",
           #512, 576, 640, 704, 768, 832, 896, 960, 1024
           "width":576, #576
           "height":960,  #960
           # "image": image,
           "num_outputs":1,
           "lora_urls":False,
           #Prompt strength when using init image.
           #1.0 corresponds to full destruction of information in init image
           "prompt_strength" : 1,
           #Number of denoising steps
           "num_inference_steps":50,
           # Allowed values:DDIM, DPMSolverMultistep, K_EULER_ANCESTRAL,HeunDiscrete, KarrasDPM, K_EULER_ANCESTRAL, K_EULER, PNDM, KLMS
           # # Default value: K_EULER
           "scheduler":"DPMSolverMultistep",
           "guidance_scale":7,
           "safety_checker":False,
           "Seed":2039823409428,
           }
)
print(output)