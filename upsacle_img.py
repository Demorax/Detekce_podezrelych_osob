import torch
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# --- Původní nastavení ---
# _SR_NET = None  <-- TOTO SMAŽ
# _SR_SCALE = 4
# _SR_PB = ...    <-- TOTO UŽ NEPOTŘEBUJEŠ

# --- NOVÉ NASTAVENÍ (PYTORCH) ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Upscaling poběží na: {DEVICE}")

# Inicializace modelu (stáhne se automaticky při prvním spuštění)
# Používáme model 'RealESRGAN_x4plus' - je mnohem kvalitnější než ESPCN
def init_upscaler():
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upscaler = RealESRGANer(
        scale=4,
        model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
        model=model,
        tile=0,  # 0 pro celý obrázek, sniž číslo (např. 400), pokud ti dojde VRAM
        tile_pad=10,
        pre_pad=0,
        half=True,  # Použije FP16 (rychlejší na RTX 3090)
        device=DEVICE
    )
    return upscaler

# Vytvoříme instanci (udělej to jen jednou na začátku skriptu!)
_SR_UPSCALER = init_upscaler()

# --- FUNKCE PRO POUŽITÍ VE SMYČCE ---
def upscale_image_gpu(img):
    """
    Nahrazuje tvůj původní cv2.dnn_superres kód.
    Vstup: numpy array (OpenCV image)
    Výstup: upscalovaný numpy array
    """
    # RealESRGAN vrací (output, _)
    output, _ = _SR_UPSCALER.enhance(img, outscale=4)
    return output

# --- PŘÍKLAD POUŽITÍ (tam, kde jsi měl dnn_superres) ---
# Místo: enhanced_img = dnn_superres.upsample(img)
# Použij:
# enhanced_img = upscale_image_gpu(img)