from pathlib import Path
from PIL import Image
root=Path(__file__).resolve().parents[1]
src=root/'assets'/'logo_256.png'; out=root/'assets'/'app.ico'
im=Image.open(src).convert('RGBA'); im.save(out,format='ICO',sizes=[(16,16),(24,24),(32,32),(48,48),(64,64),(128,128),(256,256)])
print(out)
