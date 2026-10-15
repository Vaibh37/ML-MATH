"""Render exact trigonometry reference tables. Requires Pillow and DejaVu Sans."""
from pathlib import Path
import sys
from PIL import Image, ImageDraw, ImageFont

W, H = 2400, 2560
im = Image.new('RGB', (W,H), '#f4f7fb')
d = ImageDraw.Draw(im)
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
def text(x,y,s,size=32,color='#19304b',bold=False):
    d.text((x,y),s,font=ImageFont.truetype(BOLD if bold else FONT,size),fill=color)
def table(x,y,w,title,heads,rows,footer):
    text(x,y,title,40,bold=True)
    top=y+75; rh=68; cw=w/len(heads)
    d.rounded_rectangle((x,top,x+w,top+rh*(len(rows)+1)),radius=16,fill='white')
    d.rectangle((x,top,x+w,top+rh),fill='#193b61')
    for j,v in enumerate(heads): text(x+j*cw+24,top+16,v,30,'white',True)
    for i,row in enumerate(rows):
        yy=top+(i+1)*rh
        if i%2==0: d.rectangle((x,yy,x+w,yy+rh),fill='#e9f0f7')
        for j,v in enumerate(row): text(x+j*cw+24,yy+15,v,32)
    for i,line in enumerate(footer): text(x,top+rh*(len(rows)+1)+22+i*42,line,28)

text(90,65,'TRIGONOMETRY',68,bold=True)
text(90,155,'Normal + inverse functions • Exact values • Angles in degrees',34,'#526b85')
table(90,255,2220,'01   Normal values: angle → ratio',['Angle θ','sin θ','cos θ','tan θ'],[
['0°','0','1','0'],['30°','1/2','√3/2','1/√3'],['45°','√2/2','√2/2','1'],
['60°','√3/2','1/2','√3'],['90°','1','0','Undefined'],['180°','0','−1','0'],
['270°','−1','0','Undefined'],['360°','0','1','0']],
['tan θ = sin θ / cos θ     •     Undefined when cos θ = 0     •     sin² θ + cos² θ = 1'])
inputs=['−1','−√3/2','−√2/2','−1/2','0','1/2','√2/2','√3/2','1']
table(90,1090,690,'02   Inverse sine',['x','arcsin(x)'],list(zip(inputs,['−90°','−60°','−45°','−30°','0°','30°','45°','60°','90°'])),['Domain: −1 ≤ x ≤ 1','Range: −90° ≤ angle ≤ 90°'])
table(855,1090,690,'03   Inverse cosine',['x','arccos(x)'],list(zip(inputs,['180°','150°','135°','120°','90°','60°','45°','30°','0°'])),['Domain: −1 ≤ x ≤ 1','Range: 0° ≤ angle ≤ 180°'])
table(1620,1090,690,'04   Inverse tangent',['x','arctan(x)'],list(zip(['−√3','−1','−1/√3','0','1/√3','1','√3'],['−60°','−45°','−30°','0°','30°','45°','60°'])),['Domain: all real numbers','Range: −90° < angle < 90°','Endpoints are excluded.'])
d.rounded_rectangle((90,2040,2310,2470),radius=24,fill='#193b61')
text(130,2070,'READ INVERSE FUNCTIONS CORRECTLY',34,'white',True)
for i,line in enumerate([
'sin⁻¹(x) = arcsin(x), not 1/sin(x). The same distinction applies to cos and tan.',
'Inverse functions return the principal angle: sin(150°) = 1/2, but arcsin(1/2) = 30°.',
'For nonzero vectors: angle = arccos(cosine similarity).  0 → 90°;  −1 → 180°.',
'Python uses radians: math.degrees(math.acos(0)) gives 90.0.',
'Start with 0°, 30°, 45°, 60°, 90°.  √2/2 = 1/√2 and 1/√3 = √3/3.'
]): text(130,2140+i*59,line,30,'#edf5ff')
out=Path(sys.argv[1] if len(sys.argv)>1 else 'trigonometry_normal_inverse.png')
out.parent.mkdir(parents=True,exist_ok=True)
im.save(out,optimize=True)
print(f'Saved {out}: {W} × {H}')
