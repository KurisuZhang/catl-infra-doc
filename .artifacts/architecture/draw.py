from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from html import escape
import math
import xml.etree.ElementTree as ET
out=Path('/Users/lin/Desktop/catl-infra')
W,H,S=1600,1340,2
im=Image.new('RGB',(W*S,H*S),'white'); d=ImageDraw.Draw(im)
svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><title>Token 魔方实验台架构图</title><desc>业务请求进入部署在3台CPU Worker上的MoMA和并行科技MaaS；同一CPU资源池还部署ICT设备管理软件，调用6台T4或2台910B上的模型服务。T4计划部署小型Qwen模型和YOLO，910B计划部署GLM 5.2和DeepSeek V4 Flash。3台Master统一管理调度Worker。Prometheus统一监控。</desc><rect width="1600" height="1340" fill="white"/>']
ink='#172C43'; muted='#607287'; blue='#306BB6'; teal='#297A79'; line='#CBD7E3'; gray='#7C8CA0'
fp='/System/Library/Fonts/Hiragino Sans GB.ttc'
def rect(box,fill,stroke=None,r=16,width=1):
    d.rounded_rectangle(tuple(int(v*S) for v in box),radius=r*S,fill=fill,outline=stroke,width=width*S)
    x,y,x2,y2=box; svg.append(f'<rect x="{x}" y="{y}" width="{x2-x}" height="{y2-y}" rx="{r}" fill="{fill}" stroke="{stroke or "none"}" stroke-width="{width}"/>')
def text(x,y,t,size=26,color=ink,anchor='lm',bold=False):
    f=ImageFont.truetype(fp,size*S,index=1 if bold else 0)
    d.text((x*S,y*S),t,font=f,fill=color,anchor=anchor)
    alignment={'lm':'start','mm':'middle','rm':'end'}[anchor]
    svg.append(f'<text x="{x}" y="{y}" font-family="Hiragino Sans GB, Microsoft YaHei, sans-serif" font-size="{size}" font-weight="{600 if bold else 400}" text-anchor="{alignment}" dominant-baseline="central" fill="{color}">{escape(t)}</text>')
def path(points,color=blue,width=3,dash=False,arrow=False):
    for (x,y),(xx,yy) in zip(points,points[1:]):
        if dash:
            length=math.hypot(xx-x,yy-y)
            for k in range(0,int(length),14):
                a=k/length;b=min(k+7,length)/length
                d.line(((x+(xx-x)*a)*S,(y+(yy-y)*a)*S,(x+(xx-x)*b)*S,(y+(yy-y)*b)*S),fill=color,width=width*S)
        else:d.line((x*S,y*S,xx*S,yy*S),fill=color,width=width*S)
    pts=' '.join(f'{x},{y}' for x,y in points)
    svg.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}"'+(' stroke-dasharray="7 7"' if dash else '')+'/>')
    if arrow:
        x,y=points[-1];px,py=points[-2];ang=math.atan2(y-py,x-px)
        tri=[(x,y),(x-12*math.cos(ang)+6*math.sin(ang),y-12*math.sin(ang)-6*math.cos(ang)),(x-12*math.cos(ang)-6*math.sin(ang),y-12*math.sin(ang)+6*math.cos(ang))]
        d.polygon([(int(a*S),int(b*S)) for a,b in tri],fill=color)
        svg.append('<polygon points="'+' '.join(f'{a},{b}' for a,b in tri)+f'" fill="{color}"/>')

def logo(name,cx,cy,maxw=185,maxh=48):
    asset=out/'.artifacts'/'architecture'
    pic=Image.open(asset/(name+'.png')).convert('RGBA')
    origw,origh=pic.size
    bounds=pic.getbbox()
    pic=pic.crop(bounds)
    scale=min(maxw/pic.width,maxh/pic.height)
    w,h=pic.width*scale,pic.height*scale
    x,y=cx-w/2,cy-h/2
    scaled=pic.resize((round(w*S),round(h*S)),Image.Resampling.LANCZOS)
    im.paste(scaled,(round(x*S),round(y*S)),scaled)
    root=ET.fromstring((asset/(name+'.svg')).read_text())
    vb=list(map(float,root.get('viewBox').replace(',',' ').split()))
    l,t,r,b=bounds
    root.set('viewBox',f'{vb[0]+l/origw*vb[2]} {vb[1]+t/origh*vb[3]} {(r-l)/origw*vb[2]} {(b-t)/origh*vb[3]}')
    root.set('x',str(x)); root.set('y',str(y));root.set('width',str(w));root.set('height',str(h))
    svg.append(ET.tostring(root,encoding='unicode'))

text(70,64,'Token 魔方实验台架构',42,bold=True)
text(70,119,'漳湾机房  ·  14 台服务器',24,muted)
rect((530,160,1120,240),'#F0F4F9',r=14)
text(825,200,'业务应用 / 压测工具',29,anchor='mm',bold=True)
rect((60,305,1540,1210),'#F8FAFD',line,r=22,width=2)
text(100,350,'Kubernetes 集群',31,bold=True)
text(100,395,'containerd 容器运行时',23,muted)
rect((490,410,1500,1110),'#FFFFFF','#D8E2EC',r=18,width=2)
text(520,440,'Worker 节点',22,muted)
rect((100,505,410,740),'#EAF0F6',r=18)
text(255,550,'控制节点',23,muted,'mm')
text(255,609,'3 台 Master',37,ink,'mm',True)
text(255,680,'集群管理与资源调度',23,muted,'mm')
path([(410,621),(490,621)],gray,3,True,True)
# Two peer software deployments in the CPU worker pool.
rect((530,480,1450,735),'#EAF2FF','#BCCFE9',r=16,width=2)
text(560,516,'3 台通算 CPU Worker',31,ink,bold=True)
path([(560,548),(1420,548)],'#CCDDF2',1)
path([(1090,575),(1090,705)],'#CCDDF2',1)
text(810,595,'MoMA / 并行科技 MaaS',29,blue,'mm',True)
text(810,647,'API 接入 · 鉴权路由',23,muted,'mm')
text(810,689,'Token 计量 · 流量计费',23,muted,'mm')
text(1270,610,'ICT 设备管理软件',27,blue,'mm',True)
text(1270,666,'设备管理',23,muted,'mm')
# Request path belongs to MaaS, not ICT.
path([(825,240),(825,480)],blue,3,False,True)
text(855,351,'API 请求',23,blue)
path([(810,735),(810,795)],blue,3)
path([(750,795),(1230,795)],blue,3)
path([(750,795),(750,850)],blue,3,False,True)
path([(1230,795),(1230,850)],blue,3,False,True)
text(850,770,'模型调用',22,blue)
rect((530,850,970,1080),'#EDF4FD','#C4D7ED',r=16,width=2)
rect((1010,850,1450,1080),'#EDF7F4','#C5DEDA',r=16,width=2)
logo('nvidia',750,895)
logo('ascend',1230,895)
text(750,958,'6 台 T4 Worker',31,ink,'mm',True)
text(750,1007,'小型 Qwen 模型 / YOLO',25,blue,'mm')
text(750,1049,'计划部署 · 先跑通全流程',22,muted,'mm')
text(1230,958,'2 台 910B Worker',31,ink,'mm',True)
text(1230,1007,'GLM 5.2 / DeepSeek V4 Flash',24,teal,'mm')
text(1230,1049,'计划部署 · 后续适配验证',22,muted,'mm')
path([(100,1141),(1500,1141)],line,1)
text(100,1176,'统一监控',23,ink,bold=True)
text(260,1176,'Prometheus  ·  硬件、节点、容器及模型服务',23,muted)
path([(75,1264),(145,1264)],blue,3,False,True)
text(165,1264,'请求调用',21,muted)
path([(360,1264),(430,1264)],gray,3,True,True)
text(450,1264,'管理与调度',21,muted)
svg.append('</svg>')
name='Token魔方实验台架构图'
im.save(out/(name+'.png'),dpi=(240,240))
(out/(name+'.svg')).write_text('\n'.join(svg),encoding='utf-8')
print(out/(name+'.png'))
