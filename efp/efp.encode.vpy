import vapoursynth as vs
import mvsfunc as mvf
core = vs.core

def nnedi3_supersample(clip):
    return core.nnedi3.nnedi3(clip, field=1, dh=True, nsize=0, nns=3, pscrn=None, opt=True, qual=2)

def eedi3(clip, alpha=0.125, beta=0.25, gamma=0):
    clip = core.eedi3m.EEDI3(clip, field=1, dh=True, alpha=alpha, beta=beta, gamma=gamma, vcheck=3, sclip=nnedi3_supersample(clip))
    clip = core.std.Transpose(clip)
    clip = core.eedi3m.EEDI3(clip, field=1, dh=True, alpha=alpha, beta=beta, gamma=gamma, vcheck=3, sclip=nnedi3_supersample(clip))
    clip = core.std.Transpose(clip)
    return clip

src = core.bs.VideoSource(source="/home/hq/Videos/efp/efp.avc")
src0 = src

src = mvf.Depth(src, 16)

# TODO: Dirty-line fix

# Descale to 720p
src = core.descale.Debicubic(src, width=1280, height=720, b=0, c=0.5)

# AA
src = eedi3(src, alpha=0.6, beta=0.2)

# TODO: Degrain

# TODO: Deband

src = core.resize.Spline36(src, width=1920, height=1080)

src = mvf.Depth(src, 10)

src0.set_output(0)
src.set_output(1)
