import vapoursynth as vs
import mvsfunc as mvf
import awsmfunc as awf
import kagefunc as kgf
core = vs.core

# Key frames:
# 13200 (aliasing on the weapon stick)
# 15630 (dark scene, noise, halo)
# 15785 (banding, AA artifacts)
# 31900 (dark scene, cloud at the window)
# 38700 (city scene)
# 113865 (dark UI interface)

def nnedi3_sclip(clip):
    return core.nnedi3.nnedi3(clip, field=1, dh=True, nsize=4, nns=4, pscrn=None, opt=True, qual=2)

def eedi3(clip, alpha=None, beta=None, gamma=None):
    clip = core.std.Transpose(clip)
    clip = core.eedi3m.EEDI3(clip, field=1, dh=True, alpha=alpha, beta=beta, gamma=gamma, vcheck=3, sclip=nnedi3_sclip(clip))
    clip = core.std.Transpose(clip)
    clip = core.eedi3m.EEDI3(clip, field=1, dh=True, alpha=alpha, beta=beta, gamma=gamma, vcheck=3, sclip=nnedi3_sclip(clip))

    clip=correct_edi_shift(clip,rfactor=2)
    return clip

def correct_edi_shift(clip,rfactor,plugin="zimg"):
    if clip.format.subsampling_w==1:
        hshift=-rfactor/2+0.5 # hshift(steps+1)=2*hshift(steps)-0.5
    else :
        hshift=-0.5
    if clip.format.subsampling_h==0:
        clip=core.resize.Spline36(clip=clip,width=clip.width,height=clip.height,src_left=hshift,src_top=-0.5)
    else :
        Y=core.std.ShufflePlanes(clips=clip, planes=0, colorfamily=vs.GRAY)
        U=core.std.ShufflePlanes(clips=clip, planes=1, colorfamily=vs.GRAY)
        V=core.std.ShufflePlanes(clips=clip, planes=2, colorfamily=vs.GRAY)
        Y=core.resize.Spline36(clip=Y,width=clip.width,height=clip.height,src_left=hshift,src_top=-0.5)
        U=core.resize.Spline36(clip=U,width=clip.width,height=clip.height,src_left=hshift/2,src_top=-0.5)
        V=core.resize.Spline36(clip=V,width=clip.width,height=clip.height,src_left=hshift/2,src_top=-0.5)
        clip=core.std.ShufflePlanes(clips=[Y,U,V], planes=[0,0,0], colorfamily=vs.YUV)
    return clip

def edgefix(clip):
    clip = awf.bbmod(clip, left=2, right=2, top=2, bottom=2, planes=[0, 1, 2], blur=500, scale_thresh=False, cpass2=False)
    return clip

def denoise(clip, sigma):
    clip = mvf.Depth(clip, 32)
    basic_clip = core.bm3dhip.BM3D(clip, sigma=sigma, radius=0)
    clip = core.bm3dhip.BM3D(clip, sigma=sigma, ref=basic_clip, radius=0)
    clip = mvf.Depth(clip, 16)
    return clip

def deband(clip, linemask):
    debanded_clip = core.neo_f3kdb.Deband(clip, range = 15, y = 60, cb = 20, cr = 20, grainy = 0, grainc = 0, dynamic_grain = False, sample_mode=4, output_depth = 16)
    clip = core.std.MaskedMerge(debanded_clip, clip, linemask)
    return clip

def grain(clip):
    clip = kgf.adaptive_grain(clip, strength=0.2, static=True, luma_scaling=10)
    return clip

def print_screenshots(src0, src1):
    awf.ScreenGen(src0, HOMEPATH, "a")
    awf.ScreenGen(src1, HOMEPATH, "b")

bd_src = core.bs.VideoSource(source=f"{HOMEPATH}/efp.remux.mkv")

src = mvf.Depth(bd_src, 16)

src = edgefix(src)
unfiltered_src = src

linemask = kgf.retinex_edgemask(src)

y, _, _ = kgf.split(src)
descaled_y = core.descale.Debicubic(y, width=1280, height=720, b=1/3, c=1/3)

descaled_y = denoise(descaled_y, 2.0)

aa_y = eedi3(descaled_y, alpha=0.1, beta=0.3, gamma=200)
aa_y = core.resize.Spline36(aa_y, width=1920, height=1080)

src = core.std.ShufflePlanes([aa_y, src], planes=[0, 1, 2], colorfamily=vs.YUV)

src = denoise(src, [0, 2.0, 2.0])

src = deband(src, linemask)

src = grain(src)

src = core.remap.Rfs(src, unfiltered_src, mappings="[0 599] [141874 146687] [147456 149201] [149468 149639]")

src = mvf.Depth(src, 10)

src.set_output(0)
