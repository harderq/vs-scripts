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

def denoise(clip):
    linemask = kgf.retinex_edgemask(clip).std.Maximum()
    ref_clip = core.dfttest.DFTTest(clip, sigma=8)
    denoised_clip = mvf.BM3D(clip, sigma=3.0, radius1=0, ref=ref_clip)
    clip = core.std.MaskedMerge(denoised_clip, clip, linemask)
    clip = denoised_clip
    return clip

def deband(clip, linemask):
    debanded_clip = core.neo_f3kdb.Deband(clip, range = 15, y = 60, cb = 20, cr = 20, grainy = 0, grainc = 0, dynamic_grain = False, sample_mode=4, output_depth = 16)
    clip = core.std.MaskedMerge(debanded_clip, clip, linemask)
    return clip

def print_screenshots(src0, src1):
    awf.ScreenGen(src0, HOMEPATH, "a")
    awf.ScreenGen(src1, HOMEPATH, "b")

bd_src = core.bs.VideoSource(source=f"{HOMEPATH}/efp.avc")

src = mvf.Depth(bd_src, 16)

src = edgefix(src)

y, _, _ = kgf.split(src)
descaled_y = core.descale.Debicubic(y, width=1280, height=720, b=1/3, c=1/3)

# AA on luma plane
descaled_y = denoise(descaled_y)
aa_y = eedi3(descaled_y, alpha=0.1, beta=0.3, gamma=200)
aa_y = core.resize.Spline36(aa_y, width=1920, height=1080)

src = core.std.ShufflePlanes([aa_y, src], planes=[0, 1, 2], colorfamily=vs.YUV)

linemask = kgf.retinex_edgemask(src)
src = deband(src, linemask)

src = mvf.Depth(src, 10)

bd_src.set_output(0)
src.set_output(1)
