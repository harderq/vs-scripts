import vapoursynth as vs
import mvsfunc as mvf
import awsmfunc as awf
import kagefunc as kgf
core = vs.core

# Key frames:
# 13200 (aliasing on the weapon stick)
# 15630 (dark scene, noise, halo)
# 15785 (banding, AA artifacts)

def nnedi3_supersample(clip):
    return core.nnedi3.nnedi3(clip, field=1, dh=True, nsize=0, nns=3, pscrn=None, opt=True, qual=2)

def eedi3(clip, alpha=0.2, beta=0.25, gamma=0):
    hshift = 0.0
    vshift = -0.5
    scaled_clip = core.eedi3m.EEDI3(clip, field=1, dh=True, alpha=alpha, beta=beta, gamma=gamma, vcheck=3, sclip=nnedi3_supersample(clip))
    scaled_clip = core.std.Transpose(scaled_clip)
    if scaled_clip.format.subsampling_w:
        hshift = hshift * 2 - 0.5
    else:
        hshift = -0.5
    scaled_clip = core.eedi3m.EEDI3(scaled_clip, field=1, dh=True, alpha=alpha, beta=beta, gamma=gamma, vcheck=3, sclip=nnedi3_supersample(scaled_clip))
    scaled_clip = core.std.Transpose(scaled_clip)
    scaled_clip = core.fmtc.resample(scaled_clip, w=clip.width, h=clip.height, kernel="Spline36", sx=hshift, sy=vshift)
    return scaled_clip

def dirtylinefix(clip):
    clip = awf.bbmod(clip, left=2, right=2, top=2, bottom=2, planes=[0, 1, 2], blur=500, scale_thresh=False, cpass2=False)
    return clip

def denoise(clip):
    linemask = kgf.retinex_edgemask(clip).std.Maximum()
    ref_clip = core.dfttest.DFTTest(clip, sigma=8)
    denoised_clip = mvf.BM3D(clip, sigma=3.0, radius1=0, ref=ref_clip)
    clip = core.std.MaskedMerge(denoised_clip, clip, linemask)
    clip = denoised_clip
    return clip

def deband(clip):
    linemask = kgf.retinex_edgemask(clip).std.Maximum()
    debanded_clip = core.neo_f3kdb.Deband(clip, range = 15, y = 60, cb = 20, cr = 20, grainy = 0, grainc = 0, dynamic_grain = False, sample_mode=4, output_depth = 16)
    clip = core.std.MaskedMerge(debanded_clip, clip, linemask)
    return clip

bd_src = core.bs.VideoSource(source="/home/hq/Videos/efp/efp.avc")
src = mvf.Depth(bd_src, 16)

src = dirtylinefix(src)

descaled_src = core.descale.Debicubic(src, width=1280, height=720, b=0, c=1/2)

descaled_src = denoise(descaled_src)

aa_src = eedi3(descaled_src, alpha=0.1, beta=0.8, gamma=200)

src = core.resize.Spline36(aa_src, width=1920, height=1080)

src = deband(src)

src = mvf.Depth(src, 10)

bd_src.set_output(0)
src.set_output(1)
