import vapoursynth as vs
import mvsfunc as mvf
import vsaa
import vsscale
core = vs.core

# A dumber version of nnedi3_rpow2 that hardcodes to do 2X upscaling
# original gist: https://gist.github.com/4re/342624c9e1a144a696c6
def nnedi3_rpow2(clip, correct_shift=True,
                 kernel="spline36", nsize=0, nns=3, qual=None, etype=None, pscrn=None,
                 opt=True, int16_prescreener=None, int16_predictor=None, exp=None):

    rfactor = 2
    width = clip.width * rfactor
    height = clip.height * rfactor

    hshift = 0.0
    vshift = -0.5
    pkdnnedi = dict(
        dh=True,
        nsize=nsize,
        nns=nns,
        qual=qual,
        etype=etype,
        pscrn=pscrn,
    )
    pkdchroma = dict(kernel=kernel, sy=-0.5, planes=[2, 3, 3])

    nnedi3 = core.nnedi3.nnedi3
    pkdnnedi.update(
            opt=opt,
            int16_prescreener=int16_prescreener,
            int16_predictor=int16_predictor,
            exp=exp,
        )

    if correct_shift or clip.format.subsampling_h:
        if hasattr(core, "fmtc") is not True:
            raise RuntimeError("nnedi3_rpow2: fmtconv plugin is required.")

    last = clip
    field = 1
    last = nnedi3(last, field=field, **pkdnnedi)
    last = core.std.Transpose(last)
    if last.format.subsampling_w:
        # Apparently always using field=1 for the horizontal pass somehow
        # keeps luma/chroma alignment.
        field = 1
        hshift = hshift * 2 - 0.5
    else:
        hshift = -0.5
    last = nnedi3(last, field=field, **pkdnnedi)
    last = core.std.Transpose(last)

    if clip.format.subsampling_h:
        last = core.fmtc.resample(last, w=last.width, h=last.height, **pkdchroma)

    if correct_shift is True:
        last = core.fmtc.resample(last, w=width, h=height, kernel=kernel, sx=hshift, sy=vshift)

    if last.format.id != clip.format.id:
        last = core.fmtc.bitdepth(last, csp=clip.format.id)

    return last

# This is the BD video source demuxed by dgdemux
src = core.bs.VideoSource(source="/home/hq/Videos/efp/efp.avc")
# This is a reference encode by VCB Studio
src_ref = core.bs.VideoSource(source="/home/hq/Videos/efp/efp.vcb.mkv")

src0 = src

src = mvf.Depth(src, 16)

# Did not use this approach as JET suggested, this ArtCNN + EEDI3 failed to deliver same AA quality as the one I am using now
#src = vsaa.based_aa(src, rfactor=2, mask=False, alpha=0.5, beta=0.2)

d_src = core.descale.Debicubic(src, width=1280, height=720, b=0, c=1/2)

# TODO: denoise

# TODO: deband

# Use nnedi3 to scale up 2X then resize to 1080p
src = nnedi3_rpow2(d_src, qual=2).resize.Spline36(width=1920, height=1080)

src = mvf.Depth(src, 10)

src0.set_output(0)
src.set_output(1)
src_ref.set_output(3)
