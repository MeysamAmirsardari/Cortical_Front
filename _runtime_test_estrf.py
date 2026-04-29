"""End-to-end runtime test: extract & execute the notebook's plotting
logic against local TIMIT.  Renders one figure per family for visual
inspection."""
import sys, os, io, base64, pickle, warnings
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import FancyBboxPatch
from scipy.signal import correlate2d
from scipy.ndimage import gaussian_filter

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit
import librosa

R_CODE = "/Users/eminent/Projects/Cortical_Front/r_code"
sys.path.insert(0, R_CODE)
sys.path.insert(0, "/Users/eminent/Projects/Cortical_Front")
os.chdir(R_CODE)

from supervisedSTRF import supervisedSTRF

# ── Load checkpoint ──
with open("nishit_trained_models/main_jax_phoneme_rec_timit/chkStep_200000.p","rb") as f:
    nn_params, aud_params = pickle.load(f)
ALPHA = float(aud_params["alpha"]); COMP = aud_params["compression_params"]
SR = 16000
SR_PAIRS_MODEL = np.asarray(aud_params["sr"], dtype=np.float64)

model_s = supervisedSTRF(n_phones=62, input_type="audio", encoder_type="strf",
                         decoder_type="cnn", compression_method="power",
                         update_lin=True, use_class=False,
                         conv_feats=[10,20,40], pooling_stride=2)
@jit
def _coch_jax(x): return model_s.apply(nn_params, x, COMP, ALPHA, method=model_s.wav2aud)
def cochlea_np(wav): return np.asarray(_coch_jax(jnp.asarray(wav, dtype=jnp.float32)))

cf = np.linspace(-31,97,129)/24.0
COCHLEA_CF = np.array([round(440*2**f/10)*10 for f in cf])
N_FREQ = 129

def cochleagram_to_db(coch, db_floor=60.0):
    mag = np.abs(coch.astype(np.float64))
    pk = float(mag.max())
    if pk <= 0: return np.full(mag.shape, -db_floor, dtype=np.float32)
    db = 20.0*np.log10(mag/pk + 1e-12)
    return np.clip(db, -db_floor, 0.0).astype(np.float32)

# ── Decode aggregates ──
b64_path = Path("/Users/eminent/Projects/Cortical_Front/.claude/worktrees/cranky-kowalevski/_estrf_aggregates_b64.txt")
b64 = "".join(b64_path.read_text().split())
blob = base64.b64decode(b64)
with np.load(io.BytesIO(blob), allow_pickle=False) as d:
    AGG_PHONES = [str(p) for p in d["phones"]]
    AGG_COEFS = np.asarray(d["coefs"], dtype=np.float64)
    AGG_SR_PAIRS = np.asarray(d["sr_pairs"], dtype=np.float64)
    AGG_BAND_EDGES = np.asarray(d["band_edges_hz"], dtype=np.float64)
    N_BANDS = int(d["n_bands"])

BASE_HZ = 125.0; KERNEL_T_S = 0.30; KERNEL_N_T = 60; SMOOTH_SIGMA = 0.8

def _gabor(omega, Omega, t, f_oct, cycles=0.5, of=2.0, Of=0.5):
    sw = max(abs(omega), of); sW = max(abs(Omega), Of)
    return np.outer(np.exp(-(t/(cycles/sw))**2/2), np.exp(-(f_oct/(cycles/sW))**2/2))

def reconstruct_estrf(coef_vec):
    n_strfs = AGG_SR_PAIRS.shape[0]
    coef_2d = coef_vec.reshape(N_BANDS, n_strfs)
    bc = np.log2(np.sqrt(np.maximum(AGG_BAND_EDGES[:,0],1.0)*AGG_BAND_EDGES[:,1])/BASE_HZ)
    t = np.linspace(-KERNEL_T_S/2, KERNEL_T_S/2, KERNEL_N_T)
    f_oct = np.log2(COCHLEA_CF/BASE_HZ)
    e = np.zeros((KERNEL_N_T, N_FREQ))
    for b in range(N_BANDS):
        for k in range(n_strfs):
            w = coef_2d[b,k]
            if w == 0: continue
            e += w * _gabor(AGG_SR_PAIRS[k,1], AGG_SR_PAIRS[k,0], t, f_oct - bc[b])
    return gaussian_filter(e, sigma=SMOOTH_SIGMA) if SMOOTH_SIGMA > 0 else e

ESTRF = {p: reconstruct_estrf(AGG_COEFS[i]) for i, p in enumerate(AGG_PHONES)}

def _normalised_template(K):
    K = np.asarray(K, dtype=np.float64); K = K - K.mean()
    n = float(np.linalg.norm(K))
    return K / n if n > 0 else K

def apply_estrf(coch, kern):
    K = _normalised_template(kern)
    n_t, _ = K.shape
    cp = np.abs(coch); pad = n_t//2
    cp_p = np.pad(cp, ((pad, n_t-1-pad),(0,0)), mode="constant")
    return correlate2d(cp_p, K, mode="valid")[:,0]

PHONE_61_TO_39 = {
    "aa":"aa","ae":"ae","ah":"ah","ao":"aa","aw":"aw","ax":"ah","ax-h":"ah",
    "axr":"er","ay":"ay","b":"b","bcl":"sil","ch":"ch","d":"d","dcl":"sil",
    "dh":"dh","dx":"dx","eh":"eh","el":"l","em":"m","en":"n","eng":"ng",
    "epi":"sil","er":"er","ey":"ey","f":"f","g":"g","gcl":"sil","h#":"sil",
    "hh":"hh","hv":"hh","ih":"ih","ix":"ih","iy":"iy","jh":"jh","k":"k",
    "kcl":"sil","l":"l","m":"m","n":"n","ng":"ng","nx":"n","ow":"ow",
    "oy":"oy","p":"p","pau":"sil","pcl":"sil","q":"","r":"r","s":"s",
    "sh":"sh","t":"t","tcl":"sil","th":"th","uh":"uh","uw":"uw","ux":"uw",
    "v":"v","w":"w","y":"y","z":"z","zh":"sh",
}
PHONEME_FAMILIES = {
    "Vowels":["iy","ih","eh","ae","aa","ah","ao","uh","uw","er","ey","ay","oy","aw","ow"],
    "Stops":["p","t","k","b","d","g"],
    "Fricatives":["f","th","s","sh","v","dh","z","hh"],
    "Nasals":["m","n","ng"],
    "Liquids/Glides":["l","r","w","y"],
}
FAMILY_COLORS = {"Vowels":"#E63946","Stops":"#F4A261","Affricates":"#E9C46A",
                 "Fricatives":"#2A9D8F","Nasals":"#264653","Liquids/Glides":"#8E7DBE",
                 "Silence":"#888888"}
def family_of(p):
    for fam, m in PHONEME_FAMILIES.items():
        if p in m: return fam
    return "Silence" if p in ("sil","","dx") else None

# ── TIMIT ──
from timit_dataset import TIMITDataset
ds = TIMITDataset(split="TEST")

@dataclass
class PhoneSeg:
    start_sample: int; end_sample: int; phone_61: str; phone_39: str
@dataclass
class Utt:
    utt_id: str; audio: np.ndarray; phone_segs: List["PhoneSeg"]; transcript: str = ""

def to_utt(u):
    segs = [PhoneSeg(ps.start_sample, ps.end_sample, ps.phone,
                     ps.phone_39 or PHONE_61_TO_39.get(ps.phone, ps.phone))
            for ps in u.phone_segments]
    return Utt(u.utterance_id, np.asarray(u.audio, dtype=np.float32), segs,
               getattr(u, "transcript", ""))

all_utts = [to_utt(ds[i]) for i in range(150)]

def pick_utterances(probe, n_picks=2, min_inst=3, max_dur_s=4.5):
    scored = []
    for u in all_utts:
        if len(u.audio) > int(max_dur_s*SR): continue
        n = sum(1 for ps in u.phone_segs if ps.phone_39==probe and (ps.end_sample-ps.start_sample)>=int(0.03*SR))
        if n < min_inst: continue
        nd = len({ps.phone_39 for ps in u.phone_segs})
        scored.append((n*10+nd, u))
    scored.sort(key=lambda x:-x[0])
    return [u for _, u in scored[:n_picks]]

# ── Plot helpers ──
mpl.rcParams.update({"font.family":"DejaVu Sans","figure.facecolor":"white",
                     "savefig.facecolor":"white"})
def _fmt_hz(hz): return f"{hz/1000:g}k" if hz>=1000 else f"{int(hz)}"

def _draw_phone_strip(ax, segs, total_ms):
    ax.set_xlim(0, total_ms); ax.set_ylim(0,1); ax.set_yticks([])
    for s in ("top","right","left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#bbb"); ax.tick_params(axis="x", colors="#666", length=2.5)
    for ps in segs:
        t0=ps.start_sample/SR*1000; t1=ps.end_sample/SR*1000
        c = FAMILY_COLORS.get(family_of(ps.phone_39), "#bbb")
        ax.add_patch(FancyBboxPatch((t0,0.05), max(t1-t0,0.5), 0.90,
            boxstyle="round,pad=0.0,rounding_size=1.2", linewidth=0,
            facecolor=c, alpha=0.92, transform=ax.transData, clip_on=True))
        if (t1-t0) >= 25:
            ax.text((t0+t1)/2, 0.5, ps.phone_39, ha="center", va="center",
                    fontsize=7.5, color="white", fontweight="bold",
                    transform=ax.transData, clip_on=True)

def _draw_kernel(ax, kern, vmax):
    n_t, n_f = kern.shape
    extent = [-KERNEL_T_S/2*1000, KERNEL_T_S/2*1000, -0.5, n_f-0.5]
    im = ax.imshow(kern.T, origin="lower", aspect="auto", cmap="RdBu_r",
                   extent=extent, vmin=-vmax, vmax=vmax, interpolation="bilinear")
    ax.axvline(0, color="black", lw=0.7, ls=":", alpha=0.7)
    ax.set_xlabel("Lag (ms)", fontsize=8); ax.set_ylabel("Hz", fontsize=8)
    target=[200,500,1000,2000,5000]; tk_b,tk_l=[],[]
    for hz in target:
        if COCHLEA_CF[0]<=hz<=COCHLEA_CF[-1]:
            i=int(np.argmin(np.abs(COCHLEA_CF-hz))); tk_b.append(i); tk_l.append(_fmt_hz(hz))
    ax.set_yticks(tk_b); ax.set_yticklabels(tk_l, fontsize=7)
    ax.tick_params(axis="x", labelsize=7, length=2.5, colors="#444")
    ax.tick_params(axis="y", colors="#444")
    for sp in ax.spines.values(): sp.set_color("#888"); sp.set_linewidth(0.6)
    return im


def plot_eg(utt, probe, family, save):
    color = FAMILY_COLORS[family]; estrf = ESTRF[probe]
    coch = cochlea_np(utt.audio.astype(np.float32))
    n_frames = coch.shape[0]
    ms_per = len(utt.audio)/SR*1000/n_frames
    total_ms = n_frames*ms_per
    coch_db = cochleagram_to_db(coch, 60.0)
    response = apply_estrf(coch, estrf)
    win = int(round(70.0/ms_per))
    thr = np.quantile(response, 0.80)
    peaks = []
    for i in range(n_frames):
        lo=max(0,i-win); hi=min(n_frames,i+win+1)
        if response[i]>=thr and response[i]==response[lo:hi].max(): peaks.append(i)
    probe_frames = []
    for ps in utt.phone_segs:
        if ps.phone_39 != probe: continue
        f0=int(round(ps.start_sample/SR*1000/ms_per))
        f1=int(round(ps.end_sample  /SR*1000/ms_per))
        probe_frames.append((f0,f1))

    fig = plt.figure(figsize=(14.0, 8.6))
    outer = gridspec.GridSpec(4,1, height_ratios=[1.45,1.55,0.34,1.20], hspace=0.30,
        left=0.085, right=0.975, top=0.905, bottom=0.075, figure=fig)

    top = gridspec.GridSpecFromSubplotSpec(1,2, subplot_spec=outer[0],
                                            width_ratios=[0.46,0.54], wspace=0.20)
    ax_k = fig.add_subplot(top[0])
    vmaxk = float(np.max(np.abs(estrf))) + 1e-12
    im_k = _draw_kernel(ax_k, estrf, vmaxk)
    cb = fig.colorbar(im_k, ax=ax_k, fraction=0.045, pad=0.02)
    cb.set_label("eSTRF weight (signed)", fontsize=8); cb.ax.tick_params(labelsize=7)
    ax_k.set_title(f"Effective STRF for /{probe}/   ·   {family}",
                   fontsize=11, fontweight="bold", color=color, pad=8)

    ax_c = fig.add_subplot(top[1]); ax_c.axis("off")
    n_inst = sum(1 for ps in utt.phone_segs if ps.phone_39==probe)
    ax_c.add_patch(plt.Rectangle((0,0.86),1.0,0.13,transform=ax_c.transAxes,
                   facecolor=color,edgecolor="none",clip_on=False))
    ax_c.text(0.018,0.925, family.upper(), ha="left",va="center",color="white",
              fontsize=11,fontweight="bold",transform=ax_c.transAxes)
    ax_c.text(0.985,0.925, f"probe   /{probe}/", ha="right",va="center",color="white",
              fontsize=11,fontweight="bold",transform=ax_c.transAxes)
    rows = [("Utterance", utt.utt_id),
            ("Duration", f"{len(utt.audio)/SR*1000:.0f} ms   ·   {n_frames} frames"),
            (f"Instances of /{probe}/", str(n_inst)),
            ("Response peaks (≥ p80)", str(len(peaks)))]
    meta_y, line_h = 0.74, 0.115
    for k,(lbl,val) in enumerate(rows):
        y = meta_y - k*line_h
        ax_c.text(0.020,y,lbl,ha="left",va="center",fontsize=9.5,color="#666",transform=ax_c.transAxes)
        ax_c.text(0.34, y,val,ha="left",va="center",fontsize=9.5,color="#222",
                  fontweight="bold",transform=ax_c.transAxes)
    if utt.transcript:
        words = utt.transcript.strip().rstrip(".")
        wrapped = words if len(words)<80 else words[:77]+"…"
        ax_c.text(0.020,0.13,f"“{wrapped}”",ha="left",va="center",
                  fontsize=10,color="#444",style="italic",transform=ax_c.transAxes)

    ax_co = fig.add_subplot(outer[1])
    ax_co.imshow(coch_db.T, origin="lower",aspect="auto",cmap="magma",
                 extent=[0,total_ms,-0.5,N_FREQ-0.5], vmin=-60,vmax=0,interpolation="bilinear")
    target=[200,500,1000,2000,5000]; tk_b,tk_l=[],[]
    for hz in target:
        if COCHLEA_CF[0]<=hz<=COCHLEA_CF[-1]:
            i=int(np.argmin(np.abs(COCHLEA_CF-hz))); tk_b.append(i); tk_l.append(_fmt_hz(hz))
    ax_co.set_yticks(tk_b); ax_co.set_yticklabels(tk_l)
    ax_co.set_ylim(-0.5,N_FREQ-0.5); ax_co.set_ylabel("Frequency (Hz)")
    ax_co.set_xlim(0,total_ms); ax_co.set_xticklabels([])
    ax_co.tick_params(axis="y",colors="#444")
    for sp in ax_co.spines.values(): sp.set_color("#888"); sp.set_linewidth(0.6)
    for f0,f1 in probe_frames:
        ax_co.add_patch(plt.Rectangle((f0*ms_per, N_FREQ-4),(f1-f0)*ms_per,4,
                        facecolor=color,edgecolor="none",alpha=0.85,clip_on=True))
    for f in peaks: ax_co.axvline(f*ms_per,color="white",lw=0.5,alpha=0.30)
    ax_co.text(0.005,0.965,"Auditory spectrogram   (model wav2aud, dB)",
               transform=ax_co.transAxes, ha="left",va="top",fontsize=8.5,
               color="white",alpha=0.95,
               bbox=dict(boxstyle="round,pad=0.22",fc="black",ec="none",alpha=0.50))

    ax_s = fig.add_subplot(outer[2])
    _draw_phone_strip(ax_s, utt.phone_segs, total_ms); ax_s.set_xticklabels([])

    ax_r = fig.add_subplot(outer[3])
    t_axis = np.arange(n_frames)*ms_per
    r = response - np.median(response)
    rmin,rmax = float(r.min()), float(r.max())
    pad_y = 0.10*max(rmax-rmin, 1e-6)
    ax_r.set_ylim(rmin-pad_y, rmax+pad_y*2)
    for f0,f1 in probe_frames:
        ax_r.axvspan(f0*ms_per, f1*ms_per, facecolor=color,alpha=0.10,edgecolor="none",zorder=0)
    ax_r.fill_between(t_axis,0,r,where=(r>0),color=color,alpha=0.55,linewidth=0,zorder=2)
    ax_r.fill_between(t_axis,0,r,where=(r<0),color=color,alpha=0.18,linewidth=0,zorder=2)
    ax_r.plot(t_axis,r,color=color,linewidth=1.4,zorder=3)
    ax_r.axhline(0,color="#555",lw=0.7,alpha=0.65)
    for f in peaks: ax_r.axvline(f*ms_per,color=color,lw=0.5,alpha=0.55,ls=":",zorder=1)
    ax_r.set_xlim(0,total_ms); ax_r.set_xlabel("Time (ms)")
    ax_r.set_ylabel(f"r(t)  for  /{probe}/", color="#222")
    ax_r.tick_params(axis="y",colors="#444")
    for sp in ax_r.spines.values(): sp.set_color("#888"); sp.set_linewidth(0.6)
    ax_r.text(0.005,0.96,
              f"shaded = ground-truth /{probe}/ intervals   ·   dotted lines = response peaks",
              transform=ax_r.transAxes,ha="left",va="top",fontsize=7.5,color="#444",
              bbox=dict(boxstyle="round,pad=0.22",fc="white",ec="#ccc",lw=0.5,alpha=0.85))
    fig.suptitle("eSTRF response along a TIMIT utterance   ·   trained biomimetic frontend",
                 y=0.975, fontsize=12.5, color="#222", fontweight="bold")
    fig.savefig(save, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return n_frames, peaks, response, probe_frames

out_dir = Path("/Users/eminent/Projects/Cortical_Front/r_code/analysis_outputs2")
out_dir.mkdir(exist_ok=True)
print("\nRendering test figures (polished, mean-subtracted eSTRF)...")
print(f"{'family':>16s}  {'probe':>6s}  {'utt_id':>22s}  "
      f"{'n_pks':>5s}  {'#gt-frames':>10s}  {'pks-on-gt':>9s}  {'precision':>9s}  {'recall':>7s}")
print("-"*120)
for fam, probes in [("Vowels",["iy","aa"]),("Stops",["t"]),("Fricatives",["s","sh","f"]),
                    ("Nasals",["n","m"]),("Liquids/Glides",["r","l","w"])]:
    for probe in probes:
        if probe not in ESTRF: continue
        utts = pick_utterances(probe, n_picks=1)
        if not utts: continue
        u = utts[0]
        out = out_dir / f"_test2_estrf_{probe}_{u.utt_id}.png"
        nf, pks, resp, gt = plot_eg(u, probe, fam, str(out))
        # Quantify alignment: how many peaks land inside (or within 50 ms of) a ground-truth interval?
        gt_set = set()
        for f0,f1 in gt:
            for i in range(max(0,f0-10), min(nf,f1+10)): gt_set.add(i)
        pks_on_gt = sum(1 for p in pks if p in gt_set)
        # And: within the gt frames, what fraction of them is part of a high-response patch (top 30% of r)?
        all_gt = set()
        for f0,f1 in gt:
            for i in range(f0, f1): all_gt.add(i)
        if all_gt:
            r_thresh = np.quantile(resp, 0.70)
            high_r = set(np.where(resp >= r_thresh)[0])
            recall = len(all_gt & high_r) / len(all_gt)
        else:
            recall = float("nan")
        prec = pks_on_gt/max(1,len(pks))
        print(f"{fam:>16s}  /{probe:>4s}/  {u.utt_id:>22s}  {len(pks):>5d}  {len(all_gt):>10d}  "
              f"{pks_on_gt:>9d}  {prec:>9.2f}  {recall:>7.2f}")
print("\nDONE.")
