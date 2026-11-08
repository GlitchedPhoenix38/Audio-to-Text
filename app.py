"""FoziScribe Local: private, accuracy-first Whisper transcription."""
from __future__ import annotations
import json, os, re, threading, time
from datetime import datetime
from pathlib import Path
import gradio as gr
from faster_whisper import WhisperModel
from utils import build_srt, format_duration, format_timestamp, format_speed_ratio

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"output"; OUT.mkdir(exist_ok=True)
HISTORY=OUT/"history.json"
PRESETS={"Fast":"small","Balanced":"large-v3-turbo","Accuracy (recommended)":"large-v3"}
MODELS=["Follow preset","large-v3","large-v3-turbo","medium","small","distil-large-v3"]
LANGS={"Auto-detect":None,"English":"en","Hindi":"hi","Urdu":"ur","Arabic":"ar","Marathi":"mr","Bengali":"bn","Tamil":"ta","Telugu":"te","French":"fr","German":"de","Spanish":"es","Portuguese":"pt","Russian":"ru","Japanese":"ja","Korean":"ko","Chinese":"zh","Turkish":"tr","Indonesian":"id"}
EXTS={".mp3",".wav",".m4a",".flac",".ogg",".webm",".opus",".aac",".wma",".mp4",".mkv",".mov",".m4b",".aiff",".oga"}
CACHE={}; MODEL_LOCK=threading.Lock(); JOB_LOCK=threading.Lock()

def get_history():
    try: return json.loads(HISTORY.read_text(encoding="utf-8"))
    except (OSError,json.JSONDecodeError): return []

def save_history(items):
    HISTORY.write_text(json.dumps(items[-50:],ensure_ascii=False,indent=2),encoding="utf-8")

def load_model(name,progress):
    if name in CACHE: return CACHE[name]
    errors=[]
    with MODEL_LOCK:
        if name in CACHE: return CACHE[name]
        for device,precision in [("cuda","float16"),("cuda","int8_float16"),("cuda","int8"),("cpu","int8")]:
            try:
                progress(.03,desc=f"Loading {name} · {device}/{precision}")
                model=WhisperModel(name,device=device,compute_type=precision,
                    cpu_threads=max(4,(os.cpu_count() or 8)-2) if device=="cpu" else 0,num_workers=1)
                CACHE[name]=(model,device,precision); return CACHE[name]
            except Exception as e: errors.append(f"{device}/{precision}: {e}")
    raise RuntimeError("Unable to load model. "+" | ".join(errors[-2:]))

def duration_of(path):
    try:
        import av
        with av.open(str(path)) as c:
            s=next((x for x in c.streams if x.type=="audio"),None)
            if s and s.duration and s.time_base: return float(s.duration*s.time_base)
            if c.duration: return float(c.duration/1_000_000)
    except Exception: pass
    return None

def stamp(seconds):
    ms=max(0,round(seconds*1000)); h,r=divmod(ms,3600000); m,r=divmod(r,60000); s,ms=divmod(r,1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def make_vtt(entries):
    return "WEBVTT\n\n"+"\n\n".join(f"{stamp(a).replace(',', '.')} --> {stamp(b).replace(',', '.')}\n{t}" for a,b,t in entries)+"\n"

def transcribe(path,preset,model_override,language,task,timestamps,word_ts,beam,vad,progress=gr.Progress()):
    if not path: raise gr.Error("Upload an audio or video file first.")
    p=Path(str(path))
    if p.suffix.lower() not in EXTS: raise gr.Error("Unsupported format. Use MP3, WAV, M4A, FLAC, OGG, WebM, MP4 or MOV.")
    model_name=PRESETS.get(preset,PRESETS["Accuracy (recommended)"])
    if model_override!="Follow preset": model_name=model_override
    lang=LANGS.get(language); duration=duration_of(p)
    with JOB_LOCK:
        try: model,device,precision=load_model(model_name,progress)
        except Exception as e: raise gr.Error(f"Model load failed: {e}\nFirst launch needs internet to download model weights.")
        start=time.perf_counter(); lines=[]; entries=[]
        try:
            segments,info=model.transcribe(str(p),task="translate" if task=="Translate to English" else "transcribe",
                language=lang,beam_size=int(beam),best_of=5,patience=1.0,condition_on_previous_text=True,
                vad_filter=bool(vad),word_timestamps=bool(word_ts),temperature=0.0,
                compression_ratio_threshold=2.4,log_prob_threshold=-1.0,no_speech_threshold=0.6)
            for seg in segments:
                text=(seg.text or "").strip()
                if not text: continue
                entries.append((float(seg.start),float(seg.end),text))
                lines.append(f"[{format_timestamp(seg.start)[1:-1]}] {text}" if timestamps else text)
                if duration:
                    frac=min(.99,max(.02,seg.end/duration)); elapsed=time.perf_counter()-start
                    progress(frac,desc=f"Transcribing {format_timestamp(seg.end)} · ETA {format_duration(elapsed/frac*(1-frac))}")
        except Exception as e: raise gr.Error(f"Transcription failed: {e}\nTry Balanced mode if GPU memory is limited.")
        elapsed=time.perf_counter()-start
    transcript="\n".join(lines) or "No speech detected."
    detected=getattr(info,"language","unknown"); prob=getattr(info,"language_probability",None)
    language_text=detected.upper()+(f" · detection probability {prob:.0%}" if prob is not None else "")
    tag=datetime.now().strftime("%Y%m%d-%H%M%S"); base=re.sub(r"[^A-Za-z0-9_-]+","_",p.stem)[:40] or "transcript"
    txt=OUT/f"{base}-{tag}.txt"; srt=OUT/f"{base}-{tag}.srt"; vtt=OUT/f"{base}-{tag}.vtt"
    txt.write_text(transcript,encoding="utf-8"); srt.write_text(build_srt(entries),encoding="utf-8"); vtt.write_text(make_vtt(entries),encoding="utf-8")
    hist=get_history(); hist.append({"created":datetime.now().isoformat(timespec="seconds"),"name":p.name,"model":model_name,"language":detected,"transcript":transcript,"txt":str(txt),"srt":str(srt),"vtt":str(vtt)}); save_history(hist)
    status=f"MODEL     {model_name}\nDEVICE    {device.upper()} · {precision}\nLANGUAGE  {language_text}\nDURATION  {format_timestamp(duration) if duration else 'unknown'}\nTIME      {format_duration(elapsed)}\nSPEED     {format_speed_ratio(duration,elapsed)}\nSEGMENTS  {len(entries)}"
    choices=[f"{x['created']} · {x['name']}" for x in reversed(hist[-50:])]
    progress(1,desc="Transcript ready")
    return transcript,language_text,status,str(txt),str(srt),str(vtt),gr.update(choices=choices,value=choices[0] if choices else None)

def restore(choice):
    for x in reversed(get_history()):
        if f"{x['created']} · {x['name']} " .strip()==choice:
            return x["transcript"],x["language"].upper(),f"MODEL {x['model']}\nSAVED {x['created']}",x["txt"],x["srt"],x["vtt"]
    return "","","",None,None,None

def save_edited(text):
    if not text.strip(): raise gr.Error("Transcript is empty.")
    p=OUT/f"edited-{datetime.now().strftime('%Y%m%d-%H%M%S')}.txt"; p.write_text(text,encoding="utf-8")
    return str(p),f"Saved {p.name}"

def find_text(text,query):
    if not query.strip(): return "Enter a search term."
    hits=[x for x in text.splitlines() if query.casefold() in x.casefold()]
    return "\n".join(hits[:100]) if hits else "No matches."

CSS="#hero{padding:26px;border:1px solid var(--border-color-primary);border-radius:18px;background:var(--background-fill-secondary)}#hero h1{font-size:2rem;letter-spacing:-.04em}#transcript textarea{font-size:15px!important;line-height:1.7!important}footer{display:none!important}"
with gr.Blocks(title="FoziScribe Local",theme=gr.themes.Soft(primary_hue="indigo",neutral_hue="slate"),css=CSS) as demo:
    gr.HTML('<div id="hero"><small>FOZISCRIBE LOCAL · PRIVATE TRANSCRIPTION</small><h1>Turn audio into words.</h1><p>Whisper AI running on your own computer. Your recordings stay local.</p></div>')
    with gr.Row():
        with gr.Column(scale=5):
            gr.Markdown("### 1 · Upload recording")
            upload=gr.File(label="Drop audio or video here",file_count="single",type="filepath",file_types=[".mp3",".wav",".m4a",".flac",".ogg",".webm",".opus",".aac",".mp4",".mkv",".mov"])
            gr.Markdown("### 2 · Transcription settings")
            preset=gr.Radio(["Fast","Balanced","Accuracy (recommended)"],value="Accuracy (recommended)",label="Mode",info="Accuracy: full large-v3 · Balanced: large-v3-turbo · Fast: small")
            model=gr.Dropdown(MODELS,value="Follow preset",label="Model override")
            language=gr.Dropdown(list(LANGS),value="Auto-detect",label="Audio language")
            task=gr.Radio(["Transcribe in original language","Translate to English"],value="Transcribe in original language",label="Output")
            with gr.Accordion("Advanced accuracy settings",open=False):
                timestamps=gr.Checkbox(value=True,label="Segment timestamps")
                word_ts=gr.Checkbox(value=False,label="Word-level timestamps")
                beam=gr.Slider(1,10,value=5,step=1,label="Beam size")
                vad=gr.Checkbox(value=False,label="Voice activity detection (VAD)",info="Leave off if quiet speech matters.")
            run=gr.Button("Transcribe recording",variant="primary",size="lg")
            status=gr.Textbox(label="Job details",lines=7,interactive=False)
        with gr.Column(scale=7):
            gr.Markdown("### 3 · Transcript")
            detected=gr.Textbox(label="Detected language",interactive=False)
            transcript=gr.Textbox(label="Transcript · editable",lines=20,interactive=True,elem_id="transcript")
            with gr.Row():
                txt=gr.File(label="TXT"); srt=gr.File(label="SRT"); vtt=gr.File(label="WebVTT")
            with gr.Row():
                save=gr.Button("Save edited transcript"); edited=gr.File(label="Edited TXT")
            save_info=gr.Markdown()
    history=gr.Dropdown(label="Local transcript history",choices=[f"{x['created']} · {x['name']}" for x in reversed(get_history())])
    restore_btn=gr.Button("Open selected transcript")
    with gr.Accordion("Search inside transcript",open=False):
        with gr.Row():
            query=gr.Textbox(label="Search phrase"); search=gr.Button("Find")
        matches=gr.Textbox(label="Matching lines",lines=5,interactive=False)
    gr.Markdown("Audio is processed on this computer. Transcripts and exports are saved in output/. No account, API key, subscription or cloud transcription service is used.")
    run.click(transcribe,[upload,preset,model,language,task,timestamps,word_ts,beam,vad],[transcript,detected,status,txt,srt,vtt,history])
    restore_btn.click(restore,[history],[transcript,detected,status,txt,srt,vtt])
    save.click(save_edited,[transcript],[edited,save_info])
    search.click(find_text,[transcript,query],[matches])

if __name__=="__main__":
    print("FoziScribe Local: http://127.0.0.1:7860")
    demo.queue(default_concurrency_limit=1,max_size=10)
    demo.launch(server_name="127.0.0.1",server_port=7860,inbrowser=True,show_error=True)
