# -*- coding: utf-8 -*-
"""Just Dance bonus level: fixed player zones plus ByteTrack identity locking."""
import os
import sys
import threading
import time
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO
from pose_utils import MainDancerTracker, PoseEstimator

MODEL_PATH = Path(__file__).resolve().with_name("yolov8n-pose.pt")
NUM_SIGNATURES, BONUS_POINTS = 12, 50
SIGNATURE_WINDOW, SIGNATURE_THRESHOLD = 1.5, .05
SKELETON = ((5,7),(7,9),(5,6),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16))
COLORS = ((0,235,130),(0,175,255),(255,180,0),(235,80,235))

def draw_skeleton(img, points, color, thick=3):
    if points is None: return
    for a, b in SKELETON:
        if points[a,2] > .3 and points[b,2] > .3:
            cv2.line(img, tuple(points[a,:2].astype(int)), tuple(points[b,:2].astype(int)), color, thick, cv2.LINE_AA)
    for x, y, confidence in points:
        if confidence > .3: cv2.circle(img, (int(x),int(y)), thick + 2, color, -1, cv2.LINE_AA)

def fit_pose(points, width, height, origin=(0,0), margin=42):
    valid = points[points[:,2] > .3, :2]
    if len(valid) < 4: return None
    low, high = valid.min(axis=0), valid.max(axis=0)
    scale = min((width-2*margin)/max(high[0]-low[0],1), (height-2*margin)/max(high[1]-low[1],1), 3.0)
    result = points.copy()
    result[:,0] = points[:,0]*scale + (width-(high[0]-low[0])*scale)/2-low[0]*scale+origin[0]
    result[:,1] = points[:,1]*scale + (height-(high[1]-low[1])*scale)/2-low[1]*scale+origin[1]
    return result

def pose_distance(person, signature, person_shape, signature_shape):
    if person is None or signature is None: return 1.0
    use = (person[:,2] > .3) & (signature[:,2] > .3)
    if not use.any(): return 1.0
    a = person[:,:2] / np.array([person_shape[1], person_shape[0]])
    b = signature[:,:2] / np.array([signature_shape[1], signature_shape[0]])
    return np.linalg.norm(a[use]-b[use])

class DanceScorer:
    def __init__(self, root):
        self.root = root; root.title("Just Dance  Bonus Level"); root.geometry("1400x820"); root.minsize(1000,650)
        self.num_players, self.scores = 1, [0]
        self.running = self.players_ready = False
        self.video_path = self.cap_ref = None; self.ref_fps = 30; self.ref_idx = 0
        self.signatures, self.completed = [], set()
        # A player keeps this assigned tracking ID throughout the game.
        self.track_to_player, self.last_pose, self.missing_frames = {}, [None], [0]
        self.current_pose = [None]; self.status = "Set player count, then stand in a labelled zone."
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened(): messagebox.showerror("Camera", "Cannot open camera."); sys.exit(1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,640); self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        self.audio_path = None
        self.build_ui()
        if not MODEL_PATH.exists():
            messagebox.showerror("Missing model", f"Put yolov8n-pose.pt beside this file:\n{MODEL_PATH}"); self.close(); return
        try:
            self.estimator = PoseEstimator(model_path=MODEL_PATH, confidence=.3, image_size=384)
            self.model = YOLO(str(MODEL_PATH), verbose=False)  # reference signature extraction
            self.main_tracker = MainDancerTracker()
        except Exception as error: messagebox.showerror("Model error", str(error)); self.close(); return
        self.audio_enabled = True; self.last_audio_time = 0.0
        root.protocol("WM_DELETE_WINDOW", self.close); self.tick()

    def build_ui(self):
        body = tk.Frame(self.root,bg="#15171c"); body.pack(fill=tk.BOTH,expand=True)
        self.stage = tk.Label(body,bg="black"); self.stage.place(relx=0,rely=0,relwidth=.60,relheight=.82)
        self.avatars = tk.Label(body,bg="#101116"); self.avatars.place(relx=.60,rely=0,relwidth=.40,relheight=.82)
        self.score_label = tk.Label(body,bg="#101116",fg="white",font=("Arial",16,"bold"),justify=tk.LEFT)
        self.score_label.place(relx=0,rely=.82,relwidth=1,relheight=.18)
        controls = tk.Frame(self.root,bg="#eeeeee"); controls.place(x=12,y=12)
        tk.Button(controls,text="Set Players",command=self.set_players).pack(side=tk.LEFT,padx=4,pady=4)
        self.video_button=tk.Button(controls,text="Select Dance Video",command=self.select_video,state=tk.DISABLED); self.video_button.pack(side=tk.LEFT,padx=4,pady=4)
        self.start_button=tk.Button(controls,text="Start Bonus",command=self.start_game,state=tk.DISABLED); self.start_button.pack(side=tk.LEFT,padx=4,pady=4)
        self.audio_var=tk.BooleanVar(value=True)
        tk.Checkbutton(controls,text="Audio",variable=self.audio_var,command=self.toggle_audio,bg="#eeeeee",activebackground="#eeeeee").pack(side=tk.LEFT,padx=6)
        tk.Button(controls,text="Select Music",command=self.select_audio).pack(side=tk.LEFT,padx=4,pady=4)
        self.status_label=tk.Label(self.root,bg="#15171c",fg="white",font=("Arial",11)); self.status_label.place(relx=.5,rely=.985,anchor=tk.S)

    def set_players(self):
        n=simpledialog.askinteger("Players","How many players? (1 to 4)",minvalue=1,maxvalue=4)
        if n:
            self.num_players=n; self.scores=[0]*n; self.current_pose=[None]*n; self.last_pose=[None]*n; self.missing_frames=[0]*n
            self.track_to_player.clear(); self.players_ready=False; self.status=f"Waiting for {n} player(s): stand in zones P1 P{n}."

    def toggle_audio(self):
        self.audio_enabled=bool(self.audio_var.get())
        if not self.audio_enabled: self.stop_music()

    def select_audio(self):
        """Select optional WAV music; winsound keeps deployment dependency-free on Windows."""
        path=filedialog.askopenfilename(filetypes=[("WAV audio","*.wav"),("All files","*.*")])
        if path: self.audio_path=path; self.status=f"Music loaded: {os.path.basename(path)}"

    def start_music(self):
        if not self.audio_enabled or not self.audio_path: return
        try:
            import winsound
            winsound.PlaySound(self.audio_path,winsound.SND_FILENAME|winsound.SND_ASYNC|winsound.SND_LOOP)
        except (ImportError,RuntimeError): self.status="Music could not be played; score cues remain available."

    @staticmethod
    def stop_music():
        try:
            import winsound
            winsound.PlaySound(None,winsound.SND_PURGE)
        except (ImportError,RuntimeError): pass

    def play_audio(self,event="hit"):
        """Play a short non-blocking cue without adding a mandatory audio dependency."""
        if not self.audio_enabled or time.monotonic()-self.last_audio_time<.08: return
        self.last_audio_time=time.monotonic(); frequency,duration={"start":(660,100),"hit":(880,90),"finish":(520,180)}.get(event,(880,90))
        def beep():
            try:
                import winsound
                winsound.Beep(frequency,duration)
            except (ImportError,RuntimeError): pass
        threading.Thread(target=beep,daemon=True).start()

    def detect_and_assign(self, frame):
        """Use MainDancerTracker so a passer-by cannot steal the active stickman."""
        detected,_latency=self.estimator.infer(frame)
        poses=[None]*self.num_players; seen=set(); h,w=frame.shape[:2]
        if self.num_players == 1:
            selected=self.main_tracker.select(detected,timestamp=time.monotonic())
            if selected is not None:
                poses[0]=np.column_stack((selected.points[:,0]*w,selected.points[:,1]*h,selected.confidence)); seen.add(0)
        else:
            # Multiplayer keeps the original fixed-zone contract. MainDancerTracker is an
            # identity lock for the single main-player mode, while zones prevent cross-control.
            for candidate in detected:
                player=min(self.num_players-1,int(candidate.center[0]*self.num_players))
                if poses[player] is None:
                    poses[player]=np.column_stack((candidate.points[:,0]*w,candidate.points[:,1]*h,candidate.confidence)); seen.add(player)
        for player in range(self.num_players):
            if poses[player] is not None:
                self.last_pose[player]=poses[player]; self.missing_frames[player]=0
            else:
                self.missing_frames[player]+=1
                # Match the tracker's one-second occlusion grace and keep the avatar stable.
                if self.missing_frames[player] <= 30: poses[player]=self.last_pose[player]
        self.current_pose=poses
        return seen

    def draw_zones(self, frame):
        h,w=frame.shape[:2]
        for player in range(self.num_players):
            x=int(player*w/self.num_players); x2=int((player+1)*w/self.num_players)
            cv2.rectangle(frame,(x,0),(x2-1,h),(90,90,90),2)
            cv2.putText(frame,f"P{player+1} ZONE",(x+12,58),cv2.FONT_HERSHEY_DUPLEX,.65,COLORS[player],2,cv2.LINE_AA)

    def select_video(self):
        path=filedialog.askopenfilename(filetypes=[("Video","*.mp4 *.avi *.mov *.mkv"),("All files","*.*")])
        if path: self.video_path=path; self.extract_signatures(); self.start_button.config(state=tk.NORMAL); self.status=f"Loaded {os.path.basename(path)}."

    def extract_signatures(self):
        cap=cv2.VideoCapture(self.video_path); total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); self.signatures=[]
        for index in range(0,total,max(1,total//NUM_SIGNATURES)):
            cap.set(cv2.CAP_PROP_POS_FRAMES,index); ok,frame=cap.read()
            if not ok: break
            result=self.model(frame,conf=.3,verbose=False)[0]; points=None
            if result.keypoints is not None and len(result.keypoints.xy): points=np.column_stack((result.keypoints.xy.cpu().numpy()[0],result.keypoints.conf.cpu().numpy()[0]))
            self.signatures.append((cap.get(cv2.CAP_PROP_POS_MSEC),points,frame.shape[:2]))
        cap.release(); self.completed.clear()

    def start_game(self):
        if not self.video_path: return
        self.cap_ref=cv2.VideoCapture(self.video_path); self.ref_fps=self.cap_ref.get(cv2.CAP_PROP_FPS) or 30; self.ref_idx=0; self.running=True; self.completed.clear()
        self.main_tracker.reset(); self.start_music(); self.play_audio("start"); self.status="Bonus started  main dancer lock is warming up."

    def avatar_frame(self):
        w,h=max(self.avatars.winfo_width(),320),max(self.avatars.winfo_height(),400); image=np.full((h,w,3),(22,24,31),np.uint8)
        cols=1 if self.num_players==1 else 2; rows=1 if self.num_players==1 else 2; cw,ch=w//cols,h//rows
        for player in range(self.num_players):
            row,col=divmod(player,cols); x,y=col*cw,row*ch
            if cols==2: cv2.rectangle(image,(x+4,y+4),(x+cw-4,y+ch-4),(70,70,80),1)
            cv2.putText(image,f"P{player+1}",(x+18,y+38),cv2.FONT_HERSHEY_DUPLEX,.8,COLORS[player],2)
            if self.current_pose[player] is None: cv2.putText(image,"waiting",(x+18,y+ch//2),cv2.FONT_HERSHEY_SIMPLEX,.6,(140,140,140),1)
            else: draw_skeleton(image,fit_pose(self.current_pose[player],cw,ch,(x,y)),COLORS[player],6)
        return image

    def show(self,label,frame):
        w,h=max(label.winfo_width(),1),max(label.winfo_height(),1); rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        photo=ImageTk.PhotoImage(Image.fromarray(rgb).resize((w,h))); label.configure(image=photo); label.image=photo

    def tick(self):
        try:
            ok,frame=self.cap.read()
            if ok:
                frame=cv2.flip(frame,1); seen=self.detect_and_assign(frame)
                if not self.running:
                    self.draw_zones(frame)
                    for player,points in enumerate(self.current_pose): draw_skeleton(frame,points,COLORS[player])
                    self.show(self.stage,frame)
                    ready=len(seen)>=self.num_players
                    if ready!=self.players_ready: self.players_ready=ready; self.video_button.config(state=tk.NORMAL if ready else tk.DISABLED)
            if self.running and self.cap_ref:
                ok,reference=self.cap_ref.read()
                if ok:
                    self.ref_idx+=1; self.show(self.stage,reference); now=self.ref_idx/self.ref_fps*1000
                    for index,(timestamp,signature,shape) in enumerate(self.signatures):
                        if index in self.completed or abs(now-timestamp)>SIGNATURE_WINDOW*1000: continue
                        for player,points in enumerate(self.current_pose):
                            if pose_distance(points,signature,frame.shape[:2],shape)<SIGNATURE_THRESHOLD: self.scores[player]+=BONUS_POINTS; self.completed.add(index); self.play_audio("hit"); self.status=f"P{player+1}: +{BONUS_POINTS} bonus!"
                else: self.running=False; self.cap_ref.release(); self.cap_ref=None; self.stop_music(); self.play_audio("finish"); self.status="Dance video ended."
            self.show(self.avatars,self.avatar_frame()); self.score_label.config(text="SCORES\n"+"   ".join(f"P{i+1}: {score}" for i,score in enumerate(self.scores))); self.status_label.config(text=self.status)
        except Exception as error: self.status=f"Live detection error: {type(error).__name__}: {error}"
        self.root.after(33,self.tick)

    def close(self):
        self.stop_music()
        if hasattr(self,"cap") and self.cap.isOpened(): self.cap.release()
        if self.cap_ref: self.cap_ref.release()
        self.root.destroy()

if __name__ == "__main__":
    root=tk.Tk(); DanceScorer(root); root.mainloop()
