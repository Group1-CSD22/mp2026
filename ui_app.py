"""
FINAL PRODUCT UI: Intent-Aware Work Monitoring System
Features: Supervised Mode, Smart Toasts, Embedded Analytics
"""

import customtkinter as ctk
import threading
import time
from datetime import datetime
import json
import os
import glob
from pathlib import Path
from complete_live_monitor import CompleteLiveMonitor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Modern Styling
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class WorkMonitorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Intent-Aware Monitor Dashboard v3.0")
        self.geometry("1000x700")
        self.resizable(False, False)

        # Backend Setup
        self.monitor = CompleteLiveMonitor(user_id="user_hari")
        self.monitor.ui_mode = True # Tell backend to not freeze terminal
        self.monitor_thread = None
        self.ui_update_job = None
        self.timer_job = None

        # UI State
        self.timer_seconds = 0
        self.is_timer_running = False
        self.is_timer_paused = False  # NEW: Pauses the UI clock
        self.consecutive_states = {"state": None, "count": 0}
        self.chart_canvas = None

        self.last_window_count = 0

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.setup_ui()
        self.update_dashboard_stats()

    def setup_ui(self):
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # --- SIDEBAR ---
        self.sidebar_frame = ctk.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(5, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="AI Monitor", font=ctk.CTkFont(size=24, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(30, 20))

        self.btn_dashboard = ctk.CTkButton(self.sidebar_frame, text="Dashboard", command=lambda: self.select_frame("dashboard"))
        self.btn_dashboard.grid(row=1, column=0, padx=20, pady=10)

        self.btn_monitor = ctk.CTkButton(self.sidebar_frame, text="Live Monitor", command=lambda: self.select_frame("monitor"))
        self.btn_monitor.grid(row=2, column=0, padx=20, pady=10)

        self.btn_results = ctk.CTkButton(self.sidebar_frame, text="Analytics", command=lambda: self.select_frame("results"))
        self.btn_results.grid(row=3, column=0, padx=20, pady=10)

        # Supervised Mode Toggle
        self.switch_supervised = ctk.CTkSwitch(self.sidebar_frame, text="Supervised Mode", command=self.toggle_supervised)
        self.switch_supervised.grid(row=4, column=0, padx=20, pady=30)

        # --- MAIN CONTENT ---
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)

        self.frames = {}
        self.setup_dashboard_frame()
        self.setup_monitor_frame()
        self.setup_results_frame()

        self.select_frame("dashboard")

    # ================= UI LAYOUTS =================

    def setup_dashboard_frame(self):
        frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.frames["dashboard"] = frame

        # Center the column
        frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(frame, text="System Overview", font=ctk.CTkFont(size=28, weight="bold")).grid(row=0, column=0,
                                                                                                   pady=(0, 20),
                                                                                                   sticky="w")

        # --- CENTERED PROFILE CARD ---
        profile_card = ctk.CTkFrame(frame, corner_radius=15, fg_color="#2B2B2B")
        profile_card.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)

        ctk.CTkLabel(profile_card, text="👤 Global User Profile", font=ctk.CTkFont(size=20, weight="bold")).pack(
            pady=(20, 15))
        self.lbl_profile_sessions = ctk.CTkLabel(profile_card, text="Total Recorded Sessions: 0",
                                                 font=ctk.CTkFont(size=16))
        self.lbl_profile_sessions.pack(pady=5)
        self.lbl_profile_time = ctk.CTkLabel(profile_card, text="Total Evaluated Windows: 0", font=ctk.CTkFont(size=16))
        self.lbl_profile_time.pack(pady=5)

        self.lbl_productivity = ctk.CTkLabel(profile_card, text="System Confidence: 0%",
                                             font=ctk.CTkFont(size=18, weight="bold"), text_color="#3498DB")
        self.lbl_productivity.pack(pady=(15, 20))

        # --- STATIC SYSTEM INFO CARD ---
        info_card = ctk.CTkFrame(frame, corner_radius=15, fg_color="#1E3A5F")
        info_card.grid(row=2, column=0, sticky="nsew", padx=20, pady=20)

        ctk.CTkLabel(info_card, text="ℹ️ System Status", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=(20, 5))
        ctk.CTkLabel(info_card,
                     text="Hybrid Temporal Attention Network (HTAN) is active.\nModel telemetry is evaluated locally in rolling 60-second windows.",
                     font=ctk.CTkFont(size=14), justify="center").pack(pady=10)

    def setup_monitor_frame(self):
        frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.frames["monitor"] = frame

        # Top bar: Title and Timer
        top_bar = ctk.CTkFrame(frame, fg_color="transparent")
        top_bar.pack(fill="x", pady=10)
        ctk.CTkLabel(top_bar, text="Live Activity Tracker", font=ctk.CTkFont(size=28, weight="bold")).pack(side="left")
        self.lbl_timer = ctk.CTkLabel(top_bar, text="00:00:00", font=ctk.CTkFont(size=35, weight="bold"),
                                      text_color="#2FA572")
        self.lbl_timer.pack(side="right")

        # Controls
        control_frame = ctk.CTkFrame(frame, fg_color="transparent")
        control_frame.pack(pady=10)
        self.btn_start = ctk.CTkButton(control_frame, text="▶ START", command=self.start_monitoring, width=120,
                                       fg_color="#2FA572", hover_color="#1E7A52")
        self.btn_start.pack(side="left", padx=10)
        self.btn_stop = ctk.CTkButton(control_frame, text="■ STOP", command=self.stop_monitoring, width=120,
                                      fg_color="#E03B3B", hover_color="#A82A2A", state="disabled")
        self.btn_stop.pack(side="left", padx=10)

        # Status Card (Clean & Abstract)
        self.status_card = ctk.CTkFrame(frame, corner_radius=15)
        self.status_card.pack(pady=10, fill="x", ipady=10)

        self.lbl_current_status = ctk.CTkLabel(self.status_card, text="System Standby...",
                                               font=ctk.CTkFont(size=14, slant="italic"), text_color="gray")
        self.lbl_current_status.pack(pady=5)
        self.lbl_previous_state = ctk.CTkLabel(self.status_card, text="Latest Sequence: None",
                                               font=ctk.CTkFont(size=22, weight="bold"))
        self.lbl_previous_state.pack(pady=5)
        self.lbl_toast = ctk.CTkLabel(self.status_card, text="", font=ctk.CTkFont(size=14), text_color="#FFA500")
        self.lbl_toast.pack()

        # Supervised Mode Correction Panel
        self.correction_frame = ctk.CTkFrame(frame, fg_color="transparent")
        ctk.CTkLabel(self.correction_frame, text="Supervised Override:", font=ctk.CTkFont(size=14)).pack(side="left",
                                                                                                         padx=10)
        self.state_dropdown = ctk.CTkOptionMenu(self.correction_frame,
                                                values=["Deep Work", "Active Work", "Research", "Communication",
                                                        "Distracted", "Idle"])
        self.state_dropdown.pack(side="left", padx=10)
        self.btn_correct = ctk.CTkButton(self.correction_frame, text="Confirm & Resume", command=self.submit_correction,
                                         fg_color="#F39C12", hover_color="#D68910")
        self.btn_correct.pack(side="left", padx=10)

        # --- NEW: AESTHETIC EVENT LOG ---
        ctk.CTkLabel(frame, text="Rolling Window Output Log (5-Min Context)",
                     font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", pady=(20, 5))
        self.log_textbox = ctk.CTkTextbox(frame, height=150, font=ctk.CTkFont(family="Consolas", size=13))
        self.log_textbox.pack(fill="both", expand=True)
        self.log_textbox.insert("0.0", "Awaiting start signal...\n")
        self.log_textbox.configure(state="disabled")

    def setup_results_frame(self):
        frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.frames["results"] = frame

        ctk.CTkLabel(frame, text="Session Analytics", font=ctk.CTkFont(size=28, weight="bold")).pack(pady=(0, 10), anchor="w")

        # Area for Matplotlib chart
        self.chart_frame = ctk.CTkFrame(frame, corner_radius=10, fg_color="#2B2B2B")
        self.chart_frame.pack(fill="both", expand=True, pady=10)

        ctk.CTkLabel(self.chart_frame, text="Run a session to generate analytics.", text_color="gray").pack(expand=True)

    # ================= LOGIC & ACTIONS =================

    def select_frame(self, frame_name):
        for name, frame in self.frames.items():
            if name == frame_name:
                frame.grid(row=0, column=0, sticky="nsew")
            else:
                frame.grid_forget()

    def toggle_supervised(self):
        """Enable or disable manual correction UI"""
        is_supervised = self.switch_supervised.get() == 1
        self.monitor.supervised_mode = is_supervised

        if is_supervised:
            self.correction_frame.pack(pady=20)
            self.lbl_toast.configure(text="Supervised Mode ON: Will pause after next window.", text_color="#3498DB")
        else:
            self.correction_frame.pack_forget()
            self.lbl_toast.configure(text="")
            # SAFETY CATCH: If user toggles OFF while paused, unpause it!
            if getattr(self.monitor, 'waiting_for_correction', False):
                if hasattr(self.monitor, 'correction_event'):
                    self.monitor.correction_event.set()

    def start_monitoring(self):
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")

        # Update our new split labels
        self.lbl_current_status.configure(text="▶️ Collecting Window #1 (5 min warmup)...", text_color="white")
        self.lbl_previous_state.configure(text="Latest Result: None", text_color="white")

        self.monitor.session_predictions = []
        self.last_window_count = 0

        self.is_timer_running = True
        self.is_timer_paused = False
        self.timer_seconds = 0
        self.update_timer()

        self.monitor_thread = threading.Thread(target=self.monitor.start, daemon=True)
        self.monitor_thread.start()
        self.poll_backend_results()

        self.log_textbox.configure(state="normal")
        self.log_textbox.delete("0.0", "end")
        self.log_textbox.insert("0.0", "Initiating 5-minute contextual warmup sequence...\n")
        self.log_textbox.configure(state="disabled")

    def stop_monitoring(self):
        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")

        if hasattr(self, 'lbl_current_status'):
            self.lbl_current_status.configure(text="⏹️ Monitoring Stopped.", text_color="#E74C3C")

        self.is_timer_running = False

        # Unfreeze the thread if paused
        if hasattr(self.monitor, 'correction_event'):
            self.monitor.correction_event.set()

        self.monitor.stop()

        if self.ui_update_job:
            self.after_cancel(self.ui_update_job)

        if len(self.monitor.session_predictions) > 0:
            self.generate_analytics_chart()

        self.update_dashboard_stats()

    def wait_for_backend_save(self):
        """Smart polling: Waits for the backend thread to completely finish saving the JSON"""
        # If the thread exists and is still alive, wait 1 more second and check again
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.after(1000, self.wait_for_backend_save)
        else:
            # The thread is dead! The JSON is guaranteed to be saved now.
            self.update_dashboard_stats()
            if hasattr(self, 'lbl_toast'):
                self.lbl_toast.configure(text="✓ Session log saved and ready for review.", text_color="#2ECC71")
            if hasattr(self, 'lbl_current_status'):
                self.lbl_current_status.configure(text="⏹️ System Standby.", text_color="gray")

    def refresh_after_save(self):
        """Helper method to update UI only after JSON finishes saving"""
        self.update_dashboard_stats()
        if hasattr(self, 'lbl_toast'):
            self.lbl_toast.configure(text="✓ Session log saved and ready for review.", text_color="#2ECC71")

    def submit_correction(self):
        """Send correction and trigger Phantom Window Filter"""
        import time
        selected_state = self.state_dropdown.get()
        current_windows = len(self.monitor.session_predictions)

        # 1. Send to backend
        self.monitor.apply_ui_correction(selected_state)

        # 2. START THE PHANTOM FILTER TIMER
        self.last_correction_time = time.time()

        self.is_timer_paused = False
        if hasattr(self, 'btn_correct'):
            self.btn_correct.configure(state="disabled", text="Waiting...")

        # 3. Lock the text on the screen in Orange
        if hasattr(self, 'lbl_previous_state'):
            self.lbl_previous_state.configure(text=f"Sequence #{current_windows}: {selected_state} (CORRECTED)",
                                              text_color="#F39C12")

        if hasattr(self, 'log_textbox'):
            self.log_textbox.configure(state="normal")
            self.log_textbox.insert("0.0",
                                    f"   ↳ OVERRIDE: Sequence #{current_windows} locked as {selected_state.upper()}\n")
            self.log_textbox.configure(state="disabled")

        if hasattr(self, 'lbl_toast'):
            self.lbl_toast.configure(text="▶️ Resuming... (Phantom Filter Active)", text_color="#3498DB")

    def poll_backend_results(self):
        if not self.winfo_exists(): return

        if self.is_timer_running:
            current_windows = len(self.monitor.session_predictions)

            if current_windows > self.last_window_count:
                import time

                # --- THE PHANTOM EATER ---
                # If a window arrives suspiciously fast (under 5 seconds) after unpausing,
                # it is a buffered phantom window. We eat it and ignore it!
                if hasattr(self, 'last_correction_time') and (time.time() - self.last_correction_time) < 5:
                    self.last_window_count = current_windows
                    self.ui_update_job = self.after(2000, self.poll_backend_results)
                    return

                latest = self.monitor.session_predictions[-1]
                state = latest.get('predicted', 'Unknown')
                conf = latest.get('confidence', 0) * 100

                if hasattr(self, 'lbl_previous_state'):
                    self.lbl_previous_state.configure(text=f"Sequence #{current_windows}: {state} ({conf:.1f}%)",
                                                      text_color="white")
                if hasattr(self, 'state_dropdown'):
                    self.state_dropdown.set(state)

                self.last_window_count = current_windows

            if getattr(self.monitor, 'waiting_for_correction', False):
                self.is_timer_paused = True
                if hasattr(self, 'btn_correct'):
                    self.btn_correct.configure(state="normal", text="Confirm & Resume")
                if hasattr(self, 'lbl_current_status'):
                    self.lbl_current_status.configure(
                        text=f"⏸️ Sequence #{current_windows} Paused for Supervisor Review", text_color="#F39C12")
                if hasattr(self, 'lbl_toast'):
                    self.lbl_toast.configure(text="Please verify the outcome below before proceeding.",
                                             text_color="#F39C12")
            else:
                if current_windows > 0 and hasattr(self, 'lbl_current_status'):
                    self.lbl_current_status.configure(
                        text=f"▶️ Gathering temporal data for Sequence #{current_windows + 1}...", text_color="gray")

            self.update_dashboard_stats()
            self.ui_update_job = self.after(2000, self.poll_backend_results)


    def update_smart_toasts(self, current_state):
        if self.consecutive_states["state"] == current_state:
            self.consecutive_states["count"] += 1
        else:
            self.consecutive_states["state"] = current_state
            self.consecutive_states["count"] = 1

        count = self.consecutive_states["count"]

        if current_state == "Distracted" and count >= 3:
            self.lbl_toast.configure(text="💡 Notice: Productivity dipping. Consider a 5-min stretch break.", text_color="#E67E22")
        elif current_state == "Deep Work" and count >= 5:
            self.lbl_toast.configure(text="🔥 Excellent focus! You are in the zone.", text_color="#2ECC71")
        elif current_state == "Idle" and count >= 3:
            self.lbl_toast.configure(text="💤 System idle detected. Timer is running.", text_color="#95A5A6")
        else:
            if not self.monitor.supervised_mode:
                self.lbl_toast.configure(text="")

    def update_timer(self):
        # --- BULLETPROOF KILL SWITCH ---
        if not self.winfo_exists():
            return

        if self.is_timer_running and not self.is_timer_paused:
            hours, remainder = divmod(self.timer_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            self.lbl_timer.configure(text=f"{hours:02d}:{minutes:02d}:{seconds:02d}")
            self.timer_seconds += 1

        self.timer_job = self.after(1000, self.update_timer)

    def update_dashboard_stats(self):
        sessions = getattr(self.monitor.user_profile, 'session_count', 0)
        evaluated_minutes = getattr(self.monitor.user_profile, 'total_windows', 0)
        total_corrections = getattr(self.monitor.user_profile, 'total_corrections', 0)

        prod_score = max(100 - ((total_corrections / evaluated_minutes) * 100), 0) if evaluated_minutes > 0 else 100.0

        self.lbl_profile_sessions.configure(text=f"Total Recorded Sessions: {sessions}")
        self.lbl_profile_time.configure(text=f"Total Evaluated Windows: {evaluated_minutes}")
        self.lbl_productivity.configure(text=f"System Confidence: {prod_score:.1f}%")

    def generate_analytics_chart(self):
        """Generate a dual-plot Rich Analytics Dashboard"""
        for widget in self.chart_frame.winfo_children():
            widget.destroy()

        if len(self.monitor.session_predictions) == 0:
            ctk.CTkLabel(self.chart_frame, text="Not enough data to generate analytics.", text_color="gray").pack(
                expand=True)
            return

        # Prepare Data
        dist = {}
        windows = []
        confidences = []

        for p in self.monitor.session_predictions:
            dist[p['predicted']] = dist.get(p['predicted'], 0) + 1
            windows.append(p['window'])
            confidences.append(p['confidence'] * 100)

        labels = list(dist.keys())
        sizes = list(dist.values())
        colors = ['#3498DB', '#2ECC71', '#9B59B6', '#F1C40F', '#E74C3C', '#95A5A6']

        # Create Dual-Plot Figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), facecolor='#2B2B2B')

        # Plot 1: Pie Chart (Distribution)
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, textprops={'color': "w"})
        ax1.set_title('Cognitive State Distribution', color='white', fontsize=12)

        # Plot 2: Line Graph (Confidence Trend)
        ax2.plot(windows, confidences, color='#2ECC71', marker='o', linewidth=2)
        ax2.set_facecolor('#2B2B2B')
        ax2.set_title('Model Confidence Trend', color='white', fontsize=12)
        ax2.set_xlabel('Rolling Sequence Window', color='white')
        ax2.set_ylabel('Confidence (%)', color='white')
        ax2.set_ylim(0, 105)
        ax2.tick_params(colors='white')

        for spine in ax2.spines.values():
            spine.set_color('#3A3A3A')

        fig.tight_layout()

        # Embed into Tkinter
        self.chart_canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        self.chart_canvas.draw()
        self.chart_canvas.get_tk_widget().pack(fill="both", expand=True, pady=10)

        # Add a rich text summary below the charts
        avg_conf = sum(confidences) / len(confidences) if confidences else 0
        summary_text = f"Session Finalized. Sequences Analyzed: {len(windows)} | Mean System Confidence: {avg_conf:.1f}%"
        ctk.CTkLabel(self.chart_frame, text=summary_text, font=ctk.CTkFont(size=14, weight="bold"),
                     text_color="#3498DB").pack()

    def populate_session_history(self):
        """Reads the JSON files to build the history feed with clean names"""
        for widget in self.history_scrollable.winfo_children():
            widget.destroy()

        try:
            import os
            import glob
            import json
            import customtkinter as ctk

            log_dir = "session_logs/live"
            if not os.path.exists(log_dir):
                ctk.CTkLabel(self.history_scrollable, text="No sessions recorded yet.").pack(pady=20)
                return

            files = sorted(glob.glob(f"{log_dir}/*.json"), reverse=True)[:5]
            if not files:
                ctk.CTkLabel(self.history_scrollable, text="No sessions recorded yet.").pack(pady=20)
                return

            for file in files:
                with open(file, 'r') as f:
                    data = json.load(f)

                # --- NEW: Parse the ugly filename into a clean timestamp ---
                raw_name = os.path.basename(file).replace('.json', '')
                parts = raw_name.split('_')

                clean_name = raw_name
                if len(parts) >= 3:
                    try:
                        date_str, time_str = parts[1], parts[2]
                        # Formats to: "YYYY-MM-DD at HH:MM"
                        clean_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
                        clean_time = f"{time_str[:2]}:{time_str[2:4]}"
                        clean_name = f"{clean_date} at {clean_time}"
                    except Exception:
                        pass  # Fallback to raw name if parsing fails

                # Check data length safely
                num_windows = len(data) if isinstance(data, list) else 0
                if num_windows == 0: continue

                card = ctk.CTkFrame(self.history_scrollable, corner_radius=8, border_width=1, border_color="#3A3A3A")
                card.pack(fill="x", pady=5, ipady=2)

                # The button now uses the clean_name
                btn = ctk.CTkButton(card, text=f"📄 Session: {clean_name}",
                                    fg_color="transparent", border_width=0, anchor="w",
                                    command=lambda f=file: self.show_session_summary(f))
                btn.pack(fill="x", padx=10, pady=5)

        except Exception as e:
            print(f"UI Notice: Could not load history ({e})")

    def show_session_summary(self, filepath):
        """Creates a popup window with safely abstracted session details"""
        try:
            import json
            import customtkinter as ctk

            with open(filepath, 'r') as f:
                raw_data = json.load(f)

            # --- NEW: Crash-Proof Data Extraction ---
            # Safely handle if an old file saved as a dict instead of a list
            data_list = raw_data if isinstance(raw_data, list) else raw_data.get('predictions', [])

            windows = len(data_list)
            if windows == 0:
                print("Notice: Clicked session file is empty.")
                return

            # Safely abstract the dominant state
            dist = {}
            for d in data_list:
                if isinstance(d, dict):
                    # Safely handles both 'predicted' (Live Mode) and 'predicted_state' (Test Mode)
                    state = d.get('predicted', d.get('predicted_state', 'Unknown'))
                    dist[state] = dist.get(state, 0) + 1

            top_state = max(dist, key=dist.get) if dist else "Unknown"

            # Create the popup window
            popup = ctk.CTkToplevel(self)
            popup.title("Session Abstract")
            popup.geometry("350x200")
            popup.attributes('-topmost', True)

            ctk.CTkLabel(popup, text="Session Summary", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=(15, 10))
            ctk.CTkLabel(popup, text=f"Total Evaluated Windows: {windows}", font=ctk.CTkFont(size=14)).pack(pady=5)
            ctk.CTkLabel(popup, text=f"Dominant State: {top_state}", font=ctk.CTkFont(size=14, weight="bold"),
                         text_color="#2ECC71").pack(pady=5)

            ctk.CTkButton(popup, text="Close Report", command=popup.destroy).pack(pady=15)

        except Exception as e:
            print(f"Error loading summary: {e}")

    def on_closing(self):
        """Absolute Kill Switch for graceful shutdown"""
        print("\nInitiating safe shutdown sequence...")
        self.is_timer_running = False

        if hasattr(self, 'monitor') and self.monitor:
            self.monitor.stop()

        self.quit()  # Stops the UI main loop
        self.destroy()  # Destroys the window

        # The Nuclear Option: Kills all Python background threads instantly
        import os
        print("Shutdown complete. Goodbye!")
        os._exit(0)

if __name__ == "__main__":
    app = WorkMonitorApp()
    app.mainloop()