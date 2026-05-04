"""
FINAL PRODUCT UI: Intent-Aware Work Monitoring System
Features: Supervised Mode, Smart Toasts, Embedded Analytics
"""

import customtkinter as ctk
import threading
import time
from datetime import datetime
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

        ctk.CTkLabel(frame, text="System Overview", font=ctk.CTkFont(size=28, weight="bold")).pack(pady=(0, 30), anchor="w")

        stats_frame = ctk.CTkFrame(frame, corner_radius=10)
        stats_frame.pack(fill="x", pady=10, ipady=20)

        self.lbl_profile_sessions = ctk.CTkLabel(stats_frame, text="Total Sessions: 0", font=ctk.CTkFont(size=18))
        self.lbl_profile_sessions.pack(pady=10)

        self.lbl_profile_time = ctk.CTkLabel(stats_frame, text="Evaluated Windows: 0", font=ctk.CTkFont(size=18))
        self.lbl_profile_time.pack(pady=10)

        self.lbl_profile_baseline = ctk.CTkLabel(stats_frame, text="Baseline: Building", font=ctk.CTkFont(size=18))
        self.lbl_profile_baseline.pack(pady=10)

    def setup_monitor_frame(self):
        frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.frames["monitor"] = frame

        ctk.CTkLabel(frame, text="Live Activity Tracker", font=ctk.CTkFont(size=28, weight="bold")).pack(pady=(0, 10))

        # Timer
        self.lbl_timer = ctk.CTkLabel(frame, text="00:00:00", font=ctk.CTkFont(size=65, weight="bold"), text_color="#2FA572")
        self.lbl_timer.pack(pady=10)

        # Start/Stop Controls
        control_frame = ctk.CTkFrame(frame, fg_color="transparent")
        control_frame.pack(pady=10)
        self.btn_start = ctk.CTkButton(control_frame, text="▶ START", command=self.start_monitoring, width=120, fg_color="#2FA572", hover_color="#1E7A52")
        self.btn_start.pack(side="left", padx=10)
        self.btn_stop = ctk.CTkButton(control_frame, text="■ STOP", command=self.stop_monitoring, width=120, fg_color="#E03B3B", hover_color="#A82A2A", state="disabled")
        self.btn_stop.pack(side="left", padx=10)

        # State Card
        self.status_card = ctk.CTkFrame(frame, corner_radius=15)
        self.status_card.pack(pady=20, fill="x", padx=40, ipady=20)

        self.lbl_current_state = ctk.CTkLabel(self.status_card, text="Waiting to start...", font=ctk.CTkFont(size=24, weight="bold"))
        self.lbl_current_state.pack(pady=10)

        # Smart Toast Area
        self.lbl_toast = ctk.CTkLabel(self.status_card, text="", font=ctk.CTkFont(size=14, slant="italic"), text_color="#FFA500")
        self.lbl_toast.pack()

        # Supervised Mode Correction Panel (Hidden by default)
        self.correction_frame = ctk.CTkFrame(frame, fg_color="transparent")

        ctk.CTkLabel(self.correction_frame, text="Supervised Override:", font=ctk.CTkFont(size=14)).pack(side="left", padx=10)
        self.state_dropdown = ctk.CTkOptionMenu(self.correction_frame, values=["Deep Work", "Active Work", "Research", "Communication", "Distracted", "Idle"])
        self.state_dropdown.pack(side="left", padx=10)

        self.btn_correct = ctk.CTkButton(self.correction_frame, text="Apply Correction", command=self.submit_correction, fg_color="#F39C12", hover_color="#D68910")
        self.btn_correct.pack(side="left", padx=10)

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
        self.lbl_current_state.configure(text="Collecting data (5 min warmup)...", text_color="white")
        self.monitor.session_predictions = []

        self.is_timer_running = True
        self.timer_seconds = 0
        self.update_timer()

        self.monitor_thread = threading.Thread(target=self.monitor.start, daemon=True)
        self.monitor_thread.start()
        self.poll_backend_results()

    def stop_monitoring(self):
        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self.lbl_current_state.configure(text="Monitoring Stopped.")
        self.lbl_toast.configure(text="")

        self.is_timer_running = False
        self.monitor.stop()

        if self.ui_update_job:
            self.after_cancel(self.ui_update_job)

        if len(self.monitor.session_predictions) > 0:
            self.generate_analytics_chart()

        self.after(500, self.update_dashboard_stats)

    def submit_correction(self):
        """Send manual correction to backend model and unpause"""
        selected_state = self.state_dropdown.get()
        success = self.monitor.apply_ui_correction(selected_state)

        # Unpause everything immediately for crisp UI feel
        self.is_timer_paused = False
        self.btn_correct.configure(state="disabled", text="Waiting...")
        self.lbl_current_state.configure(text=f"Live State: {selected_state} (Confirmed)")

        if self.monitor.latest_correction_data['predicted_state'] != selected_state:
            buffer_size = len(self.monitor.online_learner.memory)
            self.lbl_toast.configure(text=f"▶️ Resuming... (Correction {buffer_size}/10 saved)", text_color="#2ECC71")
        else:
            self.lbl_toast.configure(text="▶️ Resuming...", text_color="#3498DB")

    def poll_backend_results(self):
        if not self.winfo_exists():
            return

        if self.is_timer_running:
            current_windows = len(self.monitor.session_predictions)

            # 1. Update UI if we have a new window
            if current_windows > self.last_window_count:
                latest = self.monitor.session_predictions[-1]
                state = latest['predicted']
                conf = latest['confidence'] * 100

                self.lbl_current_state.configure(text=f"Live State: {state} ({conf:.1f}%)")
                self.state_dropdown.set(state)
                self.update_smart_toasts(state)
                self.last_window_count = current_windows

            # 2. Handle Pausing visually
            if getattr(self.monitor, 'waiting_for_correction', False):
                self.is_timer_paused = True
                self.btn_correct.configure(state="normal", text="Confirm & Resume")
                self.lbl_toast.configure(text="⏸️ Timer paused. Please confirm or correct the state.",
                                         text_color="#F39C12")

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
        sessions = self.monitor.user_profile.session_count
        evaluated_minutes = self.monitor.user_profile.total_windows
        has_baseline = "Established" if self.monitor.user_profile.baseline else "Building Baseline..."

        self.lbl_profile_sessions.configure(text=f"Total Sessions: {sessions}")
        self.lbl_profile_time.configure(text=f"Evaluated Windows: {evaluated_minutes}")
        self.lbl_profile_baseline.configure(text=f"Baseline Status: {has_baseline}")

    def generate_analytics_chart(self):
        """Generate an attractive Matplotlib Pie Chart in the UI"""
        # Clear old chart
        for widget in self.chart_frame.winfo_children():
            widget.destroy()

        dist = {}
        for p in self.monitor.session_predictions:
            dist[p['predicted']] = dist.get(p['predicted'], 0) + 1

        labels = list(dist.keys())
        sizes = list(dist.values())

        # Color palette matching dark UI
        colors = ['#3498DB', '#2ECC71', '#9B59B6', '#F1C40F', '#E74C3C', '#95A5A6']

        fig, ax = plt.subplots(figsize=(6, 4), facecolor='#2B2B2B')
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, textprops={'color':"w"})
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        fig.suptitle('Session Productivity Distribution', color='white', fontsize=14)

        # Embed into Tkinter
        self.chart_canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        self.chart_canvas.draw()
        self.chart_canvas.get_tk_widget().pack(fill="both", expand=True)

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