"""
Modern UI Dashboard for Intent-Aware Work Monitoring System
Separated UI logic using CustomTkinter
"""

import customtkinter as ctk
import threading
import time
from datetime import datetime
from complete_live_monitor import CompleteLiveMonitor

# Set modern appearance
ctk.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"


class WorkMonitorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window config
        self.title("Intent-Aware Monitor Dashboard")
        self.geometry("900x600")
        self.resizable(False, False)

        # Initialize core logic (Backend)
        self.user_id = "user_hari"
        self.monitor = CompleteLiveMonitor(user_id=self.user_id)
        self.monitor_thread = None
        self.ui_update_job = None
        self.timer_seconds = 0
        self.is_timer_running = False # NEW: Decoupled UI timer flag

        self.setup_ui()
        self.update_dashboard_stats()

    def setup_ui(self):
        """Build the UI layout"""
        # --- Grid Layout ---
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # --- Sidebar Navigation ---
        self.sidebar_frame = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Work Monitor",
                                       font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.btn_dashboard = ctk.CTkButton(self.sidebar_frame, text="Dashboard",
                                           command=lambda: self.select_frame("dashboard"))
        self.btn_dashboard.grid(row=1, column=0, padx=20, pady=10)

        self.btn_monitor = ctk.CTkButton(self.sidebar_frame, text="Live Monitor",
                                         command=lambda: self.select_frame("monitor"))
        self.btn_monitor.grid(row=2, column=0, padx=20, pady=10)

        self.btn_results = ctk.CTkButton(self.sidebar_frame, text="Results",
                                         command=lambda: self.select_frame("results"))
        self.btn_results.grid(row=3, column=0, padx=20, pady=10)

        # --- Main Content Area ---
        self.main_frame = ctk.CTkFrame(self, corner_radius=10, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)

        self.frames = {}
        self.setup_dashboard_frame()
        self.setup_monitor_frame()
        self.setup_results_frame()

        # Start on Dashboard
        self.select_frame("dashboard")

    def setup_dashboard_frame(self):
        frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.frames["dashboard"] = frame

        ctk.CTkLabel(frame, text="User Profile", font=ctk.CTkFont(size=24, weight="bold")).pack(pady=(0, 20),
                                                                                                anchor="w")

        self.lbl_profile_user = ctk.CTkLabel(frame, text=f"User ID: {self.user_id}", font=ctk.CTkFont(size=16))
        self.lbl_profile_user.pack(anchor="w", pady=5)

        self.lbl_profile_sessions = ctk.CTkLabel(frame, text="Total Sessions: Loading...", font=ctk.CTkFont(size=16))
        self.lbl_profile_sessions.pack(anchor="w", pady=5)

        # NEW: Total Monitored Time Label
        self.lbl_profile_time = ctk.CTkLabel(frame, text="Total Monitored Time: Loading...", font=ctk.CTkFont(size=16))
        self.lbl_profile_time.pack(anchor="w", pady=5)

        self.lbl_profile_baseline = ctk.CTkLabel(frame, text="Baseline Status: Loading...", font=ctk.CTkFont(size=16))
        self.lbl_profile_baseline.pack(anchor="w", pady=5)

    def setup_monitor_frame(self):
        frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.frames["monitor"] = frame

        ctk.CTkLabel(frame, text="Live Monitoring", font=ctk.CTkFont(size=24, weight="bold")).pack(pady=(0, 20))

        # Big Timer
        self.lbl_timer = ctk.CTkLabel(frame, text="00:00:00", font=ctk.CTkFont(size=60, weight="bold"),
                                      text_color="#1f6aa5")
        self.lbl_timer.pack(pady=20)

        # Controls
        control_frame = ctk.CTkFrame(frame, fg_color="transparent")
        control_frame.pack(pady=10)

        self.btn_start = ctk.CTkButton(control_frame, text="START", command=self.start_monitoring, fg_color="green",
                                       hover_color="darkgreen")
        self.btn_start.pack(side="left", padx=10)

        self.btn_stop = ctk.CTkButton(control_frame, text="STOP", command=self.stop_monitoring, fg_color="red",
                                      hover_color="darkred", state="disabled")
        self.btn_stop.pack(side="left", padx=10)

        # Current Status Card
        self.status_card = ctk.CTkFrame(frame, corner_radius=10)
        self.status_card.pack(pady=30, fill="x", padx=50)

        self.lbl_current_state = ctk.CTkLabel(self.status_card, text="Waiting for data (5 min warmup)...",
                                              font=ctk.CTkFont(size=18))
        self.lbl_current_state.pack(pady=20)

    def setup_results_frame(self):
        frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.frames["results"] = frame

        ctk.CTkLabel(frame, text="Session Abstract", font=ctk.CTkFont(size=24, weight="bold")).pack(pady=(0, 20),
                                                                                                    anchor="w")

        self.results_textbox = ctk.CTkTextbox(frame, width=600, height=400, font=ctk.CTkFont(size=14))
        self.results_textbox.pack(pady=10, fill="both", expand=True)
        self.results_textbox.insert("0.0", "Results will populate here during and after the session.\n\n")
        self.results_textbox.configure(state="disabled")

    def select_frame(self, frame_name):
        for name, frame in self.frames.items():
            if name == frame_name:
                frame.grid(row=0, column=0, sticky="nsew")
            else:
                frame.grid_forget()

    def update_dashboard_stats(self):
        """Fetch data from the backend profile to populate UI"""
        sessions = self.monitor.user_profile.session_count
        windows = self.monitor.user_profile.total_windows  # 1 window = 1 minute
        has_baseline = "Established" if self.monitor.user_profile.baseline else "Building..."

        self.lbl_profile_sessions.configure(text=f"Total Sessions: {sessions}")
        self.lbl_profile_time.configure(text=f"Total Monitored Time: {windows} minutes")  # NEW
        self.lbl_profile_baseline.configure(text=f"Baseline Status: {has_baseline}")

    # --- Interaction Logic ---

    def start_monitoring(self):
        if self.monitor.model is None:
            self.lbl_current_state.configure(text="Error: Model not found. Train first.", text_color="red")
            return

        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.lbl_current_state.configure(text="Collecting data...", text_color="white")

        # Start Timer safely
        self.is_timer_running = True
        self.timer_seconds = 0
        self.update_timer()

        # Run backend monitor in a separate thread
        self.monitor_thread = threading.Thread(target=self.monitor.start, daemon=True)
        self.monitor_thread.start()

        self.poll_backend_results()

        # Start polling backend for updates
        self.poll_backend_results()

    def stop_monitoring(self):
        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self.lbl_current_state.configure(text="Monitoring Stopped.")

        self.is_timer_running = False  # Stop UI timer safely
        self.monitor.stop()  # Stop backend

        if self.ui_update_job:
            self.after_cancel(self.ui_update_job)

    def update_timer(self):
        """Update the UI clock every second"""
        if self.is_timer_running: # Fixed race condition
            hours, remainder = divmod(self.timer_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            self.lbl_timer.configure(text=f"{hours:02d}:{minutes:02d}:{seconds:02d}")
            self.timer_seconds += 1
            self.after(1000, self.update_timer)

    def poll_backend_results(self):
        """Safely fetch prediction data from the backend thread"""
        if self.is_timer_running:
            if len(self.monitor.session_predictions) > 0:
                latest = self.monitor.session_predictions[-1]
                state = latest['predicted']
                conf = latest['confidence'] * 100

                self.lbl_current_state.configure(text=f"State: {state} ({conf:.1f}%)")

                self.results_textbox.configure(state="normal")
                self.results_textbox.delete("0.0", "end")

                # Format current session elapsed time
                session_mins = self.timer_seconds // 60

                summary = "LIVE SESSION ABSTRACT\n" + "-" * 30 + "\n\n"
                summary += f"Session Elapsed Time: {session_mins} min {self.timer_seconds % 60} sec\n"  # NEW
                summary += f"Windows Processed: {len(self.monitor.session_predictions)}\n"
                summary += f"Latest State: {state} (Confidence: {conf:.1f}%)\n\n"

                dist = {}
                for p in self.monitor.session_predictions:
                    dist[p['predicted']] = dist.get(p['predicted'], 0) + 1

                summary += "State Distribution:\n"
                for k, v in dist.items():
                    summary += f"• {k}: {v} windows\n"

                self.results_textbox.insert("0.0", summary)
                self.results_textbox.configure(state="disabled")

            self.ui_update_job = self.after(2000, self.poll_backend_results)
        else:
            self.update_dashboard_stats()


if __name__ == "__main__":
    app = WorkMonitorApp()
    app.mainloop()