import flet as ft
from flet import Page
import complete_live_monitor as clm
from complete_live_monitor import CompleteLiveMonitor as cl


def main(page: Page) -> None:
    page.title = 'Focus Flow'
    page.horizontal_alignment = 'center'
    page.vertical_alignment = 'center'
    page.window.width = '800'
    page.window.height = '800'




    def live_monitor(e)->None:
        clm.start_live_monitor()

    def stop_live_monitor(e):
        cl._end_session()


    home_view=ft.Row(
        [
            ft.Text("hello"),
            ft.Button(text="Start",on_click=live_monitor),
            ft.Button(text="Stop", on_click=stop_live_monitor)

    ],alignment=ft.MainAxisAlignment.CENTER
    )



    page.add(home_view)


if __name__ == '__main__':
    ft.app(target=main)