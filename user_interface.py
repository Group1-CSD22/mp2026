import flet as ft
from flet import Page

def main(page: Page) -> None:
    page.title = 'Focus Flow'
    page.horizontal_alignment = 'center'
    page.vertical_alignment = 'center'
    page.window.width = '800'
    page.window.height = '800'

    home_view=ft.Text("hello")



    page.add(home_view)


if __name__ == '__main__':
    ft.app(target=main)