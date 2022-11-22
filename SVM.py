from __future__ import annotations

import math
import tkinter
from tkinter.filedialog import askopenfilename, asksaveasfilename

import pygame
import numpy as np
from pygame.event import Event
from pygame.surface import Surface, SurfaceType
from queue import Queue

from sklearn.svm import LinearSVC, SVC

screen: Surface | SurfaceType = None
is_editing_enabled = True
is_rmb_down = False
is_lmb_down = False
last_point_pos = None
event_handlers = dict()
keypress_handlers = dict()
points = []
test_points = []
is_auto = False
svm: LinearSVC = None
current_mouse_pos = None


def event_handler(event_type: int):
    def decorator(func):
        event_handlers[event_type] = func
        return func

    return decorator


def setup():
    global screen

    pygame.init()
    screen = pygame.display.set_mode([800, 400])
    screen.fill(color='white')

    pygame.display.update()
    while True:
        loop()


def loop():
    for event in pygame.event.get():
        if event.type in event_handlers:
            print('Handling event: %s' % event)
            event_handlers[event.type](event)
    pygame.display.update()


@event_handler(pygame.QUIT)
def quit_handler(event: Event):
    pygame.quit()
    exit()


def add_point(pos, cls, is_test=False):
    if not is_editing_enabled:
        return
    p = [*np.array(pos), cls]
    pygame.draw.circle(screen, 'red' if cls == 0 else ('blue' if cls == 1 else 'gray'), pos, 4)
    if not is_test:
        points.append(p)
    else:
        test_points.append(p)
        pygame.draw.circle(screen, 'yellow', pos, 2)

    if is_auto:
        fit_svm()


@event_handler(pygame.MOUSEBUTTONDOWN)
def mouse_down_handler(event: Event):
    global is_lmb_down, is_rmb_down, last_point_pos

    if event.button == pygame.BUTTON_LEFT:
        add_point(event.pos, 0)
        is_lmb_down = True
        last_point_pos = np.array(event.pos)
    if event.button == pygame.BUTTON_RIGHT:
        add_point(event.pos, 1)
        is_rmb_down = True
        last_point_pos = np.array(event.pos)
    if event.button == pygame.BUTTON_MIDDLE:
        if svm is None:
            cls = -1
        else:
            cls = svm.predict(np.array([event.pos]))[0]
        add_point(event.pos, cls, True)


@event_handler(pygame.MOUSEBUTTONUP)
def mouse_up_handler(event: Event):
    global is_lmb_down, is_rmb_down, last_point_pos

    if event.button == pygame.BUTTON_LEFT:
        is_lmb_down = False
        last_point_pos = None
    if event.button == pygame.BUTTON_RIGHT:
        is_rmb_down = False
        last_point_pos = None


@event_handler(pygame.MOUSEMOTION)
def mouse_motion_handler(event: Event):
    global last_point_pos, current_mouse_pos
    current_mouse_pos = event.pos

    if is_lmb_down:
        if last_point_pos is None or np.sum((event.pos - last_point_pos) ** 2) > 125:
            add_point(event.pos, 0)
            last_point_pos = np.array(event.pos)
    elif is_rmb_down:
        if last_point_pos is None or np.sum((event.pos - last_point_pos) ** 2) > 125:
            add_point(event.pos, 1)
            last_point_pos = np.array(event.pos)


@event_handler(pygame.KEYDOWN)
def keypress_global_handler(event: Event):
    if event.key in keypress_handlers:
        keypress_handlers[event.key](event)


def keypress_handler(key_type: int):
    def decorator(func):
        keypress_handlers[key_type] = func
        return func

    return decorator


def clear_points():
    global points, test_points, svm
    if not is_editing_enabled:
        return

    points = []
    test_points = []
    svm = None
    screen.fill(color='white')


def redraw_points():
    screen.fill(color='white')
    for x, y, cls in points:
        pygame.draw.circle(screen, 'red' if cls == 0 else 'blue', (x, y), 4)
    for x, y, cls in test_points:
        pygame.draw.circle(screen, 'red' if cls == 0 else ('blue' if cls == 1 else 'gray'), (x, y), 4)
        pygame.draw.circle(screen, 'yellow', (x, y), 2)


@keypress_handler(pygame.K_ESCAPE)
def escape_press_handler(event: Event):
    clear_points()


@keypress_handler(pygame.K_a)
def a_press_handler(event: Event):
    global is_auto
    is_auto = not is_auto


@keypress_handler(pygame.K_t)
def t_press_handler(event: Event):
    if current_mouse_pos is None:
        return

    if svm is None:
        cls = -1
    else:
        cls = svm.predict(np.array([current_mouse_pos]))[0]
    add_point(current_mouse_pos, cls, True)


@keypress_handler(pygame.K_RETURN)
def enter_press_handler(event: Event):
    fit_svm()


def fit_svm():
    global svm, test_points
    try:
        svm = SVC(kernel='linear', C=100)
        points_arr = np.array(points)
        X, y = points_arr[:, :2], points_arr[:, 2]
        svm.fit(X, y)

        new_test_points = []
        for x, y, _ in test_points:
            cls = svm.predict(np.array([(x, y)]))[0]
            new_test_points.append([x, y, cls])
        test_points = new_test_points

        w = svm.coef_[0]
        b = svm.intercept_[0]
        x_points = np.array([0, screen.get_width()])
        y_points = -(w[0] / w[1]) * x_points - b / w[1]

        w_hat = svm.coef_[0] / (np.sqrt(np.sum(svm.coef_[0] ** 2)))
        margin = 1 / np.sqrt(np.sum(svm.coef_[0] ** 2))
        decision_boundary_points = np.array(list(zip(x_points, y_points)))
        points_of_line_above = decision_boundary_points + w_hat * margin
        points_of_line_below = decision_boundary_points - w_hat * margin

        redraw_points()
        pygame.draw.line(screen, 'black', (x_points[0], y_points[0]), (x_points[1], y_points[1]), 4)
        pygame.draw.line(screen, (100, 100, 100),
                         (points_of_line_above[0][0], points_of_line_above[0][1]),
                         (points_of_line_above[1][0], points_of_line_above[1][1]))
        pygame.draw.line(screen, (100, 100, 100),
                         (points_of_line_below[0][0], points_of_line_below[0][1]),
                         (points_of_line_below[1][0], points_of_line_below[1][1]))
    except:
        pass


if __name__ == '__main__':
    setup()