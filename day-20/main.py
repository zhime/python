from turtle import Turtle, Screen
import random


t = Turtle()
t.screen.colormode(255)

def choice_color():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return r, g, b

for _ in range(4):
    t.color(choice_color())
    t.forward(100)
    t.left(90)
#
screen = Screen()
screen.exitonclick()
