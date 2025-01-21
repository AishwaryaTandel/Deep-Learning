from src_to_implement.generator import ImageGenerator
from src_to_implement.pattern import Checker, Circle, Spectrum


def main():
    checker = Checker(50,5)
    checker.show()

    circle = Circle(400, 80, (200,100))
    circle.show()

    spectrum = Spectrum(100)
    spectrum.show()

    img = ImageGenerator('./exercise_data/'.strip(), './Labels.json'.strip(), 12,
                         [32, 32, 3], rotation=False, mirroring=False, shuffle=False)
    img.next()
    img.show()

if __name__ == '__main__':
    main()